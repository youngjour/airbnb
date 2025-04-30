import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RNNModelConfig:
    """
    RNN 관련 하이퍼파라미터 설정 클래스
    """
    input_dims: Tuple[int, ...]  # 각 임베딩별 feature 수 (예: (36, 36, 64, 3) 등)
    mode: str = '3m'             # '1m', '3m', '6m' 등
    window_size: int = 6         # 시계열 입력 윈도우 크기
    dim_opt: int = 1             # 사용자 임베딩 차원 옵션
    hidden_size: int = 64        # RNN hidden 크기
    num_layers: int = 1          # RNN layer 수
    dropout: float = 0.1
    bidirectional: bool = False  # 양방향 RNN 사용 여부

class EmbeddingRNN(nn.Module):
    """
    여러 Embedding(road/human flow/airbnb/...)을 받아서 Concatenate 한 뒤,
    단순 RNN(nn.RNN)으로 시계열 예측을 수행하는 모델 예시
    """
    def __init__(self, config: RNNModelConfig, output_dim: int = 1):
        super().__init__()
        self.config = config
        self.prediction_months = {'1m': 1, '3m': 3, '6m': 6}[config.mode]
        self.output_dim = output_dim

        # (1) 임베딩 차원 변환 레이어
        num_inputs = len(config.input_dims)
        # print(f"dim_opt: {config.dim_opt == 1} , num_inputs: {num_inputs}")
        if config.dim_opt == 1 and num_inputs==4:
            self.embedding_dims = [48, 48, 64, 4]
        elif config.dim_opt == 2 and num_inputs==4:
            self.embedding_dims = [48, 48, 128, 4]
        elif config.dim_opt == 3 and num_inputs==4:
            self.embedding_dims = [48, 64, 128, 4]
        elif config.dim_opt == 4 and num_inputs==4:
            self.embedding_dims = [64, 48, 64, 4]
        elif num_inputs==2:   # Raw 임베딩
            self.embedding_dims = [128, 4]
        elif num_inputs==3:   # Raw 임베딩
            self.embedding_dims = [48, 128, 4]

        # (2) 각 embedding 별로 Linear를 거쳐 원하는 차원으로 변환
        self.embedding_networks = nn.ModuleList()
        for in_dim, out_dim in zip(config.input_dims, self.embedding_dims):
            if in_dim >= 1024: # LLM 임베딩(3072)이 들어오는 경우
                dim1 = 768
                dim2 = 256
                dim3 = 128
                net = nn.Sequential(
                    nn.Linear(in_dim, dim1),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(dim1, dim2),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(dim2, dim3),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(dim3, out_dim),
                )
            else:
                net = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout)
                )
            self.embedding_networks.append(net)

        # 최종 feature 차원
        self.total_feature_dim = sum(self.embedding_dims)
        logger.info(f"Total feature dimension after embedding: {self.total_feature_dim}")

        # (2) RNN 정의
        self.rnn = nn.RNN(
            input_size=self.total_feature_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,     # (batch, seq, feature) 형태
            bidirectional=config.bidirectional
        )
        rnn_out_dim = config.hidden_size * (2 if config.bidirectional else 1)

        # (3) 출력 레이어
        #     예측해야 할 horizon 개수(예: 3개월)만큼 스칼라를 예측
        self.fc_outs = nn.ModuleList([
                    nn.Linear(rnn_out_dim, self.prediction_months) for _ in range(self.output_dim)])     # 3개의 변수를 각각 예측

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, window_size, n_dongs, n_features) 형태 여러 개
        Returns:
            out: (batch, prediction_months, n_dongs, 1)
        """
        batch_size = embeddings[0].size(0)
        window_size = embeddings[0].size(1)
        n_dongs = embeddings[0].size(2)

        # (1) 입력 임베딩 변환 & Concatenate
        processed_list = []
        for emb, net in zip(embeddings, self.embedding_networks):
            # emb shape: (batch, window_size, n_dongs, feat)
            emb_2d = emb.view(-1, emb.size(-1))   # (batch*window_size*n_dongs, feat)
            emb_processed = net(emb_2d)
            emb_processed = emb_processed.view(batch_size, window_size, n_dongs, -1)
            processed_list.append(emb_processed)

        x = torch.cat(processed_list, dim=-1)  # (batch, window_size, n_dongs, total_feature_dim)

        # (2) RNN에 넣기 위해 (batch*n_dongs, window_size, total_feature_dim)로 변형
        x = x.permute(0, 2, 1, 3)  # (batch, n_dongs, window_size, feat)
        x = x.reshape(batch_size * n_dongs, window_size, -1)  # (batch*n_dongs, window_size, total_feat)

        # (3) RNN Forward
        rnn_out, h_n = self.rnn(x)
        # rnn_out shape: (batch*n_dongs, window_size, hidden_size*(2)?)
        # h_n shape: (num_layers*(2?), batch*n_dongs, hidden_size)

        # 마지막 시점의 hidden state만 사용(단방향 RNN)
        if self.config.bidirectional:
            # bidirectional RNN일 때는 fw/bw 두 개 마지막 hidden concat
            # h_n[-2,:,:], h_n[-1,:,:]
            # h_n shape: (2*num_layers, batch*n_dongs, hidden_size)
            # 맨 위 레이어의 fw/bw를 concat
            fw = h_n[-2,:,:]
            bw = h_n[-1,:,:]
            h_last = torch.cat([fw, bw], dim=-1)  # (batch*n_dongs, hidden_size*2)
        else:
            # h_n[-1]: shape (batch*n_dongs, hidden_size)
            h_last = h_n[-1]

        # (4) 출력 레이어
        # out = self.fc_out(h_last)  # (batch*n_dongs, prediction_months)
        # out = out.view(batch_size, n_dongs, self.prediction_months, self.output_dim)
        # out = out.permute(0, 2, 1, 3)  # (batch, prediction_months, n_dongs, out_put_dim)
        
        task_outputs = []
        for i in range(self.output_dim):
            task_out = self.fc_outs[i](h_last)  # (batch*n_dongs, prediction_months)
            task_outputs.append(task_out)

        # task_outputs: List of (batch*n_dongs, prediction_months) → stack
        out = torch.stack(task_outputs, dim=-1)  # (batch*n_dongs, prediction_months, output_dim)
        out = out.view(batch_size, n_dongs, self.prediction_months, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # (batch, prediction_months, n_dongs, output_dim)

        return out
