import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LSTMModelConfig:
    """
    LSTM 관련 하이퍼파라미터 설정 클래스
    (기존 ModelConfig와 별도로 분리해두었습니다.)
    """
    input_dims: Tuple[int, ...]  # 각 임베딩별 feature 수 (예: (36, 36, 64, 3) 등)
    mode: str = '3m'             # '1m', '3m', '6m' 등
    window_size: int = 6         # 시계열 입력으로 사용할 윈도우 크기
    dim_opt: int = 1             # 사용자의 임베딩 차원 옵션
    hidden_size: int = 64        # LSTM hidden 크기
    num_layers: int = 2          # LSTM layer 수
    dropout: float = 0.1
    bidirectional: bool = False  # 양방향 LSTM 사용 여부

class EmbeddingLSTM(nn.Module):
    """
    여러 Embedding(road/human flow/airbnb 등)을 받아서 Concatenate 한 뒤,
    LSTM으로 시계열 예측을 수행하는 모델 예시입니다.
    """
    def __init__(self, config: LSTMModelConfig, output_dim: int = 1):
        super().__init__()
        self.config = config
        self.prediction_months = {'1m': 1, '3m': 3, '6m': 6}[config.mode]
        self.output_dim = output_dim  # ✅ 저장
        
        # (1) 임베딩 차원 변환 레이어
        #     - 필요시, 사용자 정의대로 embedding_dims를 결정
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

        # (3) LSTM 정의
        self.lstm = nn.LSTM(
            input_size=self.total_feature_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        lstm_out_dim = config.hidden_size * (2 if config.bidirectional else 1)

        # (4) 출력 레이어: 예측해야 할 horizon 개수 (ex: 3개월)만큼의 출력을 위한 Linear
        self.fc_outs = nn.ModuleList([
                        nn.Linear(lstm_out_dim, self.prediction_months) for _ in range(self.output_dim)])   # 3개의 변수를 모두 예측

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, window_size, n_dongs, n_features) 형태가
                        여러 개 (road, human flow, airbnb, label 등)
        Returns:
            out: (batch, prediction_months, n_dongs, 1)
        """
        # 1) 각 임베딩을 지정된 Linear 통과 후 Concatenate
        batch_size = embeddings[0].size(0)
        window_size = embeddings[0].size(1)
        n_dongs = embeddings[0].size(2)

        processed_list = []
        for emb, net in zip(embeddings, self.embedding_networks):
            # (batch, window, n_dongs, feat) -> (batch * window * n_dongs, feat)
            emb_2d = emb.reshape(-1, emb.size(-1))
            emb_processed = net(emb_2d)
            emb_processed = emb_processed.reshape(batch_size, window_size, n_dongs, -1)
            processed_list.append(emb_processed)

        # Concatenate on the last dim
        x = torch.cat(processed_list, dim=-1)  # (batch, window_size, n_dongs, total_feature_dim)

        # 2) LSTM에 넣기 위해 (batch * n_dongs, window_size, total_feature_dim) 형태로 변형
        x = x.permute(0, 2, 1, 3)  # (batch, n_dongs, window_size, feature)
        x = x.reshape(batch_size * n_dongs, window_size, -1)  # (batch*n_dongs, window_size, total_feature_dim)

        # 3) LSTM Forward
        lstm_out, (h_n, c_n) = self.lstm(x)  
        # h_n shape: (num_layers * num_directions, batch*n_dongs, hidden_size)

        # 마지막 time-step의 hidden state만 가져옴
        # (num_layers*num_directions, batch*n_dongs, hidden_size) -> (batch*n_dongs, hidden_size * num_directions)
        if self.config.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        
        # 4) 출력 레이어 -> (batch*n_dongs, prediction_months)
        # out = self.fc_out(h_last)  # (batch*n_dongs, prediction_months)
        # # 5) (batch, n_dongs, prediction_months, 1) 로 reshape
        # out = out.view(batch_size, n_dongs, self.prediction_months, self.output_dim).permute(0, 2, 1, 3)
        
        # 라벨별 task-specific head 적용
        task_outputs = []
        for i in range(self.output_dim):
            task_out = self.fc_outs[i](h_last)  # (batch*n_dongs, prediction_months)
            task_outputs.append(task_out)

        # (batch*n_dongs, prediction_months, output_dim)
        out = torch.stack(task_outputs, dim=-1)
        out = out.view(batch_size, n_dongs, self.prediction_months, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # (batch, prediction_months, n_dongs, output_dim)

        return out
