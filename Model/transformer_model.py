import torch
import torch.nn as nn
from typing import Tuple, List
from dataclasses import dataclass
from torch.utils.data import Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    input_dims: Tuple[int, ...]  # Features from each embedding
    mode: str = '3m'      # Options: '1m', '3m', '6m'
    window_size: int = 6  # Size of input window
    dim_opt: int = 1      # Option: 1, 2 
    num_encoder_layers: int = 4
    nhead: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class EmbeddingTransformer(nn.Module):
    def __init__(self, config: ModelConfig, output_dim: int = 1):
        super().__init__()
        self.input_dims = config.input_dims
        self.dim_opt = config.dim_opt
        self.config = config
        self.prediction_months = {'1m': 1, '3m': 3, '6m': 6}[self.config.mode]
        self.output_dim = output_dim  # ✅ 저장
        
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
        
        self.input_dim = sum(self.embedding_dims)  # Concatenated feature dimension
        # Calculate total input dimension
        self.pos_encoder = PositionalEncoding(self.input_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True   # (batch_size, sequence_length, feature_dim) 순서로 변경 -> sequence_length: 한번에 입력되는 데이터 길이 = window_size
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(self.input_dim)
        )

        # Output Layer (예측 개수 설정)
        output_months = {"1m": 1, "3m": 3, "6m": 6}[config.mode]
        # (4) Output Layer: 라벨별 task-specific Linear layer 생성
        self.fc_outs = nn.ModuleList([
            nn.Linear(self.input_dim, output_months) for _ in range(self.output_dim)
        ])


    def _log_tensor_stats(self, tensor: torch.Tensor, name: str):
        """Helper function to log tensor statistics"""
        with torch.no_grad():
            logger.debug(f"{name} stats - Shape: {tensor.shape}, "
                      f"Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}], "
                      f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:    # 추가 옵션: airbnb_counts: torch.Tensor
        """
        Args:
            *embeddings: 도로 네트워크, 생활인구, Airbnb 특성, 라벨 임베딩 (4가지)
            airbnb_counts: 동별 Airbnb 개수 (shape: [batch_size, window_size, n_dongs, 1])
        Returns:
            torch.Tensor: 예측값 (shape: [batch_size, prediction_months, n_dongs, 1])
        """
        if len(embeddings) != len(self.embedding_networks):
            raise ValueError(f"Expected {len(self.embedding_networks)} embeddings, got {len(embeddings)}")
        
        # Process all sliding windows in the batch
        batch_size = embeddings[0].size(0)
        n_dongs = embeddings[0].size(2)
        
        # 슬라이딩 윈도우 적용 여부
        #is_sliding_window = (embeddings[0].size(1) == self.config.window_size)
        
        # 각 임베딩을 지정한 차원으로 변환
        features = []
        for emb, network, target_dim in zip(embeddings, self.embedding_networks, self.embedding_dims):
            emb_flat = emb.reshape(-1, emb.size(-1))  # 2차원 변환 -> [batch_size * window_size * n_dongs, n_features]
            transformed = network(emb_flat)           # target_dim으로 feature 변환: [batch_size * window_size * n_dongs, target_dim]
            transformed = transformed.reshape(batch_size, -1, n_dongs, target_dim)  # Reshape back: [batch_size, window_size, n_dongs, target_dim]
            features.append(transformed)
        
        # Concatenate all features
        x = torch.cat(features, dim=-1)  # [batch_size, window_size, n_dongs, self.input_dim] -> 마지막 차원을 기준으로 합침
        self._log_tensor_stats(x, "Concatenated features")
        
        # Apply transformer
        sequence_length = x.size(1)  # window_size

        x = x.permute(1, 0, 2, 3).reshape(sequence_length, batch_size * n_dongs, -1)  # transformer 입력 형태(sequence_length, batch_size, feature_dim)로 변환
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)                                               # Transformer 출력 형태(sequence_length, batch_size, feature_dim)
        x = x.permute(1, 0, 2)  # (batch_size * n_dongs, sequence_length, feature_dim)

        # prediction_months 만큼 예측
        x = x[:, -1, :]       # shape: [batch*n_dongs, input_dim]
        
        # ================= AirBnB Count 추가 방식 ===========================
        # airbnb_counts = airbnb_counts[:, -1, :, 0]  # [batch, n_dongs]
        # airbnb_counts = airbnb_counts.reshape(batch_size * n_dongs, 1)  # [batch_size * n_dongs, 1]
        
        # # 최종 FC_layer에 AirBnB 개수를 추가
        # x = torch.cat([x, airbnb_counts], dim=-1)  # [batch_size * n_dongs, input_dim + 1]
        # ====================================================================
        # out = self.fc_out(x)  # [batch_size * n_dongs, prediction_months]
        # out = out.unsqueeze(-1)
        # #logger.info(f"out shape: {out.shape}")
        
        # # Reshape to final output format
        # out = out.reshape(batch_size, n_dongs, self.prediction_months, self.output_dim)   # 1개 변수 예측으로 변경
        # out = out.permute(0, 2, 1, 3)  # [batch_size, prediction_months, n_dongs, 1]
        # 라벨별로 fc_outs[i] 적용
        task_outputs = []
        for i in range(self.output_dim):
            task_out = self.fc_outs[i](x)  # [batch_size * n_dongs, prediction_months]
            task_outputs.append(task_out)

        # (batch_size * n_dongs, prediction_months, output_dim)
        out = torch.stack(task_outputs, dim=-1)
        out = out.view(batch_size, n_dongs, self.prediction_months, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # [batch_size, prediction_months, n_dongs, output_dim]

        
        self._log_tensor_stats(out, "Final output")
        return out