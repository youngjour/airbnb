import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass
import os
from pathlib import Path
import json
from Model.transformer_model import EmbeddingTransformer, ModelConfig
from RNN_model import EmbeddingRNN, RNNModelConfig
from LSTM_model import EmbeddingLSTM, LSTMModelConfig

import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 7e-6
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    base_path: str = 'outputs'
    clip_grad_norm: float = 5.0
    scheduler_patience: int = 8
    use_multi_gpu: bool = False    # 멀티 GPU 사용 여부

def get_prediction_horizon(mode: str) -> int:
    """Get prediction horizon based on mode."""
    horizons = {'1m': 1, '3m': 3, '6m': 6}
    if mode not in horizons:
        raise ValueError(f"Invalid mode. Must be one of {list(horizons.keys())}")
    return horizons[mode]

# 로그 변환 함수
def safe_log1p(x):
    x = np.where(x<0, 0, x)
    return np.log1p(x)

# SlidingWindowDataset (BaseDataset 상속)
class SlidingWindowDataset(Dataset):
    """Sliding Window 기반 학습을 위한 Dataset"""
    def __init__(self, 
                 embeddings: List[np.ndarray],
                 labels: np.ndarray,
                 window_size: int,  # 추가 인자
                 prediction_horizon: int,
                 start_month: int,
                 end_month: int,
                 is_train: bool,
                 train_means: Optional[List[np.ndarray]] = None,
                 train_stds: Optional[List[np.ndarray]] = None,
                 label_means: Optional[np.ndarray] = None,
                 label_stds: Optional[np.ndarray] = None):
        embeddings[-1] = safe_log1p(embeddings[-1])  # 마지막 임베딩(라벨 임베딩과 동일한 값) 로그 변환 적용 (나머지 입력은 로그 변환 x)
        self.embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in embeddings]
        labels = safe_log1p(labels)                  # y값 로그 변환
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.prediction_horizon = prediction_horizon
        self.start_month = start_month
        self.end_month = end_month
        self.window_size = window_size
        
        # 슬라이딩 윈도우 범위 설정
        self.windows = []

        max_start = self.end_month - self.window_size - self.prediction_horizon
        for window_start in range(self.start_month, max_start + 1):
            window = {
                'input_start': window_start,
                'input_end': window_start + self.window_size,
                'target_start': window_start + self.window_size,
                'target_end': window_start + self.window_size + self.prediction_horizon
            }
            self.windows.append(window)
            
        if is_train:
            self.emb_means = []
            self.emb_stds = []

            all_train_data = [emb[start_month:end_month] for emb in embeddings]

            for i, emb in enumerate(all_train_data):
                #emb_mean = np.mean(emb, axis=0)        # 동별 정규화 (동 개수, feature 개수)
                #emb_std = np.std(emb, axis=0)
                emb_mean = np.mean(emb, axis=(0, 1))  # feature별 정규화
                emb_std = np.std(emb, axis=(0, 1))
                emb_std = np.where(emb_std == 0, 1, emb_std)  # Zero Division 방지
                self.emb_means.append(emb_mean)
                self.emb_stds.append(emb_std)

            # self.label_means = np.mean(labels[start_month:end_month], axis=0)  # 동별 정규화 진행
            # self.label_stds = np.std(labels[start_month:end_month], axis=0)    # (동 개수,)
            self.label_means = np.mean(labels[start_month:end_month], axis=(0, 1))  # feature별 정규화 진행
            self.label_stds = np.std(labels[start_month:end_month], axis=(0, 1))
            self.label_stds = np.where(self.label_stds == 0, 1, self.label_stds)

        else:
            if train_means is None or train_stds is None or label_means is None or label_stds is None:
                raise ValueError("train_means, train_stds, label_means, and label_stds must be provided for validation/test datasets.")
            self.emb_means = train_means
            self.emb_stds = train_stds
            self.label_means = label_means
            self.label_stds = label_stds


    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """슬라이딩 윈도우를 적용하여 데이터 반환"""
        window = self.windows[idx]

        x = [emb[window['input_start']:window['input_end']].clone().detach() for emb in self.embeddings]

        # 정규화
        for i in range(len(x)):
            emb_mean = torch.tensor(self.emb_means[i], dtype=torch.float32)  # (동 개수, feature 개수)
            emb_std = torch.tensor(self.emb_stds[i], dtype=torch.float32)

            # x[i]의 차원이 (window_size, 동 개수, feature 개수)이므로 axis=1을 맞춰야 함
            #x[i] = (x[i] - emb_mean.unsqueeze(0)) / emb_std.unsqueeze(0)
            
            x[i] = (x[i] - emb_mean) / emb_std    # feature 별 정규화시

        # 미래 시점의 label
        y = self.labels[window['target_start']:window['target_end']].clone().detach()
        
        # 정규화 (각 동별 평균/표준편차 이용)
        y = (y - torch.tensor(self.label_means, dtype=torch.float32)) / torch.tensor(self.label_stds, dtype=torch.float32)

        return x, y
    
    def get_means_stds(self):
        """평균과 표준편차 반환"""
        return self.emb_means, self.emb_stds, self.label_means, self.label_stds

# 로그 변환된 데이터를 원래 값으로 되돌리는 함수
def inverse_transform(pred, mean, std):
    pred_restored = (pred * std) + mean  # 정규화 해제
    return np.expm1(pred_restored)       # 로그 변환 해제 (exp(x) - 1)


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray, horizon: int) -> Dict[str, float]:
    """
    특정 horizon (1개월, 3개월, 6개월) 후의 예측값과 실제값을 비교하여 MSE, RMSE, MAE를 계산하는 함수
    """
    pred_values = predictions[:, horizon-1, :, :]  # horizon-1을 사용하는 이유: 0-index
    actual_values = actuals[:, horizon-1, :, :]

    mse = np.mean((pred_values - actual_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_values - actual_values))

    return {
        f"{horizon}m_MSE": float(mse),
        f"{horizon}m_RMSE": float(rmse),
        f"{horizon}m_MAE": float(mae),
    }

def evaluate_per_label_mse(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """Evaluate model using MSE (same as training loss calculation)"""
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.HuberLoss(delta=1.0)

    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = [x.to(device) for x in batch_embeddings]
            batch_labels = batch_labels.to(device)

            predictions = model(*batch_embeddings)
            loss = criterion(predictions, batch_labels)  # MSE 계산

            total_loss += loss.item()
            num_batches += 1

    avg_mse_loss = total_loss / num_batches
    return avg_mse_loss

def evaluate_per_label(model: nn.Module, dataloader: DataLoader, device: str, horizon) -> Dict[str, float]:
    """Evaluate model for a single label"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            # Move to device
            batch_embeddings = [x.to(device) for x in batch_embeddings]
            batch_labels = batch_labels.to(device)

            # Forward pass
            predictions = model(*batch_embeddings)   # (batch_size, horizon, num_dongs, output_dim)

            #logger.info(f"prediction_shape: {predictions.shape}")
            #logger.info(f"label_shape: {batch_labels.shape}")
            
            # Collect predictions and labels
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_labels, axis=0)
    
    # 1개월, 3개월, 6개월 후 MSE, RMSE, MAE 계산
    metrics_1m = calculate_metrics(predictions, actuals, horizon=1)
    metrics_2m = calculate_metrics(predictions, actuals, horizon=2) if horizon >= 2 else {}
    metrics_3m = calculate_metrics(predictions, actuals, horizon=3) if horizon >= 3 else {}
    
    metrics_mean = {
        "avg_MSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MSE" in k]),
        "avg_RMSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "RMSE" in k]),
        "avg_MAE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MAE" in k])
    }

    # 최종 결과 합치기
    final_metrics = {**metrics_1m, **metrics_2m, **metrics_3m, **metrics_mean}
    
    return final_metrics

def evaluate_per_label_restored(model: nn.Module, dataloader: DataLoader, device: str, horizon, label_means, label_stds) -> Dict[str, float]:
    """Evaluate model for a single label using restored (un-normalized) values"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = [x.to(device) for x in batch_embeddings]
            batch_labels = batch_labels.to(device)
            predictions = model(*batch_embeddings)

            # Collect predictions and labels
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    # Concatenate all predictions and labels
    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_labels, axis=0)

    # 정규화 해제 + 로그 변환 해제
    predictions_restored = inverse_transform(predictions, label_means, label_stds)
    actuals_restored = inverse_transform(actuals, label_means, label_stds)

    # Evaluate using restored values
    metrics_1m = calculate_metrics(predictions_restored, actuals_restored, horizon=1)
    metrics_2m = calculate_metrics(predictions_restored, actuals_restored, horizon=2) if horizon >= 2 else {}
    metrics_3m = calculate_metrics(predictions_restored, actuals_restored, horizon=3) if horizon >= 3 else {}

    metrics_mean = {
        "avg_MSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MSE" in k]),
        "avg_RMSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "RMSE" in k]),
        "avg_MAE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MAE" in k])
    }
    
    final_metrics = {**metrics_1m, **metrics_2m, **metrics_3m, **metrics_mean}

    return final_metrics, predictions_restored, actuals_restored

# tensor를 json 형태로 저장되도록 변환
def tensor_to_list(state_dict):
    """Convert a model's state_dict (Tensor) to a JSON serializable format."""
    return {key: value.cpu().numpy().tolist() for key, value in state_dict.items()}

def create_predictions_dataframe(predictions, actuals, label_name):
    """
    실제값과 예측값을 DataFrame 형태로 변환하여 저장하는 함수
    """
    num_samples, horizon, num_dongs, output_dim = predictions.shape
    
    data = []
    
    for sample_idx in range(num_samples):
        for h in range(horizon):
            for dong_idx in range(num_dongs):
                row = {
                    "sample_id": sample_idx,
                    "horizon_month": h + 1,  # 1부터 시작하도록 조정
                    "dong_id": dong_idx,
                    "actual_value": actuals[sample_idx, h, dong_idx, 0],  # 실제값
                    "predicted_value": predictions[sample_idx, h, dong_idx, 0],  # 예측값
                }
                data.append(row)

    df = pd.DataFrame(data)
    return df

# 각 라벨별 모델 독립 학습 방식으로 변경
def train_validate_test(
    embeddings: List[np.ndarray],
    labels: np.ndarray,
    train_months: int,
    val_months: int,
    test_months: int,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    target: str,
    model_name: str
) -> Dict[str, Dict[str, Any]]:
    
    if target == 'all':
        num_labels = labels.shape[-1]  # label 개수 (3개)
        label_names = ["Reservation Days", "Revenue (USD)", "Number of Reservations"]
    else:
        num_labels = 1
        label_names = [target]
    
    results = {}
    label_statistic = {}
    horizon = get_prediction_horizon(model_config.mode)
    device = train_config.device

    for i in range(num_labels):
        if target == 'all':
            n = i
            logger.info(f"\n[Training Transformer for {label_names[n]}]")
        else:
            logger.info(f"\n[Training Transformer for {target}]")
            if target == 'Reservation Days':
                n = 0
            if target == 'Revenue':
                n = 1
            if target == 'Reservation':
                n = 2
                
        logger.info(f"\n🔹 Training model for {label_names[n]}...🔹")

        # Label별 데이터 생성 (각 label을 분리)
        label_data = labels[..., n].reshape(labels.shape[0], labels.shape[1], 1)  # (months, dongs, 1)

        # 모델 생성 -> 각 라벨별로 독립적인 모델 생성
        if model_name == 'transformer':
            model = EmbeddingTransformer(model_config).to(device)
        elif model_name == 'rnn':
            model = EmbeddingRNN(model_config).to(device)
        elif model_name == 'lstm':
            model = EmbeddingLSTM(model_config).to(device)

        # 멀티 GPU 활용시
        if train_config.use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
            model = nn.DataParallel(model)

        # ✅ Sliding Window 기반 데이터셋 생성
        train_dataset = SlidingWindowDataset(
            embeddings=embeddings,
            labels=label_data,  
            window_size=model_config.window_size,
            prediction_horizon=horizon,
            start_month=0,
            end_month=train_months + val_months + (horizon - 1),
            train_means=None,
            train_stds=None,
            is_train=True
        )
        
        logger.info(f"====Training windows {train_dataset.__len__()}====")
        
        # ✅ train 데이터셋에서 mean & std 가져오기
        train_means, train_stds, label_means, label_stds = train_dataset.get_means_stds()
        label_statistic[label_names[n]] = {'means': label_means, 'stds': label_stds}
        
        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
        criterion = nn.HuberLoss(delta=1.0)   # 매우 영향력이 큰 동들의 영향력을 축소
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=train_config.scheduler_patience, verbose=True
        )
        
        logger.info("\nStarting training...")
        
        # Training loop
        for epoch in range(train_config.epochs):
            model.train()
            train_loss = 0
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings = [x.to(device) for x in batch_embeddings]
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                predictions = model(*batch_embeddings)
                loss = criterion(predictions, batch_labels)  # 차원 유지
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Log progress
            logger.info(f"\nEpoch {epoch + 1}/{train_config.epochs}")
            logger.info(f"Train avg Loss: {train_loss/len(train_loader):.4f}")
        
        # Evaluate on test set
        logger.info(f"\nEvaluating on test set for {label_names[n]}...")
        
        test_dataset = SlidingWindowDataset(
            embeddings=embeddings,
            labels=label_data,  
            window_size=model_config.window_size,
            prediction_horizon=horizon,
            start_month=train_months + (horizon-1) + val_months + (horizon-1) - model_config.window_size,
            end_month=67,
            is_train=False,
            train_means=train_means,
            train_stds=train_stds,
            label_means=label_means,
            label_stds=label_stds
        )
        
        logger.info(f"====test windows {test_dataset.__len__()}====")
        
        test_loader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=False)
        test_metrics = evaluate_per_label(model, test_loader, device, horizon)
        real_metrics, preds, actuals = evaluate_per_label_restored(model, test_loader, device, horizon, label_means, label_stds)
        
        # 데이터프레임 생성
        predictions_df = create_predictions_dataframe(preds, actuals, label_names[n])

        # CSV로 저장
        save_path = Path(train_config.base_path) / f'predictions_vs_actuals_{label_names[n]}.csv'
        predictions_df.to_csv(save_path, index=False)
        
        logger.info(f"pred shape: {preds.shape}, actual shape: {actuals.shape}")
        
        # Save final results
        results[label_names[n]] = {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},  # float 변환 
            'test_real_metrics': {k: float(v) for k, v in real_metrics.items()},  # float 변환
        }

    return results
