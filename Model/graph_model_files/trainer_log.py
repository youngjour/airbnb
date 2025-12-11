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
from transformer_model import EmbeddingTransformer, ModelConfig
from RNN_model import EmbeddingRNN, RNNModelConfig
from LSTM_model import EmbeddingLSTM, LSTMModelConfig
from graph_model import EmbeddingGNN, GraphConfig


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
    use_curriculum_learning: bool = False # NEW: Curriculum Learning Flag

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
                 window_size: int,
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

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, int]: # ADDED INT RETURN (for Curriculum)
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

        return x, y, window['input_start'] # NEW: Return input_start index
    
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

def evaluate_per_label_mse(model: nn.Module, dataloader: DataLoader, device: str, criterion) -> float:
    """Evaluate model using MSE (same as training loss calculation)"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        # NOTE: Dataloader now returns (X, Y, Index)
        for batch_embeddings, batch_labels, _ in dataloader: 
            batch_embeddings = [x.to(device) for x in batch_embeddings]
            batch_labels = batch_labels.to(device)

            predictions = model(*batch_embeddings)
            loss = criterion(predictions, batch_labels)  # MSE 계산

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_per_label_multitask(model: nn.Module, dataloader: DataLoader, device: str, horizon: int) -> Dict[str, Dict[str, float]]:
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        # NOTE: Dataloader now returns (X, Y, Index)
        for batch_embeddings, batch_labels, _ in dataloader:
            batch_embeddings = [x.to(device) for x in batch_embeddings]
            batch_labels = batch_labels.to(device)
            preds = model(*batch_embeddings)
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_labels, axis=0)

    output_dim = predictions.shape[-1]
    label_names = ["Reservation Days", "Revenue (USD)", "Number of Reservations"]

    results = {}
    for i in range(output_dim):
        metrics_1m = calculate_metrics(predictions[..., i:i+1], actuals[..., i:i+1], horizon=1)
        metrics_2m = calculate_metrics(predictions[..., i:i+1], actuals[..., i:i+1], horizon=2) if horizon >= 2 else {}
        metrics_3m = calculate_metrics(predictions[..., i:i+1], actuals[..., i:i+1], horizon=3) if horizon >= 3 else {}
        avg_metrics = {
            "avg_MSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MSE" in k]),
            "avg_RMSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "RMSE" in k]),
            "avg_MAE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MAE" in k]),
        }
        results[label_names[i]] = {**metrics_1m, **metrics_2m, **metrics_3m, **avg_metrics}
    return results

def evaluate_per_label_restored_multitask(model: nn.Module, dataloader: DataLoader, device: str, horizon: int, label_means, label_stds) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray]:
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        # NOTE: Dataloader now returns (X, Y, Index)
        for batch_embeddings, batch_labels, _ in dataloader:
            batch_embeddings = [x.to(device) for x in batch_embeddings]
            batch_labels = batch_labels.to(device)
            preds = model(*batch_embeddings)
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_labels, axis=0)

    # Inverse transform
    preds_restored = inverse_transform(predictions, label_means, label_stds)
    actuals_restored = inverse_transform(actuals, label_means, label_stds)

    output_dim = predictions.shape[-1]
    label_names = ["Reservation Days", "Revenue (USD)", "Number of Reservations"]

    results = {}
    for i in range(output_dim):
        metrics_1m = calculate_metrics(preds_restored[..., i:i+1], actuals_restored[..., i:i+1], horizon=1)
        metrics_2m = calculate_metrics(preds_restored[..., i:i+1], actuals_restored[..., i:i+1], horizon=2) if horizon >= 2 else {}
        metrics_3m = calculate_metrics(preds_restored[..., i:i+1], actuals_restored[..., i:i+1], horizon=3) if horizon >= 3 else {}
        avg_metrics = {
            "avg_MSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MSE" in k]),
            "avg_RMSE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "RMSE" in k]),
            "avg_MAE": np.mean([v for k, v in {**metrics_1m, **metrics_2m, **metrics_3m}.items() if "MAE" in k]),
        }
        results[label_names[i]] = {**metrics_1m, **metrics_2m, **metrics_3m, **avg_metrics}
    return results, preds_restored, actuals_restored


# tensor를 json 형태로 저장되도록 변환
def tensor_to_list(state_dict):
    """Convert a model's state_dict (Tensor) to a JSON serializable format."""
    return {key: value.cpu().numpy().tolist() for key, value in state_dict.items()}

# 라벨별로 loss 가중치를 다르게 적용(이 것도 학습)
class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # 학습 가능한 파라미터

    def forward(self, predictions, targets):
        losses = []
        for i in range(predictions.shape[-1]):
            loss = nn.functional.huber_loss(predictions[..., i], targets[..., i], delta=1.0)
            precision = torch.exp(-self.log_vars[i])
            losses.append(precision * loss + self.log_vars[i])
        return sum(losses)

def test_total_metric(real_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    라벨별 평가 결과(real_metrics)를 받아 전체 라벨 평균 MSE, RMSE, MAE 계산

    Args:
        real_metrics: 각 라벨별 평가 metric dictionary

    Returns:
        전체 라벨 평균 metric을 담은 dictionary
    """
    all_mse = []
    all_rmse = []
    all_mae = []

    for label_result in real_metrics.values():
        for k, v in label_result.items():
            if "avg_MSE" in k:
                all_mse.append(v)
            elif "avg_RMSE" in k:
                all_rmse.append(v)
            elif "avg_MAE" in k:
                all_mae.append(v)

    return {
        'overall_avg_MSE': np.mean(all_mse),
        'overall_avg_RMSE': np.mean(all_rmse),
        'overall_avg_MAE': np.mean(all_mae)
    }


# 각 라벨별 모델 독립 학습 방식으로 변경
def train_validate_test(
    embeddings: List[np.ndarray],
    labels: np.ndarray,
    train_months: int,
    val_months: int,
    test_months: int,
    model_config: Any,
    train_config: TrainingConfig,
    target: str,
    model_name: str
) -> Dict[str, Dict[str, Any]]:
    
    horizon = get_prediction_horizon(model_config.mode)
    device = train_config.device
    output_dim = labels.shape[-1]  # 라벨 개수


    # Label별 데이터 생성 (각 label을 분리)
    label_data = labels  # (months, dongs, 3)

    # 모델 생성 -> 각 라벨별로 독립적인 모델 생성
    if model_name == 'transformer':
        model = EmbeddingTransformer(model_config, output_dim=output_dim).to(device)
    elif model_name == 'rnn':
        model = EmbeddingRNN(model_config, output_dim=output_dim).to(device)
    elif model_name == 'lstm':
        model = EmbeddingLSTM(model_config, output_dim=output_dim).to(device)
    elif model_name == 'graph':
        model = EmbeddingGNN(model_config, output_dim=output_dim).to(device)

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
        end_month=train_months+(horizon-1),
        train_means=None,
        train_stds=None,
        is_train=True
    )
    
    logger.info(f"====Training windows {train_dataset.__len__()}====")
    # ✅ train 데이터셋에서 mean & std 가져오기
    train_means, train_stds, label_means, label_stds = train_dataset.get_means_stds()
    
    # ✅ validation & test 데이터셋 생성
    val_dataset = SlidingWindowDataset(
        embeddings=embeddings,
        labels=label_data,  
        window_size=model_config.window_size,
        prediction_horizon=horizon,
        start_month=train_months + (horizon-1) - model_config.window_size,
        end_month=train_months + (horizon-1) + val_months + (horizon-1),
        train_means=train_means,
        train_stds=train_stds,
        is_train=False,
        label_means=label_means,
        label_stds=label_stds
    )
    logger.info(f"====Val windows {val_dataset.__len__()}====")
    
    # DataLoader 생성
    # NOTE: shuffle=False is required for the temporal curriculum logic
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)
    
    # --- Curriculum Learning Initialization ---
    if train_config.use_curriculum_learning:
        # Since shuffle=False is used, the dataset already provides the chronological order 
        # (index 0 = earliest month, last index = latest month) which serves as the curriculum.
        logger.info("Curriculum Learning is active: Training batches are ordered chronologically.")
        
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    criterion = WeightedMultiTaskLoss(num_tasks=output_dim)   # 매우 영향력이 큰 동들의 영향력을 축소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=train_config.scheduler_patience
    )

    best_val_loss = float('inf')
    best_model_state = None
    #best_metrics = None
    best_epoch = 0  # Early Stopping이 발생한 epoch 저장
    early_stopping_counter = 0
    early_stopping_patience = 20
    min_improvement = 1e-5  # stopping 기준
    
    logger.info("\nStarting training...")
    
    # Training loop
    for epoch in range(train_config.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        # NOTE: Dataloader iteration now returns (X, Y, Index)
        for batch_embeddings, batch_labels, _ in train_loader: 
            batch_embeddings = [x.to(device) for x in batch_embeddings]
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(*batch_embeddings)
            loss = criterion(predictions, batch_labels)  # 차원 유지
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # Validation loss 계산
        # NOTE: Validation loader must also be adjusted to unpack (X, Y, Index)
        avg_val_loss = evaluate_per_label_mse(model, val_loader, device, criterion)
        
        # Update learning rate
        scheduler.step(avg_val_loss)

        # Log progress
        logger.info(f"\nEpoch {epoch + 1}/{train_config.epochs}")
        logger.info(f"Train avg Loss: {avg_train_loss:.4f}")
        logger.info(f"Validation avg Loss: {avg_val_loss:.4f}")

        # Save best model (라벨별 모델 저장)
        if avg_val_loss < best_val_loss - min_improvement:
            best_val_loss = avg_val_loss
            #best_model_state = model.state_dict()  # 기존 코드
            best_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_epoch = epoch + 1
            
            # Save checkpoint (라벨별 파일명 설정)
            checkpoint_path = Path(train_config.base_path) / f'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # 🔹 Early Stopping 조건 확인
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"🚀 Early Stopping triggered at epoch {best_epoch}. Best validation loss: {best_val_loss:.4f}")
            break  # 학습 중단
    
    if model_name == 'transformer':
        final_model = EmbeddingTransformer(model_config, output_dim=output_dim).to(device)  # 새 모델 인스턴스 생성
    elif model_name == 'rnn':
        final_model = EmbeddingRNN(model_config, output_dim=output_dim).to(device)
    elif model_name == 'lstm':
        final_model = EmbeddingLSTM(model_config, output_dim=output_dim).to(device)
    elif model_name == 'graph':
        final_model = EmbeddingGNN(model_config, output_dim=output_dim).to(device)
        
    final_model.load_state_dict(best_model_state)                    # 저장된 best model 불러오기
    
    # Evaluate on test set
    logger.info(f"\nEvaluating on test set...")
    
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
    test_metrics = evaluate_per_label_multitask(final_model, test_loader, device, horizon)
    real_metrics, preds, actuals = evaluate_per_label_restored_multitask(final_model, test_loader, device, horizon, label_means, label_stds)
    
    # ✅ 전체 평균 metric 계산 함수 호출
    overall_metrics = test_total_metric(test_metrics)
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'test_real_metrics': real_metrics,
        'overall_real_metrics': overall_metrics,
        'early_stopping_epoch': best_epoch
    }

    return results