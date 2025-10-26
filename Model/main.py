import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path

import torch
import numpy as np
from transformer_model import ModelConfig
from RNN_model import RNNModelConfig
from LSTM_model import LSTMModelConfig
from trainer_log import TrainingConfig, train_validate_test  # Updated import

from preprocess import load_and_preprocess_data
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 연산 결정론 보장
    torch.backends.cudnn.benchmark = False     # 비결정적 연산 방지

# Configure logging
logger = logging.getLogger(__name__)

def str2bool(v):
    """문자열을 bool 값으로 변환"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Prediction with Sliding Window')
    
    # Data paths
    parser.add_argument('--embed1', type=str, default=None,
                      help='Optional: Path to embedding1 CSV file (static features)')  
    parser.add_argument('--embed2', type=str, default=None,
                      help='Optional: Path to embedding2 CSV file')     
    parser.add_argument('--embed3', type=str, default=None,
                      help='Optional: Path to embedding3 CSV file')
    parser.add_argument('--embed4', type=str, default=None,
                      help='Optional: Path to embedding4 CSV file')
    parser.add_argument('--label_path', type=str, default='../Data/Preprocessed_data/AirBnB_labels_dong.csv',
                      help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs_transformer',
                      help='Directory to save results and models')
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm', 'transformer'], default='rnn',
                      help='TimeSeries Model')
    
    # Data dimensions
    parser.add_argument('--train_months', type=int, default=49,
                      help='Number of months for training (default: 49)')
    parser.add_argument('--val_months', type=int, default=6,
                      help='Number of months for validation (default: 6)')
    parser.add_argument('--test_months', type=int, default=7,
                      help='Number of months for testing (default: 7)')
    parser.add_argument('--admin_unit', type=str, choices=['dong', 'less', 'not_less', 'normal', 'half', 'many'], default='dong',
                      help='Administrative unit level')
    
    # Model configuration
    parser.add_argument('--dim_opt', type=int, default=3,
                      help='Type of each embedding dimension')  # p
    parser.add_argument('--use_multi_gpu', type=str2bool, nargs='?', const=True, default=False,
                      help='Using multiple GPU')  # p
    parser.add_argument('--window_size', type=int, default=9,
                      help='Size of sliding window for input features')  # p
    parser.add_argument('--mode', type=str, choices=['1m', '3m'], default='3m',
                      help='Prediction mode (1 or 3months ahead)')
    parser.add_argument('--label', type=str, choices=['Reservation Days', 'Revenue', 'Reservation', 'all'], default='all',    # 변경 부분
                      help='Predict Label name')
    
    # Transformer-specific
    parser.add_argument('--num_encoder_layers', type=int, default=4,
                      help='Number of transformer encoder layers')
    parser.add_argument('--nhead', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                      help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # RNN, LSTM-specific
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)

    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Number of sliding windows per batch')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create embedding_paths list
    embedding_paths_dict = {
                       'road': '../Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv',
                       'hf': '../Data/Preprocessed_data/Dong/Human_flow.csv',
                       'raw': '../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv',
                       'llm_w': '../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv',
                       'llm_wo': '../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_wo.csv',
                       'road_llm': '../Data/Preprocessed_data/Dong/llm_embeddings_new/road_llm.csv',
                       'hf_llm': '../Data/Preprocessed_data/Dong/llm_embeddings_new/human_flow_llm.csv',
                       'sgis': '../Preprocess/sgis_manual/sgis_monthly_embedding_aligned_dates.csv',
                       'sgis_improved': '../Preprocess/sgis_manual/sgis_improved_final.csv',
                       # Local embeddings (LLM-generated from SGIS features)
                       'sgis_local_llm': '../Preprocess/sgis_manual/sgis_local_llm_embeddings.csv',
                       'sgis_local_llm_v2': '../Preprocess/sgis_manual/sgis_local_llm_embeddings_v2.csv',
                       # Feature selection subsets
                       'sgis_competition': '../Preprocess/sgis_manual/sgis_improved_subset_competition.csv',
                       'sgis_attractiveness': '../Preprocess/sgis_manual/sgis_improved_subset_attractiveness.csv',
                       'sgis_ratios': '../Preprocess/sgis_manual/sgis_improved_subset_ratios.csv',
                       'sgis_penetration': '../Preprocess/sgis_manual/sgis_improved_subset_penetration.csv',
                       'sgis_no_redundancy': '../Preprocess/sgis_manual/sgis_improved_subset_no_redundancy.csv',
                       # Custom user-requested subsets
                       'sgis_two_ratios': '../Preprocess/sgis_manual/sgis_improved_subset_two_ratios.csv',
                       'sgis_housing_ratios': '../Preprocess/sgis_manual/sgis_improved_subset_housing_plus_ratios.csv'}

    # Allow direct file paths if not in the dictionary
    embedding_list = []
    for p in [args.embed1, args.embed2, args.embed3, args.embed4]:
        if p is not None:
            if p in embedding_paths_dict:
                embedding_list.append(embedding_paths_dict[p])
            else:
                # Treat as direct file path
                embedding_list.append(p)
    args.embedding_paths = embedding_list
    
    # Validate paths
    for path in args.embedding_paths + [args.label_path]:
        if not os.path.exists(path):
            parser.error(f"File not found: {path}")
    
    return args

def create_experiment_dir(base_dir: str, args: argparse.Namespace) -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    exp_name = f"{args.model}_w{args.window_size}_{args.admin_unit}_dim{args.dim_opt}_{timestamp}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Add experiment-specific log file
    # log_file = exp_dir / f"{exp_name}.log"
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logger.addHandler(file_handler)
    
    return exp_dir

def save_config(config: Dict[str, Any], exp_dir: Path) -> None:
    """Save experiment configuration."""
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")

def validate_data_split(total_months: int, args: argparse.Namespace) -> None:
    """Validate data split configuration."""
    horizon = {'1m': 1, '3m': 3}[args.mode]
    min_required = horizon   # 앞으로 예측할 3개월 이상만 데이터가 존재한다면 문제 없이 진행 가능
    
    # expected_total = args.train_months + args.val_months + args.test_months + ((horizon-1)*2)   # 학습 가능 기간
    # if total_months != expected_total:
    #     raise ValueError(f"Total months in data ({total_months}) does not match "
    #                     f"expected split ({expected_total})")
    
    if args.train_months < min_required:
        raise ValueError(f"Training period ({args.train_months}) must be at least "
                        f"{min_required} months for window_size={args.window_size} "
                        f"and mode={args.mode}")
    
    if args.val_months < min_required:
        raise ValueError(f"Validation period ({args.val_months}) must be at least "
                        f"{min_required} months")

def main():
    """Main execution function."""
    seed = 43
    set_seed(seed)
    try:
        # Parse arguments
        args = parse_args()
        
        # Create experiment directory
        exp_dir = create_experiment_dir(args.output_dir, args)
        
        # Save configuration
        config_dict = vars(args)
        config_dict['seed'] = seed
        save_config(config_dict, exp_dir)
        
        # Load and preprocess data
        logger.info(f"\nLoading and preprocessing data...")
        embeddings, labels, feature_counts = load_and_preprocess_data(
            embedding_paths=args.embedding_paths,
            label_path=args.label_path,
            admin_unit=args.admin_unit,
            output_dir=exp_dir
        )
        
        # Validate data split configuration
        validate_data_split(len(labels), args)
        
        logger.info(f"\nData Configuration:")
        logger.info(f"Total months: {len(labels)}")
        logger.info(f"Training months: {args.train_months}")
        logger.info(f"Validation months: {args.val_months}")
        logger.info(f"Test months: {args.test_months}")
        logger.info(f"Window size: {args.window_size}")
        logger.info(f"Prediction mode: {args.mode}")
        logger.info(f"Number of features per embedding: {feature_counts}")
        logger.info(f"dimension option: {args.dim_opt}")
        
        # Create model configuration
        if args.model == 'transformer':
            model_config = ModelConfig(
                mode=args.mode,
                window_size=args.window_size,
                input_dims=tuple(feature_counts),
                dim_opt=args.dim_opt,
                num_encoder_layers=args.num_encoder_layers,
                nhead=args.nhead,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout
            )
        elif args.model == 'rnn':
            model_config = RNNModelConfig(
                input_dims=tuple(feature_counts),
                mode=args.mode,
                window_size=args.window_size,
                dim_opt=args.dim_opt,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional
            )
        elif args.model == 'lstm':
            model_config = LSTMModelConfig(
                input_dims=tuple(feature_counts),
                mode=args.mode,
                window_size=args.window_size,
                dim_opt=args.dim_opt,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional
            )

        # Create training configuration
        train_config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            base_path=str(exp_dir),
            use_multi_gpu=args.use_multi_gpu
        )
        
        # Train, validate and test model
        all_metrics = train_validate_test(
            embeddings=embeddings,
            labels=labels,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            model_config=model_config,
            train_config=train_config,
            target = args.label,
            model_name = args.model
        )
        
        # Save final results
        # results = {
        #     'metrics': all_metrics,
        #     'config': {
        #         'model': vars(model_config),
        #         'training': vars(train_config),
        #         'data': {
        #             'train_months': args.train_months,
        #             'val_months': args.val_months,
        #             'test_months': args.test_months,
        #             'feature_counts': feature_counts
        #         }
        #     }
        # }
        
        results_path = exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Print summary
        logger.info("\nExperiment completed!")
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()