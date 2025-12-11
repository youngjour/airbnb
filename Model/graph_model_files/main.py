import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path
from graph.build_graph_osm import build_graph, build_dense_adjacency 
from graph_model import GraphConfig 
from transformer_model import ModelConfig
from RNN_model import RNNModelConfig
from LSTM_model import LSTMModelConfig
from trainer_log import TrainingConfig, train_validate_test

from preprocess import load_and_preprocess_data
import random

import torch
import numpy as np

def set_seed(seed=42):
    """Sets seeds for reproducibility across CPU and GPU."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure logging
logger = logging.getLogger(__name__)

def str2bool(v):
    """Converts string arguments to boolean."""
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
    
    # --- Data & Paths ---
    parser.add_argument('--embed1', type=str, default=None, help='Optional: Path to embedding1 CSV file')  
    parser.add_argument('--embed2', type=str, default=None, help='Optional: Path to embedding2 CSV file')     
    parser.add_argument('--embed3', type=str, default=None, help='Optional: Path to embedding3 CSV file')
    parser.add_argument('--embed4', type=str, default=None, help='Optional: Path to embedding4 CSV file')
    parser.add_argument('--embed5', type=str, default=None, help='Optional: Path to embedding5 CSV file (tourism_llm)')
    parser.add_argument('--label_path', type=str, default='../Data/Preprocessed_data/AirBnB_labels_dong.csv', help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs_graph_sota', help='Directory to save results and models')
    
    # --- Model Selection & Variant ---
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm', 'transformer', 'graph'], default='graph', help='TimeSeries Model')
    parser.add_argument('--variant', type=str, choices=['baseline', 'adaptive', 'gat'], default='baseline', help='Graph Model Variant for comparison (baseline=static, adaptive=learned)') 
    
    # --- Graph Hyperparameters (Static) ---
    parser.add_argument('--dong_geojson', type=str, default='Data/Geo/dong_seoul.geojson', help='Dong-level GeoJSON used to build the static graph')
    parser.add_argument('--k', type=int, default=6, help='k for kNN graph')
    parser.add_argument('--sigma_km', type=float, default=1.0, help='RBF length scale (km)')
    
    # --- Data Dimensions & Splits ---
    parser.add_argument('--train_months', type=int, default=49)
    parser.add_argument('--val_months', type=int, default=6)
    parser.add_argument('--test_months', type=int, default=7)
    parser.add_argument('--admin_unit', type=str, choices=['dong', 'less', 'not_less', 'normal', 'half', 'many'], default='dong')
    parser.add_argument('--dim_opt', type=int, default=3, help='Type of each embedding dimension')
    parser.add_argument('--window_size', type=int, default=9, help='Size of sliding window for input features')
    parser.add_argument('--mode', type=str, choices=['1m', '3m', '6m'], default='3m', help='Prediction mode (1, 3 or 6 months ahead)')
    parser.add_argument('--label', type=str, choices=['Reservation Days', 'Revenue', 'Reservation', 'all'], default='all', help='Predict Label name')
    
    # --- General Hyperparameters ---
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--use_temporal_attention', type=str2bool, nargs='?', const=True, default=False,
                    help='If True, replaces GRU with Temporal Transformer Attention.')
    
    
    # --- Training Configuration ---
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_multi_gpu', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--use_curriculum_learning', type=str2bool, nargs='?', const=True, default=False,
                    help='If True, sorts batches by time index (easier to harder).')
    parser.add_argument('--feature_groups', type=str, default='1',
                    help='Defines which temporal encoder each embedX goes to. E.g., "1,2,1" means embed1 and embed3 go to Encoder 1, embed2 goes to Encoder 2.')
    
    args = parser.parse_args()
    
    # Embedding Path Dictionary (Retained)
    embedding_paths_dict = {
                    'road': '../Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv',
                    'hf': '../Data/Preprocessed_data/Dong/Human_flow.csv',
                    'raw': '../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv',
                    'llm_w': '../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv',
                    'tourism_llm': '../Preprocess/tmap_tourism/tourism_llm_embeddings_model_period.csv',
                    'sgis_improved': '../Preprocess/sgis_manual/sgis_improved_final.csv',
                    'sgis_local_llm_v2': '../Preprocess/sgis_manual/sgis_local_llm_embeddings_v2.csv',
                    }

    # Simplified embedding path creation
    embedding_list = []
    for p in [args.embed1, args.embed2, args.embed3, args.embed4, args.embed5]:
        if p is not None:
            embedding_list.append(embedding_paths_dict.get(p, p)) # Use dict value or path directly
    args.embedding_paths = embedding_list
    
    # Validate paths
    for path in args.embedding_paths + [args.label_path]:
        if not os.path.exists(path):
            parser.error(f"File not found: {path}")
    
    return args

def create_experiment_dir(base_dir: str, args: argparse.Namespace) -> Path:
    """Create timestamped experiment directory with variant name."""
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    exp_name = f"{args.model}_{args.variant}_w{args.window_size}_{args.admin_unit}_dim{args.dim_opt}_{timestamp}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def save_config(config: Dict[str, Any], exp_dir: Path) -> None:
    """Save experiment configuration."""
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")

def validate_data_split(total_months: int, args: argparse.Namespace) -> None:
    """Validate data split configuration."""
    horizon = {'1m': 1, '3m': 3, '6m': 6}[args.mode]
    min_required = horizon
    
    if args.train_months < min_required or args.val_months < min_required:
        raise ValueError(f"Train/Validation periods must be at least {min_required} months.")

def main():
    seed = 43
    set_seed(seed)
    try:
        args = parse_args()
        
        # Create experiment directory and save config
        exp_dir = create_experiment_dir(args.output_dir, args)
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
        
        validate_data_split(len(labels), args)
        
        # Get Dong order and N_nodes
        prep_cfg_path = exp_dir / "preprocessing_config.json"
        with open(prep_cfg_path, "r") as f:
            prep_cfg = json.load(f)
        dong_order = prep_cfg["unit_names"]
        n_nodes = len(dong_order)

        A_dense_torch = None
        
        # --- Graph Construction Logic (CLEANED UP) ---
        is_graph_model = args.model == 'graph'
        
        if is_graph_model:
            logger.info(f"Building static spatial graph ({args.k}-NN, sigma={args.sigma_km}km) from GeoJSON...")
            # Build graph & dense adjacency once
            edge_index, edge_weight = build_graph(
                args.dong_geojson,
                dong_order,
                k=args.k,
                sigma_km=args.sigma_km
            )
            A_dense = build_dense_adjacency(edge_index, edge_weight, n_nodes=n_nodes)
            # The static A is built once and passed to the config whether we use it as baseline or as bias for adaptive.
            A_dense_torch = A_dense.to(torch.float32)


        logger.info(f"\nData Configuration: Total months: {len(labels)}, Window size: {args.window_size}")
        
        # --- Model Configuration ---
        if args.model == 'transformer':
            model_config = ModelConfig(
                mode=args.mode,
                window_size=args.window_size,
                input_dims=tuple(feature_counts),
                dim_opt=args.dim_opt,
                num_encoder_layers=1,      
                nhead=4,                   
                dim_feedforward=64,
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

        elif args.model == 'graph':
            # --- Graph Model Variant Setup (Hierarchical Fusion Logic Retained) ---
            try:
                feature_groups = [int(g.strip()) for g in args.feature_groups.split(',')]
                if len(feature_groups) != len(feature_counts):
                    raise ValueError("Length of --feature_groups must match number of input embeddings.")
            except:
                logger.warning("Feature group definition failed. Defaulting all features to Group 1.")
                feature_groups = [1] * len(feature_counts)

            use_adaptive = args.variant == 'adaptive'
            graph_type = args.variant if args.variant in ['gat'] else 'diffusion'
            
            # --- MINIMAL ARGUMENTS PASSED TO BYPASS TYPE ERRORS ---
            model_config = GraphConfig(
                # Only pass the bare essentials that the underlying model NEEDS:
                n_nodes=n_nodes,
                A_dense=A_dense_torch
            )
            
            # --- TEMPORARY HACK: MANUALLY SET MISSING CONFIG ATTRIBUTES ---
            # NOTE: This only works if EmbeddingGNN pulls attributes from the config object directly,
            # which it is designed to do.

            model_config.tconv_channels = 128     # <-- NEW: TCN Output Channels
            model_config.K = 2                    # <-- NEW: Diffusion Step Size (K)
            model_config.dropedge_p = 0.05        # <-- NEW: DropEdge Probability
            model_config.adaptive_dim = 10        # (Ensure this line is present from the previous fix)

            model_config.input_dims = tuple(feature_counts)
            model_config.mode = args.mode
            model_config.window_size = args.window_size
            model_config.dim_opt = args.dim_opt
            model_config.hidden_size = args.hidden_size
            model_config.num_layers = args.num_layers
            model_config.dropout = args.dropout
            model_config.gnn_hidden = args.hidden_size
            model_config.gnn_layers = args.num_layers
            model_config.graph_type = graph_type
            model_config.use_adaptive_graph = use_adaptive
            model_config.adaptive_dim = 10
            model_config.use_temporal_attention = args.use_temporal_attention
            model_config.feature_groups = feature_groups

        # Create training configuration
        train_config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            base_path=str(exp_dir),
            use_multi_gpu=args.use_multi_gpu,
            use_curriculum_learning=args.use_curriculum_learning
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
        results_path = exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        logger.info("\nExperiment completed! 🎉")
        logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()