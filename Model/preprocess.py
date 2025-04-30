import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self,
                 expected_months: int = 67, # 총 67달
                 admin_unit: str = 'dong',  # 'dong' 'less', 'normal', 'many'
                 expected_units: Optional[int] = None,  # If None, will be determined from data
                 label_features: int = 3):
        """Initialize preprocessor with configuration."""
        self.expected_months = expected_months
        self.admin_unit = admin_unit.lower()
        self.unit_col = 'Dong_name'
        self.label_features = label_features
        
        # Set default expected units if not provided
        self.expected_units = self._get_default_units()
        
        logger.info(f"Initialized preprocessor for {admin_unit} level data")
        logger.info(f"Expected {self.expected_units} {admin_unit}s over {expected_months} months")
    
    def _get_default_units(self) -> int:
        """Get default number of units based on administrative level."""
        defaults = {
            'dong': 424,
            'less': 102,
            'normal': 213,
            'many': 109,
            'not_less': 322,
            'half': 224
        }
        if self.admin_unit not in defaults:
            raise ValueError(f"Unsupported administrative unit: {self.admin_unit}. Use 'dong' or 'gu'")
        return defaults[self.admin_unit]

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns excluding unit name and reporting month."""
        return [col for col in df.columns if col not in [self.unit_col, 'Reporting Month']]

    def validate_data_ranges(self, data: np.ndarray, name: str) -> bool:   # name이 뭔가?
        """Validate data ranges and check for anomalies."""
        try:
            if np.isnan(data).any():
                logger.error(f"{name}: Contains NaN values")
                return False
            if np.isinf(data).any():
                logger.error(f"{name}: Contains infinite values")
                return False
                
            # logger.info(f"{name} statistics:")
            # logger.info(f"Range: [{data.min():.4f}, {data.max():.4f}]")
            # logger.info(f"Mean: {data.mean():.4f}")
            # logger.info(f"Std: {data.std():.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Data validation failed for {name}: {str(e)}")
            return False

    def validate_static_dataframe(self, df: pd.DataFrame, name: str) -> bool:
        """Validate static dataframe structure and content."""
        try:
            if self.unit_col not in df.columns:
                logger.error(f"{name}: Missing required column {self.unit_col}")
                return False
            
            feature_cols = self.get_feature_columns(df)
            if not feature_cols:
                logger.error(f"{name}: No feature columns found")
                return False
            
            logger.info(f"Checking numeric values in {len(feature_cols)} feature columns")
            
            for col in feature_cols:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='raise')
                    if numeric_data.isna().any():
                        logger.error(f"{name}: Column {col} contains NaN values")
                        return False
                except Exception as e:
                    logger.error(f"{name}: Column {col} contains non-numeric values: {str(e)}")
                    return False
            
            # units = df[self.unit_col].nunique()
            # if self.expected_units and units != self.expected_units:
            #     logger.error(f"{name}: Expected {self.expected_units} {self.admin_unit}s, got {units}")
            #     return False
            
            duplicates = df[self.unit_col].duplicated()
            if duplicates.any():
                duplicate_units = df[self.unit_col][duplicates].unique()
                logger.error(f"{name}: Found duplicate entries for units: {duplicate_units}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {name}: {str(e)}")
            return False

    def validate_temporal_dataframe(self, df: pd.DataFrame, name: str) -> bool:
        """Validate temporal dataframe structure and content."""
        try:
            required_cols = [self.unit_col, 'Reporting Month']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"{name}: Missing required columns {required_cols}")
                return False
            
            feature_cols = self.get_feature_columns(df)
            if isinstance(df[feature_cols], pd.DataFrame):
                numeric_data = df[feature_cols].values
            else:
                numeric_data = df[feature_cols]
                
            if np.isnan(numeric_data).any() or np.isinf(numeric_data).any():
                logger.error(f"{name}: Contains invalid numeric values")
                return False
            
            months = df['Reporting Month'].nunique()
            units = df[self.unit_col].nunique()
            
            if months != self.expected_months:
                logger.error(f"{name}: Expected {self.expected_months} months, got {months}")
                return False
            # if self.expected_units and units != self.expected_units:
            #     logger.error(f"{name}: Expected {self.expected_units} {self.admin_unit}s, got {units}")
            #     return False
            
            duplicates = df.groupby([self.unit_col, 'Reporting Month']).size()
            if (duplicates > 1).any():
                logger.error(f"{name}: Found duplicate entries for some {self.admin_unit}-month combinations")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {name}: {str(e)}")
            return False

    def load_static_df(self, path: str, name: str) -> Tuple[Optional[pd.DataFrame], int]:
        """Load and validate static dataframe."""
        if path is None:
            logger.info(f"Skipping {name} (not provided)")
            return None, 0
            
        logger.info(f"Loading static {name} from {path}")
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded dataframe shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            df.columns = [col.replace('dong_name', 'Dong_name').replace('gu_name', 'Gu_name') 
                        for col in df.columns]
            
            # 필요한 동 데이터만 불러오기
            if self.admin_unit == 'less':
                less_dong_names = list(pd.read_csv('../Data/Preprocessed_data/less_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(less_dong_names)]
            elif self.admin_unit == 'normal':
                normal_dong_names = list(pd.read_csv('../Data/Preprocessed_data/normal_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(normal_dong_names)]
            elif self.admin_unit == 'many':
                many_dong_names = list(pd.read_csv('../Data/Preprocessed_data/many_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(many_dong_names)]
            elif self.admin_unit == 'not_less':
                less_dong_names = list(pd.read_csv('../Data/Preprocessed_data/less_dong_names.csv')['Dong_name'])
                df = df[~df[self.unit_col].isin(less_dong_names)]
            elif self.admin_unit == 'half':
                half_dong_names = list(pd.read_csv('../Data/Preprocessed_data/half_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(half_dong_names)]
            
            if self.unit_col not in df.columns:
                logger.error(f"Required column {self.unit_col} not found. Available columns: {df.columns.tolist()}")
                return None, 0
                
            df = df.sort_values(self.unit_col)    # 행정동 기준 가나다순 데이터 정렬
            feature_cols = self.get_feature_columns(df)
            logger.info(f"Feature columns: {feature_cols}")
            
            if not self.validate_static_dataframe(df, name):
                logger.error(f"Static dataframe validation failed for {name}")
                raise ValueError(f"Validation failed for {name}")
            
            feature_count = len(feature_cols)
            logger.info(f"{name} has {feature_count} features")
            
            return df, feature_count
            
        except Exception as e:
            logger.error(f"Error loading {name}: {str(e)}")
            raise

    def load_temporal_df(self, path: str, name: str) -> Tuple[pd.DataFrame, int]:
        """Load and validate temporal dataframe."""
        logger.info(f"Loading temporal {name} from {path}")
        try:
            df = pd.read_csv(path)
            
            df.columns = [col.replace('dong_name', 'Dong_name')
                         .replace('gu_name', 'Gu_name')
                         .replace('Reporting Month', 'Reporting Month')
                         for col in df.columns]
            
            # 필요한 동 데이터만 불러오기
            if self.admin_unit == 'less':
                less_dong_names = list(pd.read_csv('../Data/Preprocessed_data/less_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(less_dong_names)]
            elif self.admin_unit == 'normal':
                normal_dong_names = list(pd.read_csv('../Data/Preprocessed_data/normal_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(normal_dong_names)]
            elif self.admin_unit == 'many':
                many_dong_names = list(pd.read_csv('../Data/Preprocessed_data/many_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(many_dong_names)]
            elif self.admin_unit == 'not_less':
                less_dong_names = list(pd.read_csv('../Data/Preprocessed_data/less_dong_names.csv')['Dong_name'])
                df = df[~df[self.unit_col].isin(less_dong_names)]
            elif self.admin_unit == 'half':
                half_dong_names = list(pd.read_csv('../Data/Preprocessed_data/half_dong_names.csv')['Dong_name'])
                df = df[df[self.unit_col].isin(half_dong_names)]
            
            if 'Reporting Month' in df.columns:
                df['Reporting Month'] = pd.to_datetime(df['Reporting Month'])
                df = df.sort_values(['Reporting Month', self.unit_col])   # 월 - 행정동별 데이터 정렬
                df['Reporting Month'] = df['Reporting Month'].dt.strftime('%Y-%m')
            
            if not self.validate_temporal_dataframe(df, name):
                logger.error(f"Validation failed for {name}")
                raise ValueError(f"Validation failed for {name}")
            
            feature_count = len(self.get_feature_columns(df))
            logger.info(f"{name} has {feature_count} features")
            
            return df, feature_count
            
        except Exception as e:
            logger.error(f"Error loading {name}: {str(e)}")
            raise

    def process_data(self,
                    embedding_paths: List[str],
                    label_path: str,
                    output_dir: Optional[str] = None) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """Process all dataframes and prepare them for model input."""
        try:
            # 라벨 데이터의 인덱스를 기준으로 정렬 
            logger.info("Loading label data...")
            label_df, label_feature_count = self.load_temporal_df(label_path, 'labels')
            
            # Process embeddings
            embeddings_data = []
            feature_counts = []
            processed_dfs = []
            
            # Load optional embeddings (if any)
            for i, path in enumerate(embedding_paths, 1):
                logger.info(f"Loading embedding{i}...")
                if path is None:
                    continue
                
                # 고정 도로네트워크일때만 static으로 전처리
                if path == '../Data/Preprocessed_data/Dong/Road_Embeddings.csv':
                    df, feature_count = self.load_static_df(path, f'embedding{i}')
                        
                else:
                    df, feature_count = self.load_temporal_df(path, f'embedding{i}')
                
                if df is not None:
                    processed_dfs.append(df)
                    feature_counts.append(feature_count)
                    
            # Align optional embeddings with label_df
            aligned_dfs = []
            for i, df in enumerate(processed_dfs, 1):
                logger.info(f"Aligning embedding{i}...")
                if 'Reporting Month' not in df.columns:  # This is static data
                    aligned_df = self.expand_static_features(df, label_df)
                else:
                    aligned_df = pd.merge(
                        label_df[[self.unit_col, 'Reporting Month']],
                        df,
                        on=[self.unit_col, 'Reporting Month'],
                        how='left',
                        validate='1:1'
                    )
                aligned_dfs.append(aligned_df)
            
            # 라벨 데이터도 추가
            aligned_dfs.append(label_df.iloc[:, 2:])
            feature_counts.append(3)
            
            # Process all embeddings
            logger.info("Processing embeddings...")
            for i, df in enumerate(aligned_dfs, 1):
                logger.info(f"Processing embedding{i}...")
                features = self.extract_features(df)
                reshaped = self.reshape_data(features, f'embedding{i}')
                if not self.validate_data_ranges(reshaped, f'embedding{i}'):
                    raise ValueError(f"Data validation failed for embedding{i}")
                embeddings_data.append(reshaped)    # 각 임베딩을 (months, units, features) 형태로 저장
            
            # Process labels
            logger.info("Processing labels...")
            label_features = self.extract_features(label_df)
            labels_reshaped = self.reshape_data(label_features, 'labels')
            
            if output_dir:
                self._save_config({
                    'embedding_paths': embedding_paths,
                    'label_path': label_path,
                    'admin_unit': self.admin_unit,
                    'expected_months': self.expected_months,
                    'expected_units': self.expected_units,
                    'feature_counts': feature_counts,
                    'label_feature_count': label_feature_count,
                    'unit_names': sorted(label_df[self.unit_col].unique().tolist())
                }, output_dir)
            
            logger.info(f"Final feature counts: {feature_counts}")
            return embeddings_data, labels_reshaped, feature_counts
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

    def expand_static_features(self, static_df: pd.DataFrame, 
                             reference_df: pd.DataFrame) -> pd.DataFrame:
        """Expand static features to match temporal structure."""
        try:
            months = reference_df['Reporting Month'].unique()
            expanded_rows = []
            for month in months:
                temp_df = static_df.copy()
                temp_df['Reporting Month'] = month
                expanded_rows.append(temp_df)
            
            expanded_df = pd.concat(expanded_rows, ignore_index=True)
            expanded_df = expanded_df.sort_values(['Reporting Month', self.unit_col])
            return expanded_df
            
        except Exception as e:
            logger.error(f"Error expanding static features: {str(e)}")
            raise

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from dataframe."""
        feature_cols = self.get_feature_columns(df)
        data = df[feature_cols].values
        logger.info(f"Extracted features shape: {data.shape}")
        return data

    def reshape_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """Reshape data to (months, units, features) format."""
        try:
            n_features = data.shape[1]
            reshaped = data.reshape(self.expected_months, self.expected_units, n_features)
            logger.info(f"Reshaped {name} from {data.shape} to {reshaped.shape}")
            return reshaped
        except Exception as e:
            logger.error(f"Error reshaping {name}: {str(e)}")
            raise

    def _save_config(self, config: Dict[str, Any], output_dir: str) -> None:
        """Save preprocessing configuration."""
        output_path = Path(output_dir) / "preprocessing_config.json"
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)   # 한국어 동이름 저장
        logger.info(f"Preprocessing configuration saved to {output_path}")

def load_and_preprocess_data(
    embedding_paths: List[str],
    label_path: str,
    admin_unit: str = 'dong',
    expected_months: int = 67,
    expected_units: Optional[int] = None,
    label_features: int = 3,
    output_dir: Optional[str] = None
) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    """
    Convenience function for data preprocessing.
    Returns separate arrays for each embedding source and their feature counts.
    
    Args:
        embedding_paths: List of paths to embedding CSV files
        label_path: Path to labels CSV file
        admin_unit: Administrative unit level ('dong' or 'gu')
        expected_months: Expected number of months in the data
        expected_units: Expected number of administrative units
        label_features: Number of features in labels
        output_dir: Directory to save preprocessing configuration
        
    Returns:
        Tuple containing:
        - List of processed embedding arrays
        - Processed labels array
        - List of feature counts for each embedding
    """
    try:
        # Ensure embedding3 (required) is provided
        if not embedding_paths:
            raise ValueError("At least one embedding path (embedding3) must be provided")
        
        preprocessor = DataPreprocessor(
            expected_months=expected_months,
            admin_unit=admin_unit,
            expected_units=expected_units,
            label_features=label_features
        )
        
        # Process the data   -> embedding에는 해당 시점 label 데이터도 포함
        embeddings, labels, feature_counts = preprocessor.process_data(
            embedding_paths=embedding_paths,
            label_path=label_path,
            output_dir=output_dir
        )
        
        # Log final shapes and statistics
        logger.info("\nFinal processed data shapes and statistics:")
        for i, emb in enumerate(embeddings):
            logger.info(f"Embedding {i+1}:")
            logger.info(f"  Shape: {emb.shape}")
            # logger.info(f"  Range: [{emb.min():.4f}, {emb.max():.4f}]")
            # logger.info(f"  Mean: {emb.mean():.4f}")
            # logger.info(f"  Std: {emb.std():.4f}")
        
        logger.info(f"\nLabels:")
        logger.info(f"  Shape: {labels.shape}")
        # logger.info(f"  Range: [{labels.min():.4f}, {labels.max():.4f}]")
        # logger.info(f"  Mean: {labels.mean():.4f}")
        # logger.info(f"  Std: {labels.std():.4f}")
        
        logger.info(f"\nFeature counts: {feature_counts}")
        
        return embeddings, labels, feature_counts
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise