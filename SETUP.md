# Airbnb Demand Forecasting Model Setup

This repository contains the code for forecasting Airbnb demand using various embeddings (SGIS, Tourism, Road Network, Human Flow).

## 1. Environment Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Installation

1.  **Clone the repository** (if you haven't already).

2.  **Install PyTorch**:
    Since you are running in a better environment, please install the appropriate PyTorch version for your CUDA version.
    Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the command.
    Example (for CUDA 12.4):
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

3.  **Install other dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 2. Data Setup

**IMPORTANT**: Some data files are not included in the repository and must be placed manually.

1.  **Data Directory**:
    - Ensure you have the `Data` folder in the root directory.
    - Structure:
      ```
      airbnb/
      ├── Data/
      │   └── Preprocessed_data/
      │       ├── Dong/
      │       │   ├── Road_Embeddings_with_flow.csv
      │       │   ├── Human_flow.csv
      │       │   ├── AirBnB_raw_embedding.csv
      │       │   └── ...
      │       └── AirBnB_labels_dong.csv
      ```

2.  **Embedding Files**:
    - Place the emailed embedding files in the following locations:
      - `Preprocess/sgis_manual/sgis_local_llm_embeddings.csv`
      - `Preprocess/tmap_tourism/tourism_llm_embeddings.csv`

## 3. Running the Model

The main training script is located in `Model/main.py`.

### Basic Usage
To run the model with default settings (RNN):
```bash
cd Model
python main.py
```

### Transformer Model
To run the Transformer model:
```bash
python main.py --model transformer --epochs 200 --batch_size 16
```

### Using Specific Embeddings
You can specify up to **5 embeddings** (`--embed1` through `--embed5`).
Example using the new Tourism and SGIS Local embeddings:
```bash
python main.py \
  --model transformer \
  --embed1 sgis_improved \
  --embed2 tourism_llm \
  --embed3 sgis_local_llm \
  --epochs 200
```

#### Available Embedding Keys
These keys can be passed to any `--embedX` argument:

**SGIS (Demographic/Regional Data)**
- `sgis`: Original SGIS monthly embedding.
- `sgis_improved`: Improved SGIS embedding (Final version).
- `sgis_local_llm`: LLM-generated embeddings from SGIS local features.
- `sgis_local_llm_v2`: Version 2 of SGIS local LLM embeddings.

**Tourism**
- `tourism_llm`: LLM-generated embeddings from TMAP/Credit Card tourism data.

**Other**
- `road`: Road network embeddings.
- `hf`: Human flow data.
- `raw`: Airbnb raw embeddings.
- `road_llm`: Road network LLM embeddings.
- `hf_llm`: Human flow LLM embeddings.

### Key Arguments
- `--model`: Model architecture (`rnn`, `lstm`, `transformer`).
- `--mode`: Prediction horizon (`1m` for 1 month, `3m` for 3 months).
- `--window_size`: Input window size (default: 9).
- `--admin_unit`: Administrative unit level (default: `dong`).
- `--device`: `cuda` or `cpu`.

For a full list of arguments:
```bash
python main.py --help
```
