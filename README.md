# 🏘️ Enhancing Regional Airbnb Trend Forecasting
**Using LLM-Based Embeddings of Accessibility and Human Mobility**

This repository contains the official implementation of the paper:
> *Enhancing Regional Airbnb Trend Forecasting Using LLM-Based Embeddings of Accessibility and Human Mobility*  
> by Hongju Lee, Youngjun Park, Jisun An, and Dongman Lee

---

## 🧠 Overview

We propose a novel regional-level time-series forecasting framework to predict key Airbnb market indicators (Revenue, Reservation Days, and Number of Reservations) at the **dong** level in Seoul.  
Unlike prior studies that focus on individual listings, our model integrates:

- 📍 **Urban Accessibility Data**
- 🧍 **Human Mobility Data**
- 📊 **Airbnb Listing Features**

We use **prompt-based LLM embeddings (via LLaMA3)** to represent regional context, then apply RNN/LSTM/Transformer for multi-step time-series prediction.

---

## 🏗️ Model Architecture
![Model Architecture](./assets/model.pdf)


## 📂 Data Sources

- **Airbnb Listing & Reservation Data**  
  Obtained from [AirDNA](https://www.airdna.co/) under a paid academic license.  
  Due to licensing restrictions, we are **unable to share the raw Airbnb dataset** in this repository.  
  However, if you would like to access the **preprocessed regional-level dataset** used in this study, please contact the authors.

- **Urban Accessibility Data**  
  Extracted from [OpenStreetMap (OSM)](https://www.openstreetmap.org/) using the [OSMnx](https://github.com/gboeing/osmnx) API.  
  Includes road network structure, road types, and adjacency between administrative districts (dongs).

- **Human Mobility Data**  
  Retrieved from the [Seoul Open Data Portal](https://data.seoul.go.kr), including the following datasets (accessed on 20 Apr 2025):
  1. [Domestic Living Population Data](https://data.seoul.go.kr/dataList/OA-14991/S/1/datasetView.do)  
  2. [Long-Term Foreign Residents Data](https://data.seoul.go.kr/dataList/OA-14992/S/1/datasetView.do)  
  3. [Short-Term Foreign Visitors Data](https://data.seoul.go.kr/dataList/OA-14993/S/1/datasetView.do)

If you are interested in the preprocessed dataset used for model training, feel free to contact us.  
We can provide dong-level monthly features upon request for academic purposes.


## 🚀 How to Run

### 1. Preprocessed Data

Ensure that your preprocessed embedding files and label file exist at the following default paths:

- `../Data/Preprocessed_data/Dong/Human_flow.csv`
- `../Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv`
- `../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv`
- `../Data/Preprocessed_data/Dong/llm_embeddings_new/...` (if using LLM)
- `../Data/Preprocessed_data/AirBnB_labels_dong.csv`

> You can modify these paths using `--embed1`, `--embed2`, `--embed3`, and `--label_path`.

---

### 2. Example (LSTM + 3 month Prediction)

```bash
python main.py \
  --embed1 road_llm \
  --embed2 hf_llm \
  --embed3 llm_w \
  --model lstm \
  --dim_opt 3 \
  --window_size 6 \
  --mode 3m \
  --label all \
  --output_dir outputs

### 📥 Available Embedding Options

You can specify up to 3 input embeddings using `--embed1`, `--embed2`, `--embed3`.  
Each option corresponds to a specific CSV path as defined in the code:

| Option Key | Description | File Path |
|------------|-------------|-----------|
| `road`     | Road network embedding with connectivity and flow data | `../Data/Preprocessed_data/Dong/Road_Embeddings_with_flow.csv` |
| `hf`       | Raw human mobility embedding (e.g., population flow) | `../Data/Preprocessed_data/Dong/Human_flow.csv` |
| `raw`      | Airbnb-based raw feature embedding (e.g., listing counts, revenue stats) | `../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv` |
| `llm_w`    | LLM-based embedding with listing information | `../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv` |
| `llm_wo`   | LLM-based embedding without listing information | `../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_wo.csv` |
| `road_llm` | LLM embedding generated from road network context | `../Data/Preprocessed_data/Dong/llm_embeddings_new/road_llm.csv` |
| `hf_llm`   | LLM embedding generated from human flow context | `../Data/Preprocessed_data/Dong/llm_embeddings_new/human_flow_llm.csv` |

> 📌 You can mix different types (e.g., `road`, `hf`, `llm_wo`) to explore different combinations.

---


## 📬 Contact

For questions or dataset requests, please contact:  
💌 lhj5561@kaist.ac.kr