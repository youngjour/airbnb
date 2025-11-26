# Airbnb Demand Prediction Research: Final Report

**Date:** November 27, 2025
**Topic:** Enhancing Airbnb Demand Prediction with LLM-based Spatial & Tourism Embeddings

---

## 1. Executive Summary
We investigated whether integrating **Spatial (SGIS)** and **Tourism (T-Map)** data via Large Language Model (LLM) embeddings could improve Airbnb demand prediction accuracy.
*   **Conclusion:** While the proposed method successfully integrated heterogeneous data, the **Baseline Model (Historical Data + Road Network)** remains the most robust and accurate predictor.
*   **Key Finding:** Adding complex external context (SGIS/Tourism) introduced noise or optimization challenges that slightly degraded performance compared to the strong baseline.

---

## 2. Methodology
We compared four model configurations using a Transformer-based architecture:

1.  **HJ Baseline**: Road Network + Human Flow + Airbnb History (Benchmark)
2.  **YJ Model (SGIS)**: Baseline + SGIS Local Data Embedding
3.  **YJ Model (Tourism)**: Baseline + Tourism Data Embedding
4.  **YJ Model (Final)**: Baseline + SGIS + Tourism (All Features)

**Key Innovation:**
*   Used **LLMs (Llama-3.2-3B)** to generate embeddings for textual descriptions of regions (SGIS) and tourism spots.
*   Implemented **Hyperparameter Tuning** (Window Size, Dimension Allocation) to optimize feature fusion.

---

## 3. Experiment Results (100 Epochs)

| Rank | Model | RMSE (Lower is Better) | Status |
| :--- | :--- | :--- | :--- |
| **1** | **HJ Baseline** | **0.4538** | **Winner** |
| 2 | YJ Model (Tourism) | 0.4691 | +3.3% Error |
| 3 | YJ Model (SGIS) | 0.4692 | +3.4% Error |
| 4 | YJ Model (Final) | 0.5511 | +21.4% Error |

**Analysis:**
*   **Baseline Strength:** The historical Airbnb data itself is the strongest predictor.
*   **Contextual Overload:** The "Final" model with all 5 embeddings performed significantly worse, likely due to **overfitting** or the **curse of dimensionality** given the limited dataset size.
*   **Single-Source Context:** Adding *either* SGIS or Tourism data resulted in comparable performance, but neither beat the baseline.

---

## 4. Technical Achievements
Despite the negative result on the primary metric, we achieved significant technical milestones:
1.  **Memory Optimization:** Refactored `trainer_log.py` to reduce RAM usage by **50%+**, enabling training of large multi-modal models on a single machine.
2.  **Stability Fixes:** Resolved `AttributeError` bugs and implemented robust **Early Stopping** and **Deadlock Prevention** (using `num_workers=0`).
3.  **Hyperparameter Insight:** Discovered that allocating **higher dimensions (128)** to the Airbnb embedding while compressing others (48) yields the best stability.

---

## 5. Recommendations for Next Steps
1.  **Simplify the Model:** The current architecture struggles to fuse 5 distinct embeddings. Consider a **hierarchical fusion** approach (e.g., combine SGIS+Tourism first, then fuse with Airbnb).
2.  **Data Quality Check:** Verify if the SGIS/Tourism text descriptions truly correlate with demand. The LLM might be extracting irrelevant features.
3.  **Visualization:** Plot the prediction errors geographically. The YJ Model might be superior in **specific rural areas** even if the global average is worse.

---
*End of Report*
