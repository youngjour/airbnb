# Final Experiment Report: Airbnb Prediction Model

**Date:** 2025-11-25
**Objective:** Compare model performance (HJ Baseline vs. YJ Models) and optimize hyperparameters.

## 1. Executive Summary
The **Final YJ Model** (incorporating both SGIS and Tourism data) with **Baseline Hyperparameters** (`w=9`, `dim_opt=3`) achieved the best performance, outperforming the HJ Baseline.

*   **Best Model:** YJ Model (SGIS + Tourism)
*   **Best RMSE:** **0.5046** (vs Baseline 0.5053)
*   **Optimal Config:** Window Size = 9, Embedding Dimension Option = 3

## 2. Model Comparison Results

| Model | Features | Validation Loss | Overall RMSE | Status |
| :--- | :--- | :--- | :--- | :--- |
| **HJ Baseline** | Road + HF + Airbnb | 0.3405 | 0.5053 | Stable |
| **YJ Model (SGIS)** | + SGIS | 0.4529 | N/A | Stalled (Worse) |
| **YJ Model (Final)** | **+ SGIS + Tourism** | **0.3342** | **0.5046** | **Best Performance** |
| **YJ Model (Tourism)** | + Tourism (No SGIS) | 0.3503 | 0.5158 | Worse than Baseline |

**Conclusion:**
*   Adding **SGIS + Tourism** data improves prediction accuracy.
*   Adding **Tourism only** (without SGIS) degrades performance, suggesting SGIS provides necessary context.

## 3. Hyperparameter Tuning Analysis

### 3.1 Window Size (`w`)
We tested if changing the history length affects performance.

*   **w=9 (Baseline):** **RMSE 0.5046**. Best balance of performance and stability.
*   **w=6:** RMSE 0.5099. Slightly worse performance.
*   **w=12:** Failed to converge efficiently. High computational cost with poor initial results.

### 3.2 Embedding Dimensions (`dim_opt`)
We analyzed how internal dimension allocation affects the model.

*   **dim=3 (Optimal):** `[48, 48, 128, 48, 48, 4]`
    *   **Strategy:** High capacity for **Airbnb** (128), low for others (48).
    *   **Result:** Best performance.
*   **dim=1:** `[48, 48, 48, 48, 48, 4]`
    *   **Strategy:** Balanced dimensions.
    *   **Result:** Training stalled.
*   **dim=4:** `[64, 48, 128, 48, 48, 4]`
    *   **Strategy:** Increased Road dimension.
    *   **Result:** Training stalled.

**Insight:** The model relies heavily on the **Airbnb embedding**. Allocating it more capacity (128) while keeping auxiliary features compressed (48) is crucial for stability and performance.

## 4. System Optimization
*   **Memory Issue:** Initial runs with 5 embeddings caused OOM (Out of Memory) errors.
*   **Solution:** Optimized `trainer_log.py` to share tensor references instead of copying data.
*   **Result:** Successfully ran full training with ~17GB RAM usage.
