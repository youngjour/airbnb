# SGIS Feature Engineering - Performance Comparison

## Model Configuration
- Model: Transformer
- Window size: 9
- Prediction mode: 3-month ahead
- Epochs: 5
- Batch size: 8

## Results Comparison

### Overall Normalized Metrics (Test Set)

| Configuration | Avg RMSE | Avg MAE | Change vs Raw-only | Change vs Old SGIS |
|--------------|----------|---------|-------------------|-------------------|
| **Raw-only baseline** | 0.810 | - | - | - |
| **Old SGIS (counts)** | 1.301 | - | **+60.6% (worse)** | - |
| **Improved SGIS (ratios)** | **1.097** | **0.854** | **+35.4% (worse)** | **-15.7% (better)** |

### Detailed Test Metrics by Label (Normalized)

#### Reservation Days
| Configuration | 1m RMSE | 2m RMSE | 3m RMSE | Avg RMSE |
|--------------|---------|---------|---------|----------|
| Improved SGIS | 1.242 | 0.977 | 1.164 | 1.128 |

#### Revenue (USD)
| Configuration | 1m RMSE | 2m RMSE | 3m RMSE | Avg RMSE |
|--------------|---------|---------|---------|----------|
| Improved SGIS | 1.036 | 1.199 | 1.006 | 1.080 |

#### Number of Reservations
| Configuration | 1m RMSE | 2m RMSE | 3m RMSE | Avg RMSE |
|--------------|---------|---------|---------|----------|
| Improved SGIS | 0.973 | 1.211 | 1.065 | 1.083 |

### Real-Scale Test Metrics

#### Reservation Days (days)
- 1m ahead: RMSE = 814.16, MAE = 403.42
- 2m ahead: RMSE = 747.42, MAE = 220.11
- 3m ahead: RMSE = 872.66, MAE = 254.56
- **Average: RMSE = 811.41, MAE = 292.70**

#### Revenue (USD)
- 1m ahead: RMSE = $62,213, MAE = $16,794
- 2m ahead: RMSE = $62,639, MAE = $25,534
- 3m ahead: RMSE = $67,942, MAE = $18,144
- **Average: RMSE = $64,265, MAE = $20,157**

#### Number of Reservations
- 1m ahead: RMSE = 333.00, MAE = 92.84
- 2m ahead: RMSE = 309.34, MAE = 124.90
- 3m ahead: RMSE = 356.91, MAE = 99.07
- **Average: RMSE = 333.08, MAE = 105.60**

## Feature Engineering Summary

### Old SGIS Features (5 features - absolute counts)
- housing_units
- total_companies
- retail_count
- accommodation_count
- restaurant_count

**Problems:**
- Scale bias (favors large dongs)
- Mixed competitor/attractiveness signals
- No market saturation indicator

### Improved SGIS Features (6 features - ratios + penetration)
- **retail_ratio** (%) - attractiveness indicator
- **accommodation_ratio** (%) - competitor indicator
- **restaurant_ratio** (%) - attractiveness indicator
- **housing_units** - market size context
- **airbnb_listing_count** - absolute supply
- **airbnb_per_1k_housing** - market penetration/saturation

**Improvements:**
✓ Ratios solve scale problem
✓ Clear separation of competitors vs attractiveness
✓ Added market penetration metric
✓ Snapshot broadcast (acknowledges slow change rate)

## Key Findings

### ✅ Success: Improved SGIS beats Old SGIS
- **15.7% improvement** in RMSE (1.097 vs 1.301)
- Feature engineering worked as intended
- Ratios, separation, and penetration metrics are valuable

### ⚠️ Challenge: Still worse than Raw-only
- Improved SGIS: 35.4% worse than raw-only baseline
- Old SGIS: 60.6% worse than raw-only baseline

### Possible Explanations
1. **Information redundancy**: SGIS features may overlap with raw features
2. **Signal dilution**: Additional features add noise without strong predictive power
3. **Model capacity**: Transformer may be overfitting to raw features
4. **Temporal mismatch**: Snapshot broadcast may miss real temporal dynamics
5. **Missing features**: Tourism-specific POIs, accessibility might be more valuable

## Recommendations

### Short-term
1. Try **SGIS-only** model (without raw) to isolate SGIS feature value
2. Test **feature selection** - which of the 6 SGIS features contribute most?
3. Analyze **feature importance** from model attention weights

### Medium-term
1. Add **tourism-specific features** (tourist POIs, transportation access)
2. Collect **temporal housing price data** if available
3. Try **interaction features** (e.g., restaurant_ratio × airbnb_penetration)

### Long-term
1. **LLM-enhanced embeddings** as originally discussed
2. Alternative **data sources** for dynamic neighborhood characteristics
3. **Ensemble methods** combining multiple feature sets

## Data Quality Improvements Made
- Removed 67 duplicate rows (28,341 → 28,274 records)
- Added NaN filling for 13 missing dongs (99.5% coverage)
- Fixed infinity values in penetration rate calculation
- Perfect date alignment (2017-01 to 2022-07, 67 months)

## Experiment Details
- Output directory: `outputs_transformer/transformer_w9_dong_dim3_1023_075716/`
- Training time: ~11 minutes (5 epochs)
- Final validation loss: 1.1459
- Model saved successfully
