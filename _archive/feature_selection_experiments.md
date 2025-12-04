# SGIS Feature Selection Experiments

## Goal
Find the optimal feature combination that beats the raw-only baseline (RMSE: 0.810)

## Hypothesis
The improved SGIS features contain valuable information, but some features may be:
1. Redundant with raw features (e.g., airbnb_listing_count overlaps with raw)
2. Adding noise without predictive power
3. Beneficial only in specific combinations

## Experiment Design

### Baseline Reference
- **Raw-only**: RMSE = 0.810 ← Target to beat
- **Raw + All SGIS (6 features)**: RMSE = 1.097 (current result)

### Feature Combinations to Test

#### Experiment 1: SGIS-only (Isolation Test)
**Purpose**: Does SGIS have standalone predictive value?
```bash
--embed1 sgis_improved
```
**Features (6)**: All SGIS features without raw
**Hypothesis**: If this performs reasonably, SGIS features have value being masked by raw

---

#### Experiment 2: Raw + Competition/Saturation
**Purpose**: Test market dynamics indicators
```bash
--embed1 raw --embed2 sgis_improved_subset_competition
```
**Features (2)**:
- accommodation_ratio (competitor density)
- airbnb_per_1k_housing (market saturation)

**Hypothesis**: Market competition and saturation are unique signals not in raw data
**Rationale**: Raw features don't capture competitor density or market saturation rate

---

#### Experiment 3: Raw + Attractiveness Indicators
**Purpose**: Test demand drivers
```bash
--embed1 raw --embed2 sgis_improved_subset_attractiveness
```
**Features (2)**:
- retail_ratio (shopping attractiveness)
- restaurant_ratio (dining/nightlife attractiveness)

**Hypothesis**: Neighborhood attractiveness for tourists is predictive
**Rationale**: Business mix indicates tourist appeal

---

#### Experiment 4: Raw + Business Mix (Ratios Only)
**Purpose**: Test all ratio features without counts
```bash
--embed1 raw --embed2 sgis_improved_subset_ratios
```
**Features (3)**:
- retail_ratio
- accommodation_ratio
- restaurant_ratio

**Hypothesis**: Ratios provide scale-invariant neighborhood characteristics
**Rationale**: Pure ratios without counts avoid redundancy with raw features

---

#### Experiment 5: Raw + Market Saturation Only
**Purpose**: Isolate the penetration metric
```bash
--embed1 raw --embed2 sgis_improved_subset_penetration
```
**Features (1)**:
- airbnb_per_1k_housing

**Hypothesis**: Market saturation is a unique, valuable signal
**Rationale**: This feature is derived from Airbnb data + SGIS, potentially most informative

---

#### Experiment 6: Raw + No Redundancy
**Purpose**: Remove features that overlap with raw data
```bash
--embed1 raw --embed2 sgis_improved_subset_no_redundancy
```
**Features (4)**:
- retail_ratio
- accommodation_ratio
- restaurant_ratio
- airbnb_per_1k_housing

**Removed**:
- housing_units (size info may be in raw)
- airbnb_listing_count (definitely in raw)

**Hypothesis**: Removing redundant features reduces noise
**Rationale**: Raw data already contains Airbnb listing info

---

#### Experiment 7: Raw + Top 2 Features (Data-driven)
**Purpose**: Minimal feature set
```bash
--embed1 raw --embed2 sgis_improved_subset_top2
```
**Features (2)**: To be determined from Experiments 2-6 results

**Hypothesis**: Less is more - minimal features maximize signal-to-noise
**Rationale**: Based on which individual features show best improvement

---

## Execution Plan

### Phase 1: Quick Tests (Batch 1)
Run in parallel:
1. Experiment 1 (SGIS-only)
2. Experiment 5 (Penetration only)
3. Experiment 6 (No redundancy)

**Reason**: These test core hypotheses - standalone value, single feature, and redundancy removal

### Phase 2: Combination Tests (Batch 2)
Based on Phase 1 results:
4. Experiment 2 (Competition/Saturation)
5. Experiment 3 (Attractiveness)
6. Experiment 4 (Ratios only)

### Phase 3: Optimization (Batch 3)
7. Experiment 7 (Top 2 features based on Phases 1-2)
8. Any custom combinations discovered from analysis

---

## Data Preparation

For each subset, create a CSV with only selected features:
- `sgis_improved_subset_competition.csv`
- `sgis_improved_subset_attractiveness.csv`
- `sgis_improved_subset_ratios.csv`
- `sgis_improved_subset_penetration.csv`
- `sgis_improved_subset_no_redundancy.csv`

All subsets maintain:
- Same structure: (Reporting Month, Dong_name, features...)
- Same alignment: 28,274 records (422 dongs × 67 months)
- Same date range: 2017-01-01 to 2022-07-01

---

## Success Criteria

**Must achieve**: RMSE < 0.810 (beat raw-only baseline)
**Good result**: RMSE < 0.750 (significant improvement)
**Excellent result**: RMSE < 0.700 (major improvement)

**Analysis metrics**:
- Overall avg RMSE (primary)
- Per-label RMSE (Reservation Days, Revenue, Reservations)
- Per-horizon RMSE (1m, 2m, 3m)

---

## Expected Insights

From this systematic testing we'll learn:
1. **Do SGIS features have standalone value?** (Exp 1)
2. **Which individual features are most predictive?** (Exp 5 vs others)
3. **Is redundancy the main problem?** (Exp 6 comparison)
4. **What's the optimal feature count?** (Compare 1, 2, 3, 4, 6 features)
5. **Which feature categories work best?** (Competition vs Attractiveness vs Mix)

---

## Next Steps After Results

Based on results, we can:
- **If SGIS-only works**: Consider it as alternative to raw
- **If minimal features work best**: Focus on feature quality over quantity
- **If no combination beats raw**: Conclude SGIS features aren't predictive, pivot to tourism POIs or LLM embeddings
- **If some combination works**: Optimize further with interaction terms
