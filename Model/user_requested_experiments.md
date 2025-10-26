# User-Requested Feature Selection Experiments

## Clarification: What is "RAW"?
**RAW = Airbnb raw embedding only (659 features)**
- File: `../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv`
- This is the baseline that achieved RMSE = 0.810
- Does NOT include road embeddings or human flow

## Your 5 Experiments

### ✓ Experiment 2: RAW + SGIS (all 6 features) - COMPLETED
**Status**: Already completed! This is what we just ran.
**Configuration**: `--embed1 raw --embed2 sgis_improved`
**Features (6)**:
- retail_ratio
- accommodation_ratio
- restaurant_ratio
- housing_units
- airbnb_listing_count
- airbnb_per_1k_housing

**Result**: RMSE = 1.097 (35.4% worse than raw-only)

---

### Experiment 1: SGIS-only (6 features)
**Purpose**: Test if SGIS has standalone predictive value
**Configuration**: `--embed1 sgis_improved`
**Features (6)**: All SGIS features without raw

**Command**:
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 sgis_improved --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

**Why this matters**: If SGIS-only performs reasonably (e.g., RMSE < 2.0), it shows SGIS features have value that's being masked/diluted by raw features.

---

### Experiment 3: RAW + Two Ratios
**Purpose**: Minimal SGIS features - competition + attractiveness
**Configuration**: `--embed1 raw --embed2 sgis_two_ratios`
**Features (2)**:
- accommodation_ratio (competition)
- retail_ratio (attractiveness)

**Command**:
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_two_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

**Why this matters**: Tests if fewer, focused features perform better than full set. This is the minimal SGIS configuration.

---

### Experiment 4: RAW + Three Ratios
**Purpose**: Business mix without counts
**Configuration**: `--embed1 raw --embed2 sgis_ratios`
**Features (3)**:
- retail_ratio
- accommodation_ratio
- restaurant_ratio

**Command**:
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

**Why this matters**: Pure ratio features are scale-invariant and don't overlap with raw counts.

---

### Experiment 5: RAW + Housing + Three Ratios
**Purpose**: Market size context + business mix
**Configuration**: `--embed1 raw --embed2 sgis_housing_ratios`
**Features (4)**:
- housing_units (market size)
- retail_ratio
- accommodation_ratio
- restaurant_ratio

**Command**:
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_housing_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

**Why this matters**: Tests if adding market size (housing_units) to ratios improves performance vs ratios alone.

---

## Experiment Summary Table

| # | Name | embed1 | embed2 | Features | Status |
|---|------|--------|---------|----------|--------|
| 1 | SGIS-only | sgis_improved | - | 6 | NEW |
| 2 | RAW + All SGIS | raw | sgis_improved | 6 | ✓ DONE (1.097) |
| 3 | RAW + Two Ratios | raw | sgis_two_ratios | 2 | NEW |
| 4 | RAW + Three Ratios | raw | sgis_ratios | 3 | NEW |
| 5 | RAW + Housing + Ratios | raw | sgis_housing_ratios | 4 | NEW |

---

## Running All 4 New Experiments

### Option 1: Run them all in sequence manually
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"

# Experiment 1
python main.py --embed1 sgis_improved --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all

# Experiment 3
python main.py --embed1 raw --embed2 sgis_two_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all

# Experiment 4
python main.py --embed1 raw --embed2 sgis_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all

# Experiment 5
python main.py --embed1 raw --embed2 sgis_housing_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

### Option 2: Run in background (parallel)
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"

# Start all 4 experiments in background
start /B python main.py --embed1 sgis_improved --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all > exp1.log 2>&1
start /B python main.py --embed1 raw --embed2 sgis_two_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all > exp3.log 2>&1
start /B python main.py --embed1 raw --embed2 sgis_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all > exp4.log 2>&1
start /B python main.py --embed1 raw --embed2 sgis_housing_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all > exp5.log 2>&1
```

**Note**: Running in parallel may cause memory issues if system RAM is limited. Sequential is safer.

---

## Expected Timeline
- Each experiment: ~10-15 minutes
- Total for 4 new experiments: ~40-60 minutes (sequential)
- If running parallel: ~15-20 minutes (if enough RAM)

---

## Success Criteria
**Target**: Beat raw-only baseline (RMSE = 0.810)

**Analysis questions**:
1. Does SGIS-only perform reasonably? (Exp 1)
2. Do fewer features work better than 6 features? (Compare Exp 2, 3, 4, 5)
3. Does housing_units add value? (Compare Exp 4 vs Exp 5)
4. What's the optimal feature count? (1, 2, 3, 4, or 6 features?)

---

## Files Created
All feature subset files in: `Preprocess/sgis_manual/`
- `sgis_improved_final.csv` (6 features)
- `sgis_improved_subset_two_ratios.csv` (2 features)  ← NEW
- `sgis_improved_subset_ratios.csv` (3 features)
- `sgis_improved_subset_housing_plus_ratios.csv` (4 features)  ← NEW
