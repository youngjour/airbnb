# Feature Selection - Quick Command Reference

## Baselines (for comparison)
- **Raw-only**: RMSE = 0.810 ← **Target to beat**
- **Old SGIS**: RMSE = 1.301
- **Raw + All SGIS**: RMSE = 1.097 (completed)

---

## Phase 1: Critical Tests

### Experiment 1: SGIS-only (Isolation Test)
**Test if SGIS has standalone value**
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 sgis_improved --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```
Features: 6 (all SGIS features)

### Experiment 2: Raw + Penetration Only
**Test market saturation metric alone**
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_penetration --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```
Features: 1 (airbnb_per_1k_housing)

### Experiment 3: Raw + No Redundancy
**Remove features that overlap with raw**
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_no_redundancy --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```
Features: 4 (retail_ratio, accommodation_ratio, restaurant_ratio, airbnb_per_1k_housing)

---

## Phase 2: Category Tests

### Experiment 4: Raw + Competition/Saturation
**Test market dynamics indicators**
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_competition --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```
Features: 2 (accommodation_ratio, airbnb_per_1k_housing)

### Experiment 5: Raw + Attractiveness
**Test tourist demand drivers**
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_attractiveness --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```
Features: 2 (retail_ratio, restaurant_ratio)

### Experiment 6: Raw + Ratios Only
**Test business mix without counts**
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python main.py --embed1 raw --embed2 sgis_ratios --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```
Features: 3 (retail_ratio, accommodation_ratio, restaurant_ratio)

---

## Automated Approach

Run all experiments automatically:
```bash
cd "C:\Users\jour\Documents\GitHub\airbnb\Model"
python run_feature_selection_experiments.py
```

This will:
- Run all 6 experiments sequentially
- Extract RMSE from each
- Create comparison table
- Identify best performing combination
- Save results to CSV

Estimated time: ~60-90 minutes total

---

## Results Analysis

After each experiment completes, check:
1. **Overall RMSE**: `outputs_transformer/[latest_dir]/results.json`
   - Look for: `overall_real_metrics.overall_avg_RMSE`
2. **Compare to baseline**: RMSE < 0.810 means success!
3. **Per-label breakdown**: See which predictions improved most

---

## Feature Subsets Available

| Subset Name | Features | File |
|-------------|----------|------|
| competition | 2 | sgis_improved_subset_competition.csv |
| attractiveness | 2 | sgis_improved_subset_attractiveness.csv |
| ratios | 3 | sgis_improved_subset_ratios.csv |
| penetration | 1 | sgis_improved_subset_penetration.csv |
| no_redundancy | 4 | sgis_improved_subset_no_redundancy.csv |
| improved (full) | 6 | sgis_improved_final.csv |

All files located in: `Preprocess/sgis_manual/`

---

## Expected Outcomes

**If Penetration (1 feature) works best:**
- Market saturation is the key signal
- Focus on Airbnb-specific metrics

**If No Redundancy (4 features) works best:**
- Redundancy was the main problem
- Raw data overlaps with counts

**If SGIS-only performs reasonably:**
- SGIS has value being masked by raw
- Consider ensemble or alternative architecture

**If nothing beats raw-only:**
- SGIS features aren't predictive for this task
- Pivot to tourism POIs or LLM embeddings
