# Tmap Tourism Navigation Search Data - Integration Plan

## Data Overview

**Source**: Tmap Mobility - Navigation destination search data
**Type**: Actual navigation searches + movement confirmation (>100m, >1min)
**Coverage**: 2018.01 - present (user confirmed)
**Current sample**: 2020.01 - 2020.12 (2,819 rows)

**Geographic**: 25 Seoul gu (구) districts
**Temporal**: Monthly aggregation
**Categories**: 10 search categories

### Category Structure
1. **전체** (Total) - All searches combined
2. **자연관광** (Natural tourism) - Mountains, parks, nature sites
3. **역사관광** (Historical tourism) - Palaces, heritage sites
4. **체험관광** (Experience tourism) - Cultural experiences
5. **문화관광** (Cultural tourism) - Museums, galleries
6. **레저스포츠** (Leisure/Sports) - Sports, recreation
7. **쇼핑** (Shopping) - Markets, shopping centers
8. **음식** (Food/Dining) - Restaurants, cafes
9. **숙박** (Accommodation) - Hotels, guesthouses
10. **기타관광** (Other tourism) - Miscellaneous

---

## Why This Data is Valuable for Airbnb Demand Prediction

### Strong Theoretical Foundation
1. **Intent Signal**: People searching for destinations likely need accommodation nearby
2. **Temporal Variation**: Captures seasonal tourism patterns
3. **Category Richness**: Different destination types attract different Airbnb guests
4. **Behavioral Data**: Actual navigation use (not just passive interest)

### Hypothesis
**Tourism destination search volume in a district predicts Airbnb demand** because:
- Tourists want to stay close to their planned activities
- Different categories appeal to different guest types:
  - Shopping/Dining → Short-term leisure travelers
  - Cultural/Historical → International tourists
  - Leisure/Sports → Weekend domestic travelers

---

## Implementation Workflow

### Step 1: Data Preprocessing (Gu → Dong Distribution)

**Challenge**: Data at gu level (25 districts), need dong level (422 dongs)

**Solution**: Population-weighted distribution
```
For each (gu, month, category):
  1. Get dong-to-gu mapping from SGIS
  2. Get population by dong (or use existing Airbnb density as proxy)
  3. For each dong in that gu:
     dong_search_count = gu_search_count × (dong_weight / gu_total_weight)
```

**Alternative**: Airbnb-density weighted distribution
- Use existing Airbnb listing counts to distribute searches
- Assumption: More Airbnb listings = higher tourism appeal

### Step 2: Feature Engineering

**Raw features** (9 categories + total = 10 dimensions per month):
- Total searches
- Natural tourism searches
- Historical tourism searches
- Cultural tourism searches
- Leisure/Sports searches
- Shopping searches
- Dining searches
- Accommodation searches
- Other tourism searches

**Engineered features** (reduce to 5-7 key indicators):
```python
# Tourism intensity
tourism_intensity = total_searches / dong_population

# Category mix (what kind of destination is this?)
cultural_ratio = (historical + cultural) / total_searches
leisure_ratio = (shopping + dining + leisure) / total_searches
nature_ratio = natural_tourism / total_searches

# Growth trends
mom_growth = (current_month - previous_month) / previous_month
seasonal_index = current_month / annual_average

# Accommodation interest
accommodation_ratio = accommodation_searches / total_searches
```

### Step 3: LLM Prompt Generation

**Prompt template**:
```
"In {month} {year}, {dong_name} in {gu_name} received {total_searches:,} tourism-related
navigation searches, ranking {rank_in_seoul} among all Seoul neighborhoods.

The destination mix shows {interpretation}:
- Cultural/Historical sites: {cultural_pct}% ({cultural_interpretation})
- Shopping & Dining: {leisure_pct}% ({leisure_interpretation})
- Nature & Recreation: {nature_pct}% ({nature_interpretation})

This represents a {growth_pct}% {increase/decrease} from the previous month, suggesting
{seasonal_interpretation}. The high proportion of {dominant_category} searches indicates
this area appeals primarily to {tourist_profile}.

For Airbnb hosts, this tourism pattern suggests {airbnb_opportunity}."
```

**Example**:
```
"In June 2020, Sinsa-dong in Gangnam-gu received 45,230 tourism navigation searches,
ranking 3rd among all Seoul neighborhoods.

The destination mix shows strong lifestyle tourism appeal:
- Cultural/Historical sites: 15% (moderate interest in heritage)
- Shopping & Dining: 68% (dominant attraction - trendy cafes and boutiques)
- Nature & Recreation: 12% (limited outdoor attractions)

This represents a 23% increase from May, suggesting peak summer tourism season.
The high proportion of dining and shopping searches indicates this area appeals
primarily to young leisure travelers and Instagram tourists.

For Airbnb hosts, this tourism pattern suggests strong demand for stylish,
centrally-located accommodations near Garosu-gil and Apgujeong shopping areas."
```

### Step 4: Generate LLM Embeddings

Use same methodology as local SGIS embeddings:
- Model: meta-llama/Llama-3.2-3B-Instruct
- Embedding dimension: 3,072
- Method: Mean pooling over last hidden state
- Output: `tmap_tourism_llm_embeddings.csv`

### Step 5: Model Integration

**Configuration**:
```bash
python main.py \
  --embed1 road_llm \
  --embed2 hf_llm \
  --embed3 llm_w \
  --embed4 sgis_local_llm_v2 \
  --embed5 tmap_tourism_llm \
  --model transformer \
  --epochs 5 \
  --batch_size 8 \
  --window_size 9 \
  --mode 3m \
  --label all
```

**Expected architecture**:
- 5 LLM embeddings + 1 preprocessing artifact = 6 inputs
- Need to add `--embed5` support to main.py
- Need 6-input configurations to transformer_model.py

---

## Next Steps

### Immediate Actions Needed from User

1. **Provide complete Tmap data**: 2018.01 - 2021.11 (not just 2020 sample)
2. **Confirm dong-to-gu mapping**: Either extract from SGIS or provide separate file
3. **Decide on distribution method**: Population-weighted or Airbnb-density-weighted?

### Implementation Sequence

1. ✓ Analyze current sample data structure
2. [ ] Extract dong-to-gu mapping from SGIS
3. [ ] Create preprocessing script (gu → dong distribution)
4. [ ] Engineer tourism features from 9 categories
5. [ ] Generate tourism-specific LLM prompts
6. [ ] Generate Tmap LLM embeddings (Llama 3.2)
7. [ ] Add `--embed5` support to model code
8. [ ] Test model with all 5 embeddings
9. [ ] Compare performance vs HJ baseline + local v2

---

## Expected Performance Impact

**Hypothesis**: Adding Tmap tourism data should improve RMSE by 1-3%

**Reasoning**:
- Local SGIS v2 improved by 0.47% (RMSE 0.5287 vs 0.5312)
- Tmap data provides complementary signal (temporal tourism trends)
- Expected combined benefit: 0.5% (local static) + 0.5-1.5% (tourism temporal) = 1-2% total

**Target**: RMSE < 0.520 (vs baseline 0.5312)

---

## Questions for User

1. Can you provide the complete Tmap dataset (2018.01-2021.11)?
2. What distribution method do you prefer (population vs Airbnb density)?
3. Should we proceed with implementation once we have the complete dataset?
