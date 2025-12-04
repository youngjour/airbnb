# Correct Experimental Setup - Adding SGIS to HJ Baseline

## Understanding the Real Goal

### HJ Baseline Model (from Hongju's repository)
**Features**:
- LLM embedding of road network
- LLM embedding of human flow
- LLM embedding of Airbnb features

**Performance**: RMSE = ??? (need to identify which experiment this is)

### Your Enhancement Goal
**Add SGIS census features to HJ baseline**:
- Accommodation ratio (competitor indicator)
- Retail ratio (tourism attraction)
- Restaurant ratio (tourism attraction)
- Number of houses (market size)

**Method**: Convert SGIS features to LLM embeddings (same method as HJ baseline)

---

## Questions to Clarify

### 1. What is the exact HJ baseline configuration?

**Option A**: LLM embeddings only
```bash
--embed1 llm_w --embed2 road_llm --embed3 hf_llm
```

**Option B**: LLM + Raw
```bash
--embed1 raw --embed2 llm_w --embed3 road_llm (or some combination)
```

**Option C**: Something else?

**Please specify**: What command produces the HJ baseline that you want to beat?

---

### 2. How are LLM embeddings created?

Looking at available files:
- `llm_w`: LLM embedding with something
- `llm_wo`: LLM embedding without something
- `road_llm`: LLM embedding of road network
- `hf_llm`: LLM embedding of human flow

**Questions**:
- What LLM model/method was used? (GPT, Claude, BERT, etc.)
- What's the input format to the LLM?
- What's the output embedding dimension?
- Can you share the script/method used to create these embeddings?

This will help us create LLM embeddings for SGIS features.

---

### 3. What's the HJ baseline performance?

Need to identify which experiment directory contains the HJ baseline results.

**Please provide**:
- RMSE of HJ baseline
- Which output directory contains it?
- Or the exact command to reproduce it?

---

## Proposed Experimental Approach

### Phase 1: Establish HJ Baseline
1. Identify/reproduce the exact HJ baseline
2. Record its performance (RMSE)
3. This is our target to beat

### Phase 2: Create SGIS LLM Embeddings
1. Format SGIS features for LLM input
2. Generate LLM embeddings using same method as HJ baseline
3. Save as new embedding file (e.g., `sgis_llm.csv`)

### Phase 3: Test SGIS Enhancement
Compare these configurations:

**Experiment A: HJ Baseline (reproduce)**
```bash
[HJ baseline command - to be specified]
```

**Experiment B: HJ Baseline + SGIS Raw**
```bash
[HJ baseline embeddings] + --embedX sgis_improved
```

**Experiment C: HJ Baseline + SGIS LLM** (preferred)
```bash
[HJ baseline embeddings] + --embedX sgis_llm
```

**Experiment D: Feature selection on SGIS**
Test different SGIS feature subsets with HJ baseline:
- HJ + SGIS (2 ratios: accommodation + retail)
- HJ + SGIS (3 ratios: all business mix)
- HJ + SGIS (4 features: housing + ratios)

---

## What We've Done So Far (and why it's not quite right)

### Current Work Summary:
✓ Collected improved SGIS features (ratios + penetration)
✓ Created 6 SGIS features with good engineering
✓ Tested: RAW + SGIS (RMSE = 1.097)

### Why this doesn't answer your question:
- "RAW" might not be the HJ baseline
- Wepared against "raw-only" (0.810) which might not be the right baseline
- SGIS features are not LLM-embedded yet

---

## Next Steps (after clarification)

1. **You provide**:
   - Exact HJ baseline command/config
   - HJ baseline performance (RMSE)
   - Method for creating LLM embeddings

2. **I will**:
   - Reproduce HJ baseline
   - Create LLM embeddings for SGIS features
   - Run proper comparison experiments
   - Show whether SGIS improves HJ baseline

---

## Immediate Questions

1. **What is the exact command for HJ baseline?**
   ```bash
   python main.py --embed1 ??? --embed2 ??? --embed3 ??? ...
   ```

2. **What is the HJ baseline RMSE?**
   - Need this as our target to beat

3. **How do you create LLM embeddings?**
   - What's the process/script?
   - Can you share the method?

Once you provide these, I can design the correct experiments to test your hypothesis!
