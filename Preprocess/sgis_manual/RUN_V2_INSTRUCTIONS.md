# Generate V2 SGIS Local LLM Embeddings

## FIXED: Accelerate Library Error
The script has been updated to remove the `device_map="auto"` parameter that required the `accelerate` library. It now uses a simple `model.to(device)` approach.

## To Generate V2 Embeddings:

### Option 1: Run the batch file (Recommended)
1. Open PowerShell or Command Prompt
2. Navigate to this directory:
   ```
   cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual
   ```
3. Run the batch file:
   ```
   run_v2_embeddings.bat
   ```

### Option 2: Manual activation
1. Open PowerShell or Command Prompt
2. Activate the conda environment:
   ```
   conda activate yj_pytorch
   ```
3. Navigate to the directory:
   ```
   cd C:\Users\jour\Documents\GitHub\airbnb\Preprocess\sgis_manual
   ```
4. Run the script:
   ```
   python generate_local_embeddings_v2.py
   ```

## Expected Output:
- Input: `sgis_local_prompts_v2.csv` (28,274 prompts)
- Output: `sgis_local_llm_embeddings_v2.csv` (28,274 rows × 3,074 columns)
- Embedding dimensions: 3,072 (Llama-3.2-3B hidden size)
- Estimated time: 30-45 minutes on RTX 4070 GPU

## After Generation:
The next step will be to test the model with the v2 embeddings:
```bash
cd C:\Users\jour\Documents\GitHub\airbnb\Model
python main.py --embed1 road_llm --embed2 hf_llm --embed3 llm_w --embed4 sgis_local_llm_v2 --model transformer --epochs 5 --batch_size 8 --window_size 9 --mode 3m --label all
```

## Comparison Target:
- HJ Baseline (road + hf + airbnb): RMSE = 0.505
- HJ + Local v1: RMSE = 0.531 (5.1% worse)
- HJ + Local v2: TBD (expecting improvement with better prompts)
