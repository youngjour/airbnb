"""
Generate LLM embeddings for HJ baseline features using Llama-3.2-3B-Instruct

Generates embeddings for:
- Airbnb features (llm_w, llm_wo)
- Road network (road_llm)
- Human flow (hf_llm)

Based on successful SGIS local embeddings generation approach.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
import gc

print("="*80)
print("HJ BASELINE LLM EMBEDDINGS GENERATION")
print("="*80)

# Configuration
EMBEDDING_CONFIG = {
    'llm_wo': {
        'prompt_file': '../dong_prompts_new/AirBnB_SSP_wo_prompts.csv',
        'output_file': '../../../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_wo.csv',
        'name': 'Airbnb (without listings)'
    },
    'llm_w': {
        'prompt_file': '../dong_prompts_new/AirBnB_SSP_w_prompts.csv',
        'output_file': '../../../Data/Preprocessed_data/Dong/llm_embeddings_new/Airbnb_SSP_w.csv',
        'name': 'Airbnb (with listings)'
    },
    'road_llm': {
        'prompt_file': '../dong_prompts/road_prompts.csv',
        'output_file': '../../../Data/Preprocessed_data/Dong/llm_embeddings_new/road_llm.csv',
        'name': 'Road network'
    },
    'hf_llm': {
        'prompt_file': '../dong_prompts/human_flow_prompts.csv',
        'output_file': '../../../Data/Preprocessed_data/Dong/llm_embeddings_new/human_flow_llm.csv',
        'name': 'Human flow'
    }
}

# Check command line argument
if len(sys.argv) > 1:
    embedding_type = sys.argv[1]
    if embedding_type not in EMBEDDING_CONFIG:
        print(f"Error: Unknown embedding type '{embedding_type}'")
        print(f"Valid types: {', '.join(EMBEDDING_CONFIG.keys())}")
        sys.exit(1)
    TYPES_TO_GENERATE = [embedding_type]
else:
    # Generate all if no argument provided
    TYPES_TO_GENERATE = list(EMBEDDING_CONFIG.keys())

print(f"\nEmbedding types to generate: {', '.join(TYPES_TO_GENERATE)}")
print(f"Total: {len(TYPES_TO_GENERATE)} embedding files")

# Model setup
def setup_model(device_id=0):
    """Initialize Llama model with memory optimizations."""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Get Hugging Face token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("\nError: HF_TOKEN environment variable not set!")
        print("Please set your Hugging Face token:")
        print("  Windows: set HF_TOKEN=your_token_here")
        print("  Linux/Mac: export HF_TOKEN=your_token_here")
        sys.exit(1)

    print(f"\nLoading model: {model_id}")
    print("This may take a few minutes on first run (downloading ~6GB model)...")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device('cpu')
        print("[Warning] CUDA not available, using CPU (will be slower)")

    # Load model config
    config = AutoConfig.from_pretrained(model_id, token=hf_token)
    config.use_cache = False  # Disable KV cache for memory efficiency

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with memory optimizations
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        token=hf_token,
        low_cpu_mem_usage=True
    ).to(device)

    model.eval()  # Set to evaluation mode

    print(f"[OK] Model loaded successfully")
    print(f"  Embedding dimension: {config.hidden_size}")

    return model, tokenizer, device

def process_chunk(text, model, tokenizer, device, chunk_size=512):
    """Generate embedding for a text chunk."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=chunk_size,
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)

    # Convert to float32 for compatibility and move to CPU
    embedding = embedding.to(torch.float32).cpu().numpy()

    return embedding

def embed_dataframe(df, model, tokenizer, device, name="prompts"):
    """Generate embeddings for all prompts in dataframe."""
    print(f"\nGenerating embeddings for {len(df)} {name}...")

    embeddings = []
    failed_indices = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {name}"):
        try:
            embedding = process_chunk(row['prompt'], model, tokenizer, device)
            embeddings.append(embedding.flatten())

            # Periodic GPU cache clearing
            if (i + 1) % 1000 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\n[Error] Failed to process row {i}: {e}")
            failed_indices.append(i)
            # Use zero embedding for failed cases
            embeddings.append(np.zeros(3072))

    # Stack embeddings
    emb_array = np.stack(embeddings)  # Shape: (N, 3072)

    print(f"\n[OK] Embedding generation complete")
    print(f"  Total prompts: {len(df)}")
    print(f"  Failed: {len(failed_indices)}")
    print(f"  Success rate: {100*(1-len(failed_indices)/len(df)):.1f}%")

    # Create dimension column names
    emb_dim_cols = [f"dim_{i}" for i in range(emb_array.shape[1])]

    # Combine with original date and dong info
    result_df = pd.concat([
        df[['Reporting Month', 'Dong_name']],
        pd.DataFrame(emb_array, columns=emb_dim_cols)
    ], axis=1)

    return result_df

# Main execution
def main():
    # Setup model (load once, use for all embeddings)
    model, tokenizer, device = setup_model()

    # Process each embedding type
    for emb_type in TYPES_TO_GENERATE:
        config = EMBEDDING_CONFIG[emb_type]

        print("\n" + "="*80)
        print(f"Processing: {config['name']} ({emb_type})")
        print("="*80)

        # Check if prompt file exists
        prompt_path = config['prompt_file']
        if not os.path.exists(prompt_path):
            print(f"[Error] Prompt file not found: {prompt_path}")
            print("Skipping this embedding type...")
            continue

        # Load prompts
        print(f"\nLoading prompts from: {prompt_path}")
        df_prompts = pd.read_csv(prompt_path)
        print(f"[OK] Loaded {len(df_prompts)} prompts")
        print(f"  Columns: {df_prompts.columns.tolist()}")

        # Verify columns
        if 'prompt' not in df_prompts.columns:
            print(f"[Error] 'prompt' column not found in {prompt_path}")
            continue

        # Generate embeddings
        df_embeddings = embed_dataframe(df_prompts, model, tokenizer, device, config['name'])

        # Save embeddings
        output_path = config['output_file']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\nSaving embeddings to: {output_path}")
        df_embeddings.to_csv(output_path, index=False)
        print(f"[OK] Saved successfully")
        print(f"  Shape: {df_embeddings.shape}")
        print(f"  Expected: ({len(df_prompts)}, 3074)")

        # Verify output
        if df_embeddings.shape == (len(df_prompts), 3074):
            print(f"[OK] Output shape verified!")
        else:
            print(f"[Warning] Output shape mismatch!")

    print("\n" + "="*80)
    print("ALL EMBEDDINGS COMPLETE!")
    print("="*80)
    print(f"\nGenerated {len(TYPES_TO_GENERATE)} embedding files:")
    for emb_type in TYPES_TO_GENERATE:
        config = EMBEDDING_CONFIG[emb_type]
        print(f"  - {config['name']}: {config['output_file']}")

    print("\nNext step: Test HJ baseline + local embeddings with model")

if __name__ == "__main__":
    main()
