"""
Generate LLM Embeddings from Improved SGIS Local Prompts (v2)
Uses identical methodology to v1 but processes improved Airbnb-specific prompts.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
from datetime import datetime

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT_FILE = "sgis_local_prompts_v2.csv"
OUTPUT_FILE = "sgis_local_llm_embeddings_v2.csv"
BATCH_SIZE = 16
CHUNK_SIZE = 512

def load_model_and_tokenizer(device):
    """Load Llama model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set pad_token to eos_token (required for Llama models)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )

    # Move model to device
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded successfully")
    print(f"[OK] Hidden size: {model.config.hidden_size}")

    return model, tokenizer

def generate_embedding(text, model, tokenizer, device):
    """Generate embedding for a single text using mean pooling."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=CHUNK_SIZE,
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)

    # Convert to float32 for storage
    embedding = embedding.to(torch.float32).cpu().numpy()

    return embedding[0]

def process_batch(texts, model, tokenizer, device):
    """Process a batch of texts."""
    embeddings = []
    for text in texts:
        emb = generate_embedding(text, model, tokenizer, device)
        embeddings.append(emb)
    return np.array(embeddings)

def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[OK] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"[OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("[WARNING] No GPU available, using CPU (this will be slower)")

    # Load prompts
    print(f"\nLoading prompts from: {PROMPT_FILE}")
    df = pd.read_csv(PROMPT_FILE, encoding='utf-8-sig')
    print(f"[OK] Loaded {len(df)} prompts")
    print(f"[OK] Columns: {df.columns.tolist()}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(device)
    embedding_dim = model.config.hidden_size

    # Process prompts
    print(f"\nGenerating embeddings (batch size: {BATCH_SIZE})...")
    print(f"Expected output dimensions: {len(df)} x {embedding_dim}")

    all_embeddings = []
    start_time = datetime.now()

    for i in range(0, len(df), BATCH_SIZE):
        batch_texts = df['prompt'].iloc[i:i+BATCH_SIZE].tolist()
        batch_embeddings = process_batch(batch_texts, model, tokenizer, device)
        all_embeddings.append(batch_embeddings)

        # Progress update
        processed = min(i + BATCH_SIZE, len(df))
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (len(df) - processed) / rate if rate > 0 else 0

        if processed % 100 == 0 or processed == len(df):
            print(f"  Progress: {processed}/{len(df)} ({100*processed/len(df):.1f}%) | "
                  f"Rate: {rate:.1f} prompts/sec | "
                  f"ETA: {remaining/60:.1f} min")

    # Combine all embeddings
    embeddings_array = np.vstack(all_embeddings)
    print(f"\n[OK] Generated embeddings shape: {embeddings_array.shape}")

    # Create output DataFrame
    print("\nCreating output DataFrame...")
    output_data = {
        'Reporting Month': df['Reporting Month'],
        'Dong_name': df['Dong_name']
    }

    # Add embedding dimensions
    for i in range(embedding_dim):
        output_data[f'dim_{i}'] = embeddings_array[:, i]

    output_df = pd.DataFrame(output_data)

    # Save to CSV
    print(f"Saving to: {OUTPUT_FILE}")
    output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*100}")
    print(f"EMBEDDING GENERATION COMPLETE (v2 - Improved Prompts)")
    print(f"{'='*100}")
    print(f"Input file: {PROMPT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Output shape: {output_df.shape}")
    print(f"Embedding dimensions: {embedding_dim}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {len(df)/total_time:.2f} prompts/second")
    print(f"\nOutput columns: {output_df.columns.tolist()[:5]} ... {output_df.columns.tolist()[-3:]}")
    print(f"\nSample data:")
    print(output_df.head(3).to_string())

    # Verify embeddings
    print(f"\nEmbedding statistics:")
    embedding_cols = [col for col in output_df.columns if col.startswith('dim_')]
    sample_embeddings = output_df[embedding_cols].values
    print(f"  Mean: {sample_embeddings.mean():.6f}")
    print(f"  Std: {sample_embeddings.std():.6f}")
    print(f"  Min: {sample_embeddings.min():.6f}")
    print(f"  Max: {sample_embeddings.max():.6f}")

if __name__ == "__main__":
    main()
