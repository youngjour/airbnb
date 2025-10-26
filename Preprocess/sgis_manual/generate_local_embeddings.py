"""
Generate LLM Embeddings for SGIS Local Features
Uses meta-llama/Llama-3.2-3B-Instruct to create "local embeddings"

Based on Hongju's embedding generation approach from dong_airbnb_llama_embedding.ipynb
"""

import os
import sys
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set print encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def setup_model(device_id=0):
    """
    Initialize Llama model with memory optimizations.

    Args:
        device_id: CUDA device ID (0 for first GPU, 'cpu' for CPU)

    Returns:
        model, tokenizer, device
    """
    print("=" * 80)
    print("SETUP: Initializing Llama-3.2-3B-Instruct Model")
    print("=" * 80)

    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"✓ CUDA available: Using GPU {device_id}")
        print(f"  GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"  Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available: Using CPU (this will be SLOW)")

    # Hugging Face token (set environment variable or edit here)
    hf_token = os.environ.get('HF_TOKEN', '')
    if not hf_token:
        print("\n⚠ Warning: HF_TOKEN not set in environment")
        print("  Set it with: os.environ['HF_TOKEN'] = 'your_token_here'")
        print("  Or get token from: https://huggingface.co/settings/tokens")
        hf_token = input("\nEnter your Hugging Face token (or press Enter to try without): ").strip()

    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    try:
        # Load config with memory optimizations
        print(f"\nLoading model configuration from {model_id}...")
        config = AutoConfig.from_pretrained(
            model_id,
            token=hf_token if hf_token else None
        )
        config.use_cache = False  # Reduce memory usage

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token if hf_token else None
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Load model with optimizations
        print("Loading model (this may take a few minutes)...")
        model = AutoModel.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            token=hf_token if hf_token else None,
            low_cpu_mem_usage=True
        ).to(device)

        print(f"✓ Model loaded successfully")
        print(f"  Embedding dimension: {model.config.hidden_size}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        return model, tokenizer, device

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have access to meta-llama/Llama-3.2-3B-Instruct on Hugging Face")
        print("2. Check your Hugging Face token is valid")
        print("3. Install required packages: pip install transformers torch")
        raise


def process_chunk(text, model, tokenizer, device, chunk_size=512):
    """
    Generate embedding for a text chunk.

    Args:
        text: Input text prompt
        model: Llama model
        tokenizer: Tokenizer
        device: torch device
        chunk_size: Maximum token length

    Returns:
        numpy array of embedding (shape: hidden_size)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=chunk_size,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)

    # Convert to float32 and move to CPU
    embedding = embedding.to(torch.float32).cpu().numpy()

    # Clean up
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embedding


def embed_dataframe(df, model, tokenizer, device, name="local"):
    """
    Generate embeddings for all prompts in dataframe.

    Args:
        df: DataFrame with columns [Reporting Month, Dong_name, prompt]
        model: Llama model
        tokenizer: Tokenizer
        device: torch device
        name: Name for progress bar

    Returns:
        DataFrame with embeddings: [Reporting Month, Dong_name, dim_0, dim_1, ..., dim_3071]
    """
    embeddings = []
    failed_count = 0

    print(f"\nGenerating embeddings for {len(df)} prompts...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {name} prompts"):
        prompt = row['prompt']

        try:
            # Clear cache periodically
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate embedding
            embedding = process_chunk(prompt, model, tokenizer, device)

        except RuntimeError as e:
            # Handle out of memory errors
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    # Try with smaller chunk size
                    embedding = process_chunk(prompt, model, tokenizer, device, chunk_size=256)
                except Exception as e2:
                    print(f"\n⚠ Failed prompt {i} (even with reduced size): {e2}")
                    embedding = None
                    failed_count += 1
            else:
                print(f"\n⚠ Error processing prompt {i}: {e}")
                embedding = None
                failed_count += 1

        except Exception as e:
            print(f"\n⚠ Unexpected error on prompt {i}: {e}")
            embedding = None
            failed_count += 1

        # Add embedding or zero vector if failed
        if embedding is not None:
            embeddings.append(embedding.flatten())
        else:
            # Use zero vector for failed prompts
            embeddings.append(np.zeros((model.config.hidden_size,), dtype=np.float32))

    # Convert to array
    emb_array = np.stack(embeddings)  # Shape: (N, hidden_size)

    # Create column names
    emb_dim_cols = [f"dim_{i}" for i in range(emb_array.shape[1])]

    # Create output dataframe
    emb_df = pd.concat([
        df[['Reporting Month', 'Dong_name']].reset_index(drop=True),
        pd.DataFrame(emb_array, columns=emb_dim_cols)
    ], axis=1)

    print(f"\n✓ Embedding generation complete")
    print(f"  Total prompts: {len(df)}")
    print(f"  Failed: {failed_count}")
    print(f"  Success rate: {100*(len(df)-failed_count)/len(df):.1f}%")

    return emb_df


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SGIS Local Embeddings Generation")
    print("=" * 80 + "\n")

    # Configuration
    INPUT_FILE = 'sgis_local_prompts.csv'
    OUTPUT_FILE = 'sgis_local_llm_embeddings.csv'
    DEVICE_ID = 0  # GPU device ID (change to 1 if using second GPU)

    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        print("   Run generate_local_prompts.py first to create prompts.")
        return

    # Load prompts
    print(f"Loading prompts from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    print(f"✓ Loaded {len(df)} prompts")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['Reporting Month'].min()} to {df['Reporting Month'].max()}")
    print(f"  Unique dongs: {df['Dong_name'].nunique()}")

    # Setup model
    try:
        model, tokenizer, device = setup_model(device_id=DEVICE_ID)
    except Exception as e:
        print(f"\n❌ Failed to setup model. Exiting.")
        return

    # Generate embeddings
    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION")
    print("=" * 80)

    emb_df = embed_dataframe(df, model, tokenizer, device, name="local")

    # Save results
    print(f"\nSaving embeddings to {OUTPUT_FILE}...")
    emb_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"✓ Output saved to: {OUTPUT_FILE}")
    print(f"  Shape: {emb_df.shape}")
    print(f"  Embedding dimensions: {emb_df.shape[1] - 2}")  # Minus ID columns
    print(f"\nThis file can now be used with main.py as 'sgis_local_llm' embedding.")


if __name__ == "__main__":
    # Configure numpy for consistent output
    np.set_printoptions(suppress=True, precision=8, threshold=np.inf)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Partial results may not be saved.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
