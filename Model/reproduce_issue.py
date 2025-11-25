import torch
from transformer_model import EmbeddingTransformer, ModelConfig

def test_initialization():
    # Simulate the condition from the log
    # dim_opt: 3
    # feature_counts: [3072, 3072, 3072, 3072, 3]
    
    input_dims = (3072, 3072, 3072, 3072, 3)
    dim_opt = 3
    
    config = ModelConfig(
        input_dims=input_dims,
        dim_opt=dim_opt,
        mode='3m',
        window_size=9
    )
    
    print(f"Testing with dim_opt={dim_opt}, input_dims={input_dims}, num_inputs={len(input_dims)}")
    
    try:
        model = EmbeddingTransformer(config)
        print("Successfully initialized model.")
        print(f"embedding_dims: {model.embedding_dims}")
    except AttributeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

if __name__ == "__main__":
    test_initialization()
