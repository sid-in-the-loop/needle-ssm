import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        raise NotImplementedError() ###
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            # Calculate total number of elements from shape
            total += np.prod(param.shape)
    return int(total)


def create_s4_model(
    size: str = "small",
    *,
    device=None,
    dtype: str = "float32",
    dropout: float = 0.1,
    sequence_len: int = 2048,
):
    """
    Create an S4 model with preset configurations.
    
    Available sizes:
    - "small": ~10M parameters (embedding_size=512, hidden_size=2048, num_layers=4, state_size=64)
    - "medium": ~30M parameters (embedding_size=768, hidden_size=3072, num_layers=6, state_size=64)
    - "large": ~50M parameters (embedding_size=1024, hidden_size=4096, num_layers=6, state_size=64)
    - "xlarge": ~100M parameters (embedding_size=1024, hidden_size=4096, num_layers=12, state_size=64)
    
    Args:
        size: Model size preset ("small", "medium", "large", "xlarge")
        device: Device to run on
        dtype: Data type
        dropout: Dropout rate
        sequence_len: Maximum sequence length
    
    Returns:
        S4 model instance
    """
    configs = {
        "small": {
            "embedding_size": 512,
            "hidden_size": 2048,
            "num_layers": 4,
            "state_size": 64,
        },
        "medium": {
            "embedding_size": 768,
            "hidden_size": 3072,
            "num_layers": 6,
            "state_size": 64,
        },
        "large": {
            "embedding_size": 1024,
            "hidden_size": 4096,
            "num_layers": 6,
            "state_size": 64,
        },
        "xlarge": {
            "embedding_size": 1024,
            "hidden_size": 4096,
            "num_layers": 12,
            "state_size": 64,
        },
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(configs.keys())}")
    
    config = configs[size]
    
    model = nn.S4(
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        state_size=config["state_size"],
        dropout=dropout,
        device=device,
        dtype=dtype,
        batch_first=True,
        sequence_len=sequence_len,
    )
    
    return model


if __name__ == "__main__":
    # Test S4 model creation and parameter counting
    print("Creating S4 models...")
    
    for size in ["small", "medium", "large", "xlarge"]:
        model = create_s4_model(size, device=ndl.cpu())
        param_count = count_parameters(model)
        print(f"S4-{size}: {param_count:,} parameters ({param_count/1e6:.2f}M)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model = create_s4_model("small", device=ndl.cpu())
    batch_size, seq_len, dim = 2, 128, 512
    x = ndl.Tensor(np.random.randn(batch_size, seq_len, dim).astype(np.float32), device=ndl.cpu())
    out, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ S4 model forward pass successful!")
