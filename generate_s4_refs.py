#!/usr/bin/env python3
import sys
sys.path.append('./python')
sys.path.append('./apps')

import numpy as np
import needle as ndl
import needle.nn as nn
from pathlib import Path

np.random.seed(12345)

data_dir = Path('tests/hw4_extra/data')
data_dir.mkdir(parents=True, exist_ok=True)

# Generate S4 layer reference
batch_size, seq_len, dim = 2, 6, 16
hidden_size = 32
state_size = 8

print("Creating S4Layer...")
x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
layer = nn.S4Layer(
    q_features=dim,
    hidden_size=hidden_size,
    state_size=state_size,
    dropout=0.0,
    device=ndl.cpu(),
)
print("Running forward pass...")
layer_out = layer(ndl.Tensor(x, device=ndl.cpu()))
np.save(data_dir / 'test_s4_layer.npy', layer_out.numpy())
print('Saved test_s4_layer.npy')

# Generate S4 model reference
print("Creating S4 model...")
model = nn.S4(
    embedding_size=dim,
    hidden_size=hidden_size,
    num_layers=2,
    state_size=state_size,
    dropout=0.0,
    device=ndl.cpu(),
    batch_first=True,
    sequence_len=seq_len,
)
print("Running forward pass...")
model_out, _ = model(ndl.Tensor(x, device=ndl.cpu()))
np.save(data_dir / 'test_s4_model.npy', model_out.numpy())
print('Saved test_s4_model.npy')
print('Done!')

