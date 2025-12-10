# Needle-SSM: S4 State Space Model Implementation

## Project Overview

**needle-ssm** extends the Needle deep learning framework (NumPy-style PyTorch) with a complete S4 (Structured State Space Sequence Model) implementation. S4 replaces attention mechanisms with state-space dynamics, enabling efficient long-sequence modeling.

---

## What Needle Had Originally

- Basic neural network components (Linear, ReLU, Dropout, LayerNorm, Embedding)
- RNN/LSTM sequence models
- Transformer blocks
- Autograd system with CPU/CUDA backends
- Standard operations (matmul, elementwise ops, reductions)
- Optimizers (SGD, Adam)

**Missing:** State Space Models, causal convolution, HiPPO initialization, SSM discretization

---

## What Was Added for S4

### 1. Core S4 Components (`python/needle/nn/nn_ssm.py`)

**`hippo_legs_init(state_size)`**
- Initializes HiPPO-LegS diagonal eigenvalues Λ and B vector
- Λ = [-(i+1) for i in range(state_size)]
- B = [√(2i+1) for i in range(state_size)]

**`S4Layer`** - Single S4 transformer-like block
- **Parameters:**
  - `Lambda` (state_size): Diagonal state matrix (HiPPO initialized)
  - `B` (state_size): Input matrix (HiPPO initialized)
  - `C` (q_features × state_size): Output/readout matrix (learned)
  - `log_step`: Learnable step size for discretization
- **Components:**
  - LayerNorm → Linear projection → Split(u, gate)
  - `discretize()`: Converts continuous SSM to discrete (Ā = e^(ΛΔ), B̄ = (e^(ΛΔ)-I)/Λ * B)
  - `run_ssm(u)`: Computes SSM via parallel convolution
    - Builds kernel: K = Σ(C · A^k · B) over state dimension
    - Calls `causal_conv(u, kernel)` for parallel computation
  - GLU activation (ssm_out * sigmoid(gate))
  - Residual connection + Feed-forward network

**`S4`** - Full model with multiple S4Layer blocks
- Learned positional/time embeddings
- Stack of S4Layer blocks
- Supports batch_first and sequence_len configuration

### 2. Backend Operations (`python/needle/ops/ops_mathematic.py`)

**`CausalConv`** - Custom operator for parallel causal convolution
- **Forward:** `y[b,k,c] = Σ_{j=0}^k u[b,j,c] * kernel[k-j,c]`
- **Backward:** Implements gradient computation via flipped convolution
- Reduces kernel gradient over batch dimension: `(seq_len, channels)` shape

**`causal_conv_backend()`** - Python wrapper calling C++/CUDA kernel

### 3. C++/CUDA Backend (`src/ndarray_backend_cpu.cc`, `src/ndarray_backend_cuda.cu`)

**`CausalConv()`** - Low-level implementation
- **CPU:** Triple nested loop (batch × seq_len × channels)
- **CUDA:** Parallel kernel with thread blocks
- Computes causal convolution: output at position k depends only on inputs ≤ k

### 4. Fixed Operations

**`BroadcastTo.gradient()`** - Fixed shape mismatch bug
- Properly handles broadcasting reversal by:
  - Padding input shape to match output rank
  - Summing only broadcasted dimensions
  - Reshaping through padded shape back to original

---

## Architecture Flow

### S4 Model Forward Pass

```
Input (batch, seq_len, dim)
    ↓
Add learned positional embeddings
    ↓
Stack of S4Layer blocks (num_layers)
    ↓
Output (batch, seq_len, dim)
```

### S4Layer Forward Pass

```
Input x (B, L, D)
    ↓
Save residual
    ↓
LayerNorm → Linear(D → 2D) → Split(u, gate)
    ↓
run_ssm(u):
    1. discretize() → Ā, B̄
    2. Compute kernel: K = Σ(C · A^k · B)
    3. causal_conv(u, K) → ssm_out
    ↓
GLU: ssm_out * sigmoid(gate)
    ↓
Residual + Dropout
    ↓
Feed-forward (D → H → D) + Residual
    ↓
Output (B, L, D)
```

### SSM Core (`run_ssm`)

**Mathematical Formulation:**
- Continuous SSM: ẋ = Ax + Bu, y = Cx
- Discretized: x[k+1] = Āx[k] + B̄u[k], y[k] = Cx[k]
- Convolution form: y = K * u where K[k] = CA^kB

**Implementation:**
1. Compute A^k powers for k = 0 to seq_len-1
2. Broadcast C, A^k, B to (seq_len, channels, state_size)
3. Sum over state dimension: kernel = Σ(C · A^k · B)
4. Apply causal convolution: `causal_conv(u, kernel)`

---

## Key Implementation Details

### Discretization (`discretize()`)
- Converts continuous-time SSM parameters to discrete-time
- For diagonal Λ: Ā = e^(ΛΔ), B̄ = (e^(ΛΔ) - I) / Λ * B
- Uses element-wise division (not matrix inverse) since Λ is diagonal

### Parallel Convolution vs Sequential Recurrence
- **Traditional RNN:** Sequential recurrence O(L) sequential steps
- **S4 approach:** Parallel convolution O(L) parallel operations
- **Speedup:** FFT can accelerate convolution for large kernels (O(L log L) vs O(L²))

### Kernel Computation
- Kernel size: (seq_len, channels)
- Each kernel element: K[k,c] = Σ_{i=0}^{state_size-1} C[c,i] · A[k]^i · B[i]
- Precomputed once per forward pass, reused for all batch elements

---

## Testing (`tests/hw4_extra/test_ssm.py`)

**Test Coverage:**
- `test_hippo_legs_init()`: Validates HiPPO initialization correctness
- `test_s4_discretization()`: Tests discretize() output shapes/values
- `test_s4_layer_forward()`: Full S4Layer forward pass (golden reference)
- `test_s4_model_forward()`: Full S4 model forward pass (golden reference)
- Parametrized tests across state sizes [4, 8, 16] and devices [CPU, CUDA]

**Golden Reference Files:**
- `tests/hw4_extra/data/test_s4_layer.npy`
- `tests/hw4_extra/data/test_s4_model.npy`
- Generated on first run, compared on subsequent runs

---

## Training & Experiments (`train_SSM_RNN.py`)

**Sentinel Detection Task:**
- **Dataset:** `SentinelDataset` - sequences with optional sentinel token
- **Task:** Binary classification (sentinel present/absent)
- **Models:**
  - `RNNClassifierLM`: Baseline RNN with embedding → RNN → pooling → linear
  - `S4Classifier`: S4 model with embedding → S4 → pooling → linear

**Training:**
- Binary cross-entropy loss with logits
- Adam optimizer
- Supports configurable: seq_len, vocab_size, batch_size, epochs, hidden/embedding sizes
- Compares RNN vs S4 performance on same task

**Usage:**
```bash
python3 train_SSM_RNN.py --seq_len 10 --vocab 50 --train_size 100 --val_size 50 \
    --batch 2 --epochs 10 --hidden 64 --emb 32
```

---

## Model Presets (`apps/models.py`)

**`create_s4_model(size, ...)`** - Pre-configured S4 models:
- **small:** ~10M params (512 dim, 4 layers, 2048 hidden)
- **medium:** ~30M params (768 dim, 6 layers, 3072 hidden)
- **large:** ~50M params (1024 dim, 6 layers, 4096 hidden)
- **xlarge:** ~100M params (1024 dim, 12 layers, 4096 hidden)

**`count_parameters(model)`** - Counts trainable parameters

---

## Performance Characteristics

### Computational Complexity
- **Forward:** O(L × channels × state_size) for kernel computation + O(L × channels) for convolution
- **Memory:** O(L × channels × state_size) for kernel storage
- **Backward:** Similar complexity via autograd

### Advantages Over RNNs
- **Parallelization:** Convolution enables parallel computation vs sequential recurrence
- **Long-range dependencies:** HiPPO initialization captures long-range patterns
- **Scalability:** FFT optimization (future) enables O(L log L) for long sequences

### Current Limitations
- Kernel computation requires O(L) memory for A^k powers
- FFT acceleration not yet implemented (uses direct convolution)
- State size typically fixed at 64 (trade-off between expressiveness and efficiency)

---

## File Structure

```
needle-ssm/
├── python/needle/
│   ├── nn/nn_ssm.py          # S4Layer, S4, hippo_legs_init
│   └── ops/ops_mathematic.py # CausalConv operator
├── src/
│   ├── ndarray_backend_cpu.cc   # CPU causal_conv implementation
│   └── ndarray_backend_cuda.cu  # CUDA causal_conv implementation
├── tests/hw4_extra/
│   ├── test_ssm.py           # S4 unit tests
│   └── data/                 # Golden reference outputs
├── train_SSM_RNN.py         # RNN vs S4 comparison training
├── apps/models.py            # S4 model presets
└── README.md                 # Usage documentation
```

---

## Key Innovations

1. **Parallel SSM Computation:** Converts sequential recurrence to parallel convolution
2. **Custom Backend Kernel:** C++/CUDA implementation for efficient causal convolution
3. **HiPPO Initialization:** Structured initialization for long-range dependency modeling
4. **Gradient Fixes:** Fixed BroadcastTo gradient to handle SSM shape requirements
5. **End-to-End Integration:** Full model, training, and testing pipeline

---

## Future Optimizations

- **FFT-based Convolution:** Replace direct convolution with FFT for O(L log L) complexity
- **Kernel Truncation:** Approximate long kernels for memory efficiency
- **Multi-head SSM:** Parallel SSM channels for increased capacity
- **Learnable HiPPO:** Make initialization parameters learnable

---

## Dependencies

- Python 3
- NumPy
- C++/CUDA compiler (for backend)
- pytest (for testing)
- tqdm (for training progress)

---

## Build Instructions

```bash
make clean
make lib  # Builds CPU and CUDA backends
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

---

## Summary

This project adds a complete S4 implementation to Needle, including:
- **Core:** S4Layer and S4 model classes
- **Backend:** Custom causal convolution operator (CPU/CUDA)
- **Initialization:** HiPPO-LegS for structured state matrices
- **Training:** End-to-end RNN vs S4 comparison
- **Testing:** Comprehensive unit tests with golden references

The implementation enables efficient long-sequence modeling through parallel convolution, replacing sequential RNN recurrence with parallelizable operations.


