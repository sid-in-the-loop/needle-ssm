## needle-ssm: S4 / State Space Models

This repo is a small deep learning framework (NumPy-style PyTorch) with CPU/CUDA backends, plus a custom **S4-style State Space Model (SSM)** layer and model.

The S4 code lives in:
- `python/needle/nn/nn_ssm.py`  → `S4Layer`, `S4`, and HiPPO-style init
- `tests/hw4_extra/test_ssm.py` → tests for S4 pieces
- `apps/models.py`              → ready-made S4 model presets

All examples below assume you are in the repo root: `needle-ssm/`.

---

## 1. Quick setup (minimal)

You just need:
- Python 3
- A C/C++ toolchain (for the ndarray backends)

Basic steps:

```bash
cd needle-ssm

# (optional but recommended) create a venv
python3 -m venv venv
source venv/bin/activate

# install python deps
pip install --upgrade pip
pip install -r requirements.txt

# optional: S4Torch reference (if you need torch-based comparisons)
# (requires torch already installed; disable build isolation so pip is available)
pip install --no-build-isolation git+https://github.com/TariqAHassan/S4Torch.git

# build the C++ / CUDA backends
make clean
make lib
```

To verify that Needle imports correctly:

```bash
python3 - << 'PY'
import sys
sys.path.insert(0, "python")
import needle as ndl
print("needle imported ok")
print("available devices:", ndl.all_devices())
PY
```

If that runs, you’re good to use the S4 code and tests.

---

## 2. How to run the S4 / SSM tests

All S4 tests are under `tests/hw4_extra/test_ssm.py`.

From the repo root:

```bash
cd needle-ssm
export PYTHONPATH=$PWD/python:$PYTHONPATH
python3 -m pytest -l -v tests/hw4_extra/test_ssm.py
```

Notes:
- On the **first run**, the tests will create two reference `.npy` files under `tests/hw4_extra/data/`:
  - `test_s4_layer.npy`
  - `test_s4_model.npy`
- Later runs compare current outputs against those reference files.
- Tests cover:
  - HiPPO-style initialization
  - Discretizationin 
  - The core SSM step
  - `S4Layer` forward
  - `S4` model forward

If you only want to run a subset of tests, you can use `-k`:

```bash
python3 -m pytest -l -v tests/hw4_extra/test_ssm.py -k "s4_layer"
```

---

## 3. How to build and inspect an S4 model

You can create S4 models directly from `needle.nn`, or use presets in `apps/models.py`.

### 3.1 Simple S4 example

```bash
cd needle-ssm
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

```python
import needle as ndl
import needle.nn as nn
import numpy as np

batch, seq_len, dim = 2, 4, 16

x_np = np.random.randn(batch, seq_len, dim).astype(np.float32)
x = ndl.Tensor(x_np, device=ndl.cpu())

model = nn.S4(
    embedding_size=dim,
    hidden_size=32,
    num_layers=2,
    state_size=8,
    dropout=0.0,
    device=ndl.cpu(),
    batch_first=True,
    sequence_len=seq_len,
)

out, h = model(x)
print("input shape: ", x.shape)
print("output shape:", out.shape)
```

Expected:
- Input shape: `(batch, seq_len, dim)`
- Output shape: `(batch, seq_len, dim)` (same shape, processed through S4 layers)

### 3.2 Using S4 presets from `apps/models.py`

`apps/models.py` defines:
- `count_parameters(model)` → number of trainable parameters
- `create_s4_model(size=..., device=..., dtype=..., sequence_len=...)`

Example:

```bash
cd needle-ssm
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

```python
import needle as ndl
from apps.models import create_s4_model, count_parameters
import numpy as np

model = create_s4_model("small", device=ndl.cpu())
print("params:", count_parameters(model))

batch, seq_len, dim = 2, 128, 512
x = ndl.Tensor(np.random.randn(batch, seq_len, dim).astype(np.float32), device=ndl.cpu())
out, _ = model(x)
print("output shape:", out.shape)
```

Available sizes (approximate parameter scales):
- `"small"`  → ~10M
- `"medium"` → ~30M
- `"large"`  → ~50M
- `"xlarge"` → ~100M

---

## 4. S4 / S4Layer architecture (simplified flow)

At a high level, the S4 model does:

```text
Input x (batch, seq_len, dim)
   │
   │ add learned time / position embeddings
   ▼
Positional + content representation
   │
   ▼
Stack of S4Layer blocks (num_layers times)
   │
   ▼
Processed sequence (same shape as input)
```

Inside one `S4Layer` block:

```text
Input x (B, L, D)
   │
   ├─ save as residual
   │
   ├─ LayerNorm over last dim
   │
   ├─ Linear: D → 2D
   │
   ├─ Split into (u, gate), each (B, L, D)
   │
   ├─ SSM core runs over time on u  → ssm_out (B, L, D)
   │
   ├─ GLU: ssm_out * sigmoid(gate)
   │
   ├─ Dropout + residual add
   │
   ├─ Feed-forward (D → H → D) + residual add
   │
   ▼
Output x (B, L, D)
```

Inside the SSM core (`run_ssm`), for each channel:
- The state is a small vector of length `state_size`.
- At every time step, the state is updated with simple elementwise formulas derived from a diagonal state matrix (`Lambda`) and step size (`log_step`).
- A readout matrix `C` turns the state into an output per channel.

This gives you a sequence model that:
- Looks like a Transformer block from the outside (batch, seq_len, dim in / out),
- But uses a state space model internally instead of self-attention.

---

## 5. Notes

- CUDA is optional. If CUDA is not available, the CPU backend still works.
- If you change Python versions, you may need to rebuild the `.so` files:

  ```bash
  make clean
  make lib
  ```


