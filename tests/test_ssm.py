import sys
sys.path.append("./python")
sys.path.append("./apps")

import numpy as np
import pytest
from pathlib import Path

import needle as ndl
import needle.nn as nn
from needle.nn.nn_ssm import hippo_legs_init

import torch
from s4torch import S4Model as TorchS4Model
from s4torch.layer import S4Layer as TorchS4Layer


_DATA_DIR = "./tests/hw4_extra/data"

_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(),
        marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]


def _label_path(name: str) -> str:
    return f"{_DATA_DIR}/{name}"


def test_s4_layer_forward():
    np.random.seed(12345)
    batch_size, seq_len, dim = 2, 6, 16
    hidden_size = 32
    state_size = 8

    x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    layer = nn.S4Layer(
        q_features=dim,
        hidden_size=hidden_size,
        state_size=state_size,
        dropout=0.0,
        device=ndl.cpu(),
    )

    result = layer(ndl.Tensor(x, device=ndl.cpu()))
    np_result = result.numpy()

    labels_path = _label_path("test_s4_layer.npy")
    if not Path(labels_path).exists():
        # Generate reference file on first run
        Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, np_result)
        print(f"Generated reference file: {labels_path}")
        return
    
    with open(labels_path, "rb") as f:
        label = np.load(f)

    np.testing.assert_allclose(np_result, label, atol=1e-5, rtol=1e-5)


def test_s4_model_forward():
    np.random.seed(12345)
    batch_size, seq_len, dim = 2, 6, 16
    hidden_size = 32
    num_layers = 2
    state_size = 8

    x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    model = nn.S4(
        embedding_size=dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        state_size=state_size,
        dropout=0.0,
        device=ndl.cpu(),
        batch_first=True,
        sequence_len=seq_len,
    )

    result, _ = model(ndl.Tensor(x, device=ndl.cpu()))
    np_result = result.numpy()

    labels_path = _label_path("test_s4_model.npy")
    if not Path(labels_path).exists():
        # Generate reference file on first run
        Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, np_result)
        print(f"Generated reference file: {labels_path}")
        return
    
    with open(labels_path, "rb") as f:
        label = np.load(f)

    np.testing.assert_allclose(np_result, label, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("state_size", [4, 8, 16])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_hippo_legs_init(state_size, device):
    """Test HiPPO-LegS initialization produces correct diagonal and B vector."""
    np.random.seed(42)
    
    lambda_init, b_init = hippo_legs_init(state_size, device=device, dtype="float32")
    
    # Check shapes
    assert lambda_init.shape == (state_size,)
    assert b_init.shape == (state_size,)
    
    # Check diagonal values: A[i,i] = -(i+1)
    lambda_np = lambda_init.numpy()
    expected_diag = np.array([-(i + 1) for i in range(state_size)], dtype=np.float32)
    np.testing.assert_allclose(lambda_np, expected_diag, atol=1e-5, rtol=1e-5)
    
    # Check B vector: B[i] = sqrt(2*i + 1)
    b_np = b_init.numpy()
    expected_b = np.array([np.sqrt(2 * i + 1) for i in range(state_size)], dtype=np.float32)
    np.testing.assert_allclose(b_np, expected_b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [5, 11])
@pytest.mark.parametrize("dim", [16, 32])
@pytest.mark.parametrize("hidden_size", [32, 64])
@pytest.mark.parametrize("state_size", [8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_s4_layer_variants(
    batch_size, seq_len, dim, hidden_size, state_size, dropout, device
):
    """Test S4Layer with various configurations."""
    np.random.seed(19943)
    
    x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    layer = nn.S4Layer(
        q_features=dim,
        hidden_size=hidden_size,
        state_size=state_size,
        dropout=dropout,
        device=device,
    )
    
    result = layer(ndl.Tensor(x, device=device))
    np_result = result.numpy()
    
    # Check output shape
    assert np_result.shape == (batch_size, seq_len, dim)
    
    # Check output is finite
    assert np.all(np.isfinite(np_result))


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [5, 11])
@pytest.mark.parametrize("dim", [16, 32])
@pytest.mark.parametrize("hidden_size", [32, 64])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("state_size", [8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_s4_model_variants(
    batch_size, seq_len, dim, hidden_size, num_layers, state_size, dropout, device
):
    """Test S4 model with various configurations."""
    np.random.seed(19943)
    
    x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    model = nn.S4(
        embedding_size=dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        state_size=state_size,
        dropout=dropout,
        device=device,
        batch_first=True,
        sequence_len=seq_len,
    )
    
    result, h = model(ndl.Tensor(x, device=device))
    np_result = result.numpy()
    
    # Check output shape
    assert np_result.shape == (batch_size, seq_len, dim)
    
    # Check hidden state is zeros (as per Transformer convention)
    assert h.shape == np_result.shape
    np.testing.assert_allclose(h.numpy(), np.zeros_like(np_result), atol=1e-5)
    
    # Check output is finite
    assert np.all(np.isfinite(np_result))


def test_s4_discretization():
    """Test that discretization produces valid outputs."""
    np.random.seed(42)
    
    layer = nn.S4Layer(
        q_features=16,
        hidden_size=32,
        state_size=8,
        dropout=0.0,
        device=ndl.cpu(),
    )
    
    # Test discretize method
    a_bar, b_bar = layer.discretize()
    
    # Check shapes
    assert a_bar.shape == (1, 1, 8)
    assert b_bar.shape == (1, 1, 8)
    
    # Check values are finite
    assert np.all(np.isfinite(a_bar.numpy()))
    assert np.all(np.isfinite(b_bar.numpy()))
    
    # Check delta is positive (exp(log_step) > 0)
    delta = ndl.exp(layer.log_step).numpy()
    assert delta > 0


def test_s4_ssm_forward():
    """Test SSM forward computation produces correct output shape."""
    np.random.seed(42)
    
    batch_size, seq_len, dim = 2, 6, 16
    state_size = 8
    
    layer = nn.S4Layer(
        q_features=dim,
        hidden_size=32,
        state_size=state_size,
        dropout=0.0,
        device=ndl.cpu(),
    )
    
    u = ndl.Tensor(
        np.random.randn(batch_size, seq_len, dim).astype(np.float32),
        device=ndl.cpu(),
    )
    
    ssm_out = layer.run_ssm(u)
    np_out = ssm_out.numpy()
    
    # Check output shape matches input
    assert np_out.shape == (batch_size, seq_len, dim)
    
    # Check output is finite
    assert np.all(np.isfinite(np_out))



# -------------------------
# Black-box comparison with S4Torch
# -------------------------
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_s4_layer_blackbox(device):
    """Black-box test: ensure Needle S4Layer runs and output shape matches Torch S4Layer."""
    np.random.seed(42)
    batch_size, seq_len, dim = 2, 6, 16
    state_size = 8

    x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

    # Torch reference
    torch_layer = TorchS4Layer(d_model=dim, n=state_size, l_max=seq_len)
    torch_layer.eval()
    with torch.no_grad():
        torch_out = torch_layer(torch.tensor(x))
    
    # Needle S4Layer
    needle_layer = nn.S4Layer(
        q_features=dim,
        hidden_size=dim,
        state_size=state_size,
        dropout=0.0,
        device=device
    )
    needle_out = needle_layer(ndl.Tensor(x, device=device))

    # Check shape and finiteness
    assert needle_out.shape == torch_out.shape
    assert np.all(np.isfinite(needle_out.numpy()))
    assert np.all(np.isfinite(torch_out.numpy()))

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_s4_model_blackbox(device):
    """Black-box test: ensure Needle S4 model runs and output shape matches Torch S4Model."""
    np.random.seed(42)
    batch_size, seq_len, dim = 2, 6, 16
    state_size = 8
    num_layers = 2

    x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)

    # Torch reference
    torch_model = TorchS4Model(
        d_input=dim,
        d_model=dim,
        d_output=dim,
        n_blocks=num_layers,
        n=state_size,
        l_max=seq_len
    )
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(torch.tensor(x))

    # Needle model
    needle_model = nn.S4(
        embedding_size=dim,
        hidden_size=dim,
        num_layers=num_layers,
        state_size=state_size,
        dropout=0.0,
        device=device,
        batch_first=True,
        sequence_len=seq_len,
    )
    needle_out, _ = needle_model(ndl.Tensor(x, device=device))

    # Check shape and finiteness
    assert needle_out.shape == torch_out.shape
    assert np.all(np.isfinite(needle_out.numpy()))
    assert np.all(np.isfinite(torch_out.numpy()))
