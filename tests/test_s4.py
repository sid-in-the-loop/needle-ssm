import pytest
import numpy as np
from needle.autograd import Tensor
from needle.nn.nn_ssm import hippo_legs_init, S4Layer, S4
import needle as ndl

# Test hippo_legs_init
def test_hippo_legs_init():
    state_size = 4
    diag, b = hippo_legs_init(state_size, device=ndl.cpu())

    # Check shapes
    assert diag.shape == (state_size,)
    assert b.shape == (state_size,)

    # Check values
    expected_diag = np.array([-(i + 1) for i in range(state_size)], dtype=np.float32)
    expected_b = np.array([np.sqrt(2 * i + 1) for i in range(state_size)], dtype=np.float32)

    np.testing.assert_allclose(diag.numpy(), expected_diag, rtol=1e-5)
    np.testing.assert_allclose(b.numpy(), expected_b, rtol=1e-5)

# Test S4Layer discretize
def test_s4layer_discretize():
    state_size = 4
    layer = S4Layer(q_features=8, hidden_size=16, state_size=state_size, device=ndl.cpu())

    a_bar, b_bar = layer.discretize()

    # Check shapes
    assert a_bar.shape == (1, 1, state_size)
    assert b_bar.shape == (1, 1, state_size)

# Test S4Layer run_ssm
def test_s4layer_run_ssm():
    state_size = 4
    seq_len = 10
    batch_size = 2
    channels = 8

    layer = S4Layer(q_features=channels, hidden_size=16, state_size=state_size, device=ndl.cpu())

    u = ndl.Tensor(np.random.randn(batch_size, seq_len, channels).astype(np.float32), device=ndl.cpu())
    output = layer.run_ssm(u)

    # Check output shape
    assert output.shape == (batch_size, seq_len, channels)

# Test S4 forward pass
def test_s4_forward():
    batch_size = 2
    seq_len = 10
    embedding_size = 8
    hidden_size = 16
    num_layers = 2

    model = S4(embedding_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, device=ndl.cpu())

    x = ndl.Tensor(np.random.randn(batch_size, seq_len, embedding_size).astype(np.float32), device=ndl.cpu())
    output, _ = model(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, embedding_size)

if __name__ == "__main__":
    pytest.main()