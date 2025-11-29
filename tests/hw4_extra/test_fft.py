import numpy as np
import time
import pytest
import needle as ndl
import torch

# Timing parameters (subset to keep test quick)
conv_timing_params = [
    (128, 16, 32, 7, 1),
    (128, 16, 32, 11, 1),
    (128, 16, 32, 15, 1),
    (128, 16, 32, 19, 1),
    (128, 16, 32, 31, 1),
    (128, 16, 32, 63, 1),
]

DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


@pytest.mark.parametrize("s,cin,cout,k,stride", conv_timing_params)
@pytest.mark.parametrize("device", DEVICES)  # CPU only to reduce variability
def test_nn_conv_timing(s, cin, cout, k, stride, device):
    np.random.seed(0)
    fast = ndl.nn.FastConv2d(cin, cout, k, stride=stride, device=device)
    np.random.seed(0)  # Reset seed to ensure same weights
    slow = ndl.nn.Conv(cin, cout, k, stride=stride, device=device)

    x = ndl.init.rand(8, cin, s, s, device=device)

    _ = fast(x); _ = slow(x)

    def time_call(fn, inp):
        t0 = time.perf_counter(); out = fn(inp); return out, time.perf_counter() - t0

    out_fast, t_fast = time_call(fast, x)
    out_slow, t_slow = time_call(slow, x)

    # Torch reference for correctness check
    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(fast.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(fast.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy())
    out_torch = g(z)

    err_fast = np.linalg.norm(out_fast.cached_data.numpy() - out_torch.detach().numpy())
    err_slow = np.linalg.norm(out_slow.cached_data.numpy() - out_torch.detach().numpy())
    assert err_fast < 5e-3
    assert err_slow < 5e-3

    print(f"Conv timing (s={s}, cin={cin}, cout={cout}, k={k}, stride={stride}): slow={t_slow*1e3:.2f}ms fast={t_fast*1e3:.2f}ms speedup={t_slow/t_fast if t_fast>0 else float('inf'):.2f}x")

    # assert t_fast < t_slow * 1.2, "FastConv2d should be at least near or faster than baseline"
