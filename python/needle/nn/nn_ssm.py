from typing import Optional

import numpy as np

from needle.autograd import Tensor
from needle import ops
import needle.init as init
from .nn_basic import (
    Module,
    Parameter,
    LayerNorm1d,
    Linear,
    Dropout,
    ReLU,
    Sequential,
)
from .nn_sequence import Embedding


def causal_conv(u: Tensor, kernel: Tensor) -> Tensor:
    """
    Parallel causal convolution backed by the custom kernel.
    """
    batch, seq_len, channels = u.shape
    return ops.causal_conv_backend(u, kernel, batch, seq_len, channels)


def hippo_legs_init(
    state_size: int,
    *,
    device=None,
    dtype: str = "float32",
) -> tuple[Tensor, Tensor]:
    """Return initial diagonal (eigenvalues) and B vector for HiPPO-LegS."""
    diag = np.array([-(i + 1) for i in range(state_size)], dtype=np.float32)
    b = np.array(
        [np.sqrt(2 * i + 1) for i in range(state_size)],
        dtype=np.float32,
    )
    return (
        Tensor(diag, device=device, dtype=dtype, requires_grad=True),
        Tensor(b, device=device, dtype=dtype, requires_grad=True),
    )


class S4Layer(Module):
    def __init__(
        self,
        q_features: int,
        hidden_size: int,
        state_size: int = 64,
        *,
        dropout: float = 0.0,
        use_fft: bool = False,
        device=None,
        dtype: str = "float32",
    ):
        super().__init__()

        self.q_features = q_features
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.device = device
        self.dtype = dtype
        self.use_fft = use_fft

        lambda_init, b_init = hippo_legs_init(
            state_size,
            device=device,
            dtype=dtype,
        )

        self.Lambda = Parameter(lambda_init)
        self.B = Parameter(b_init)
        self.C = Parameter(
            init.randn(
                q_features,
                state_size,
                device=device,
                dtype=dtype,
            )
        )
        self.log_step = Parameter(
            init.zeros(
                1,
                device=device,
                dtype=dtype,
            )
        )

        self.layernorm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.in_proj = Linear(q_features, 2 * q_features, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)

        self.ffn_linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.ffn_dropout = Dropout(dropout)
        self.ffn_linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.ffn_dropout2 = Dropout(dropout)

    def discretize(self):
        """
        Converts continuous-time SSM to discrete-time.
        For diagonal SSM: Ā = e^(ΛΔ), B̄ = (e^(ΛΔ) - I) / Λ * B
        Element-wise division is correct for diagonal Lambda (not matrix inverse).
        """
        delta = ops.broadcast_to(ops.exp(self.log_step), self.Lambda.shape)
        lambda_delta = self.Lambda * delta
        a_bar = ops.exp(lambda_delta)
        ones = init.ones_like(self.Lambda, requires_grad=False)
        b_bar = ops.divide(a_bar - ones, self.Lambda) * self.B
        return (
            a_bar.reshape((1, 1, self.state_size)),
            b_bar.reshape((1, 1, self.state_size)),
        )

    def run_ssm(self, u: Tensor) -> Tensor:
        """
        Runs the state-space model using the parallel convolution kernel.
        """
        _, seq_len, channels = u.shape
        a_bar, b_bar = self.discretize()

        a_bar_flat = a_bar.reshape((self.state_size,))
        b_bar_flat = b_bar.reshape((self.state_size,))
        c_matrix = self.C

        ones = init.ones(
            self.state_size,
            device=u.device,
            dtype=u.dtype,
            requires_grad=False,
        )
        a_powers = [ones]
        current = ones
        for _ in range(1, seq_len):
            current = current * a_bar_flat
            a_powers.append(current)
        a_powers_tensor = ops.stack(tuple(a_powers), axis=0)

        a_powers_expanded = ops.broadcast_to(
            a_powers_tensor.reshape((seq_len, 1, self.state_size)),
            (seq_len, channels, self.state_size),
        )
        b_expanded = ops.broadcast_to(
            b_bar_flat.reshape((1, 1, self.state_size)),
            (seq_len, channels, self.state_size),
        )
        c_expanded = ops.broadcast_to(
            c_matrix.reshape((1, channels, self.state_size)),
            (seq_len, channels, self.state_size),
        )

        kernel = ops.summation(
            c_expanded * a_powers_expanded * b_expanded,
            axes=2,
        )

        if self.use_fft:
            # FFT-based causal convolution path (keeps same semantics as backend)
            return ops.causal_conv1d_fft(u, kernel)
        return causal_conv(u, kernel)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape

        residual = x

        y = x.reshape((batch * seq_len, dim))
        y = self.layernorm(y)
        y = y.reshape((batch, seq_len, dim))

        proj = self.in_proj(y.reshape((batch * seq_len, dim)))
        proj = proj.reshape((batch, seq_len, 2, self.q_features))
        u, gate = ops.split(proj, 2)
        u = u.reshape((batch, seq_len, self.q_features))
        gate = gate.reshape((batch, seq_len, self.q_features))

        ssm_out = self.run_ssm(u)
        glu = ssm_out * ops.sigmoid(gate)

        x = residual + self.dropout(glu)

        ff = x.reshape((batch * seq_len, self.q_features))
        ff = self.ffn_linear1(ff)
        ff = self.relu(ff)
        ff = self.ffn_dropout(ff)
        ff = self.ffn_linear2(ff)
        ff = self.ffn_dropout2(ff)
        ff = ff.reshape((batch, seq_len, self.q_features))

        x = x + ff
        return x


class S4(Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        *,
        state_size: int = 64,
        dropout: float = 0.0,
        use_fft: bool = False,
        device=None,
        dtype: str = "float32",
        batch_first: bool = True,
        sequence_len: int = 2048,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.sequence_len = sequence_len
        self.device = device
        self.dtype = dtype
        self.use_fft = use_fft

        self.embedding = Embedding(
            sequence_len,
            embedding_size,
            device=device,
            dtype=dtype,
        )

        self.layers = Sequential(
            *[
                S4Layer(
                    q_features=embedding_size,
                    hidden_size=hidden_size,
                    state_size=state_size,
                    dropout=dropout,
                    use_fft=use_fft,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, h: Optional[Tensor] = None):
        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        batch, seq_len, _ = x.shape
        timestamps = np.arange(seq_len)
        timestamps = np.broadcast_to(timestamps, (batch, seq_len))
        timestamps = Tensor(
            timestamps,
            device=x.device,
            dtype=x.dtype,
            requires_grad=False,
        )
        time_emb = self.embedding(timestamps)
        x = x + time_emb

        x = self.layers(x)

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)

