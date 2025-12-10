"""
State Space Model (S4) implementation for sequence modeling.

This module implements the Structured State Space Sequence (S4) model, which is
a powerful architecture for long-range sequence modeling. S4 models continuous-time
state space systems and discretizes them for efficient parallel computation.

Key components:
- S4Layer: Single layer implementing state space dynamics with gating and FFN
- S4: Full model stacking multiple S4 layers with positional embeddings
- HiPPO initialization: Specialized initialization for capturing long dependencies
- Causal convolution: Efficient parallel computation of recurrent dynamics

References:
- "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al.)
"""

from typing import Optional
import math

from needle import backend_ndarray as nd

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
    Parallel causal convolution using custom backend kernel.
    
    Computes causal convolution where output at time t only depends on inputs
    at times <= t. This is essential for autoregressive sequence modeling.
    
    Args:
        u: Input tensor of shape (batch, seq_len, channels).
        kernel: Convolution kernel of shape (seq_len, channels).
    
    Returns:
        Causal convolution result of shape (batch, seq_len, channels).
    """
    batch, seq_len, channels = u.shape
    return ops.causal_conv_backend(u, kernel, batch, seq_len, channels)


def hippo_legs_init(
    state_size: int,
    *,
    device=None,
    dtype: str = "float32",
) -> tuple[Tensor, Tensor]:
    """
    Initialize state space parameters using HiPPO-LegS (Legendre) basis.
    
    HiPPO (High-order Polynomial Projection Operators) provides a principled
    way to initialize state space models for capturing long-range dependencies.
    The LegS variant uses Legendre polynomials as the orthogonal basis.
    
    The initialization provides:
    - Lambda (diagonal eigenvalues): Controls state dynamics decay rates
    - B vector: Input-to-state projection weights
    
    These parameters define the continuous-time SSM dynamics:
        dx/dt = Λx(t) + Bu(t)
    
    Args:
        state_size: Dimension of the latent state space (typically 64 or 256).
        device: Device to place tensors on (CPU/CUDA).
        dtype: Data type for tensors.
    
    Returns:
        Tuple of (Lambda, B) tensors:
        - Lambda: Diagonal eigenvalues of shape (state_size,) with negative values
        - B: Input projection vector of shape (state_size,)
    
    References:
        "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (Gu et al.)
    """
    # HiPPO-LegS initialization: eigenvalues are -(i+1) for i in [0, state_size)
    # These negative eigenvalues ensure stable dynamics (states decay over time)
    diag = nd.NDArray([-(i + 1) for i in range(state_size)], device=device)
    
    # B vector scales with sqrt(2i+1) to maintain proper normalization
    # with Legendre polynomial basis functions
    b = nd.NDArray([math.sqrt(2 * i + 1) for i in range(state_size)], device=device)
    
    # Return as trainable parameters - the model will learn adjustments
    # to these theoretically-motivated initializations
    return (
        Tensor(diag, device=device, dtype=dtype, requires_grad=True),
        Tensor(b, device=device, dtype=dtype, requires_grad=True),
    )


class S4Layer(Module):
    """
    Single layer of Structured State Space Sequence (S4) model.
    
    This layer implements the core S4 computation which combines:
    1. State space dynamics: Efficient modeling of long sequences via SSM
    2. Gated Linear Unit (GLU): Multiplicative gating for controlled information flow
    3. Feed-forward network: Additional expressiveness for complex patterns
    4. Residual connections: Gradient flow and training stability
    
    The state space model computes:
        x_{t+1} = Ā x_t + B̄ u_t    (discrete-time state update)
        y_t = C x_t                  (output projection)
    
    This is efficiently computed via parallel causal convolution in the frequency
    domain, avoiding sequential recurrence during training.
    
    Architecture flow:
        input → LayerNorm → Linear projection (2x) → [SSM path, Gate path]
        SSM output * sigmoid(gate) → Dropout → + residual
        → FFN (Linear → ReLU → Dropout → Linear → Dropout) → + residual
    
    Args:
        q_features: Input/output feature dimension (embedding size).
        hidden_size: Hidden dimension for feed-forward network.
        state_size: Latent state dimension for SSM (controls memory capacity).
        dropout: Dropout probability for regularization.
        use_fft: If True, use FFT-based convolution (faster for long sequences).
        device: Computation device (CPU/CUDA).
        dtype: Data type for parameters.
    """
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

        # Store configuration
        self.q_features = q_features
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.device = device
        self.dtype = dtype
        self.use_fft = use_fft

        # Initialize state space parameters using HiPPO
        # Lambda: diagonal state transition matrix (eigenvalues)
        # B: input-to-state projection
        lambda_init, b_init = hippo_legs_init(
            state_size,
            device=device,
            dtype=dtype,
        )

        # SSM parameters (learnable)
        self.Lambda = Parameter(lambda_init)  # State dynamics: (state_size,)
        self.B = Parameter(b_init)             # Input projection: (state_size,)
        self.C = Parameter(                    # Output projection: (q_features, state_size)
            init.randn(
                q_features,
                state_size,
                device=device,
                dtype=dtype,
            )
        )
        # Log step size (Δ) - controls discretization timescale
        # Learned in log-space for numerical stability and positive constraint
        self.log_step = Parameter(
            init.zeros(
                1,
                device=device,
                dtype=dtype,
            )
        )

        # Pre-SSM layers: normalization and projection
        self.layernorm = LayerNorm1d(q_features, device=device, dtype=dtype)
        # Project to 2x features: one for SSM input, one for gating
        self.in_proj = Linear(q_features, 2 * q_features, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)

        # Post-SSM feed-forward network for additional expressiveness
        self.ffn_linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.ffn_dropout = Dropout(dropout)
        self.ffn_linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.ffn_dropout2 = Dropout(dropout)

    def discretize(self):
        """
        Convert continuous-time SSM to discrete-time using zero-order hold (ZOH).
        
        Continuous-time dynamics: dx/dt = Λx(t) + Bu(t)
        Discrete-time dynamics: x_{k+1} = Ā x_k + B̄ u_k
        
        Where:
            Ā = exp(ΛΔ)                    (matrix exponential)
            B̄ = (exp(ΛΔ) - I) Λ^{-1} B    (exact discretization)
        
        For diagonal Λ, matrix operations become element-wise.
        Element-wise division is correct for diagonal Lambda (not matrix inverse).
        
        Returns:
            Tuple of (Ā, B̄) tensors, shape (1, 1, state_size) for broadcasting.
        """
        # Compute step size Δ = exp(log_step) and broadcast to Lambda shape
        delta = ops.broadcast_to(ops.exp(self.log_step), self.Lambda.shape)
        
        # Compute ΛΔ element-wise
        lambda_delta = self.Lambda * delta
        
        # Discretized state transition: Ā = exp(ΛΔ)
        a_bar = ops.exp(lambda_delta)
        
        # Discretized input matrix: B̄ = (exp(ΛΔ) - I) / Λ * B
        ones = init.ones_like(self.Lambda, requires_grad=False)
        b_bar = ops.divide(a_bar - ones, self.Lambda) * self.B
        
        # Reshape for broadcasting in subsequent operations
        return (
            a_bar.reshape((1, 1, self.state_size)),
            b_bar.reshape((1, 1, self.state_size)),
        )

    def run_ssm(self, u: Tensor) -> Tensor:
        """
        Run state-space model using parallel convolution (S4's key insight).
        
        Instead of sequential recurrence (slow), S4 computes a convolution kernel
        that captures the full SSM dynamics, then applies it via efficient
        causal convolution (can be parallelized and use FFT).
        
        The convolution kernel at lag k is: K_k = C * Ā^k * B̄
        where Ā^k represents k applications of the state transition.
        
        Args:
            u: Input sequence of shape (batch, seq_len, channels).
        
        Returns:
            SSM output of shape (batch, seq_len, channels).
        """
        _, seq_len, channels = u.shape
        
        # Get discretized SSM parameters
        a_bar, b_bar = self.discretize()

        # Flatten for power computation
        a_bar_flat = a_bar.reshape((self.state_size,))
        b_bar_flat = b_bar.reshape((self.state_size,))
        c_matrix = self.C

        # Compute powers of Ā: [Ā^0, Ā^1, Ā^2, ..., Ā^(seq_len-1)]
        # For diagonal matrices, powers are element-wise
        ones = init.ones(
            self.state_size,
            device=u.device,
            dtype=u.dtype,
            requires_grad=False,
        )
        # Build list of powers: a_powers[k] = Ā^k
        a_powers = [ones]
        current = ones
        for _ in range(1, seq_len):
            current = current * a_bar_flat  # Element-wise multiplication for diagonal
            a_powers.append(current)
        # Stack into tensor of shape (seq_len, state_size)
        a_powers_tensor = ops.stack(tuple(a_powers), axis=0)

        # Broadcast all components to (seq_len, channels, state_size)
        # This prepares for computing K_k = C * Ā^k * B̄ for all k and channels
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

        # Compute convolution kernel: K = C * Ā^k * B̄ for each lag k
        # Sum over state dimension to get kernel of shape (seq_len, channels)
        kernel = ops.summation(
            c_expanded * a_powers_expanded * b_expanded,
            axes=2,
        )

        # Apply convolution kernel to input
        if self.use_fft:
            # FFT-based causal convolution (faster for long sequences)
            return ops.causal_conv1d_fft(u, kernel)
        # Custom backend causal convolution
        return causal_conv(u, kernel)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of S4 layer.
        
        Architecture:
            1. Residual 1: LayerNorm → Linear(2x) → Split → [SSM, Gate]
               → SSM output * sigmoid(gate) → Dropout → + input
            2. Residual 2: FFN (Linear → ReLU → Dropout → Linear → Dropout)
               → + previous output
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim).
        
        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        batch, seq_len, dim = x.shape

        # Save input for first residual connection
        residual = x

        # ==== Pre-processing: LayerNorm ====
        # Flatten batch and sequence for layer norm (operates on last dim)
        y = x.reshape((batch * seq_len, dim))
        y = self.layernorm(y)
        y = y.reshape((batch, seq_len, dim))

        # ==== Projection and Gating ====
        # Project to 2x dimension, then split for SSM input and gate
        proj = self.in_proj(y.reshape((batch * seq_len, dim)))
        proj = proj.reshape((batch, seq_len, 2, self.q_features))
        u, gate = ops.split(proj, 2)  # Split along axis=2
        u = u.reshape((batch, seq_len, self.q_features))
        gate = gate.reshape((batch, seq_len, self.q_features))

        # ==== SSM Computation ====
        ssm_out = self.run_ssm(u)
        
        # ==== Gated Linear Unit (GLU) ====
        # Multiply SSM output by sigmoid-gated values for adaptive filtering
        glu = ssm_out * ops.sigmoid(gate)

        # ==== First Residual Connection ====
        x = residual + self.dropout(glu)

        # ==== Feed-Forward Network (FFN) ====
        # Two-layer MLP with ReLU activation for additional expressiveness
        ff = x.reshape((batch * seq_len, self.q_features))
        ff = self.ffn_linear1(ff)
        ff = self.relu(ff)
        ff = self.ffn_dropout(ff)
        ff = self.ffn_linear2(ff)
        ff = self.ffn_dropout2(ff)
        ff = ff.reshape((batch, seq_len, self.q_features))

        # ==== Second Residual Connection ====
        x = x + ff
        return x


class S4(Module):
    """
    Full S4 model: stacks multiple S4 layers with positional embeddings.
    
    This is the complete sequence model that:
    1. Adds learned positional embeddings to input
    2. Processes through multiple S4 layers
    3. Returns final hidden states
    
    Suitable for tasks like:
    - Language modeling (next-token prediction)
    - Sequence classification
    - Time-series forecasting
    - Any task requiring long-range dependency modeling
    
    Args:
        embedding_size: Dimension of input embeddings and model hidden states.
        hidden_size: Hidden dimension for FFN in each S4 layer.
        num_layers: Number of stacked S4 layers.
        state_size: Latent state dimension for SSM (memory capacity).
        dropout: Dropout probability for regularization.
        use_fft: If True, use FFT-based convolution (faster for long sequences).
        device: Computation device (CPU/CUDA).
        dtype: Data type for parameters.
        batch_first: If True, input shape is (batch, seq, features).
        sequence_len: Maximum sequence length for positional embeddings.
    """
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

        # Store configuration
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.sequence_len = sequence_len
        self.device = device
        self.dtype = dtype
        self.use_fft = use_fft

        # Learned positional embeddings for temporal information
        # Maps position indices [0, sequence_len) to embedding vectors
        self.embedding = Embedding(
            sequence_len,
            embedding_size,
            device=device,
            dtype=dtype,
        )

        # Stack multiple S4 layers sequentially
        # Each layer processes the output of the previous layer
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
        """
        Forward pass of full S4 model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) if batch_first=True,
               or (seq_len, batch, features) if batch_first=False.
            h: Optional hidden state (unused, kept for compatibility with RNN interface).
        
        Returns:
            Tuple of (output, dummy_state):
            - output: Processed tensor of same shape as input
            - dummy_state: Zeros tensor (for RNN interface compatibility)
        """
        # Transpose if sequence-first format is used
        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        batch, seq_len, _ = x.shape
        
        # ==== Add Positional Embeddings ====
        # Create position indices [0, 1, 2, ..., seq_len-1] for each batch
        timestamps = Tensor(
            [i for i in range(seq_len)],
            device=x.device,
            dtype=x.dtype,
            requires_grad=False,
        )
        timestamps = ops.broadcast_to(timestamps, (batch, seq_len))
        # Look up positional embeddings and add to input
        time_emb = self.embedding(timestamps)
        x = x + time_emb

        # ==== Process through S4 Layers ====
        x = self.layers(x)

        # Transpose back if sequence-first format was used
        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        # Return output and dummy hidden state (for RNN interface compatibility)
        return x, init.zeros_like(x)

