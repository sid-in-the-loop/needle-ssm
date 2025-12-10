"""
FFT (Fast Fourier Transform) operations for neural network computations.

This module implements FFT-based operations that are differentiable and can be used
in automatic differentiation frameworks. It includes:
- 2D FFT/IFFT for spatial convolutions
- 1D FFT/IFFT for sequence processing
- Complex arithmetic operations
- Padding and cropping utilities
- Causal convolution using FFT
"""

from typing import Optional

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from .ops_tuple import *
from .ops_mathematic import *

import numpy as np
from ..backend_selection import array_api


class FFT(TensorOp):
    """
    Fast Fourier Transform operation for batched 2D arrays with channels.
    
    This operation computes the 2D FFT along spatial dimensions (height and width)
    for each batch and channel independently. The FFT is computed using NumPy's
    rfft2 (real FFT) which is optimized for real-valued inputs.
    
    Input: A real tensor of shape (batch_size, cin, height, width)
           - batch_size: number of samples in batch
           - cin: number of input channels
           - height, width: spatial dimensions
           
    Output: A complex tensor of shape (batch_size, cin, height, width//2+1, 2)
            - Last dimension stores [real, imaginary] components
            - Width dimension is reduced due to rfft2's output (exploits conjugate symmetry)
    
    Args:
        shape: Optional output shape for FFT. If None, uses input shape.
        device: Device to place the output tensor on.
        dtype: Data type for the output tensor.
    """
    def __init__(self, shape: Optional[tuple] = None, device=None, dtype="float32"):
        """
        Initialize 2D FFT operation on spatial dimensions (axes 2 and 3).
        
        Args:
            shape: Target shape for FFT output. If provided, input will be zero-padded
                   or truncated to match this shape before FFT.
            device: Computation device (CPU/CUDA).
            dtype: Output data type.
        """
        super().__init__()
        self.shape = shape
        self.device = device
        self.dtype = dtype

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Compute 2D FFT on spatial dimensions.
        
        Args:
            x: Input array of shape (N, C_in, H, W) where
               N = batch size, C_in = input channels, H = height, W = width
        
        Returns:
            Complex array of shape (N, C_in, H, W//2+1, 2) where last dim is [real, imag]
        """
        # Apply 2D real FFT along spatial dimensions (axes 2 and 3) for each channel
        # rfft2 is optimized for real inputs and only computes positive frequencies
        fft_result = np.fft.rfft2(x.numpy(), s=self.shape, axes=(2, 3))
        
        # Separate real and imaginary components into separate arrays
        real = fft_result.real.astype(np.float32)
        img = fft_result.imag.astype(np.float32)
        N, C_in, P, Q = real.shape

        # Create output array with extra dimension for complex representation
        # Shape: (N, C_in, P, Q, 2) where [..., 0] is real and [..., 1] is imaginary
        result = array_api.empty((N, C_in, P, Q, 2), device=self.device)
        result[:, :, :, :, 0] = array_api.array(real, device=self.device)
        result[:, :, :, :, 1] = array_api.array(img, device=self.device)

        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of FFT is IFFT.
        
        The derivative of the Discrete Fourier Transform with respect to its input
        is the inverse DFT matrix. This follows from the orthogonality of the DFT.
        
        Args:
            out_grad: Gradient flowing back from subsequent operations.
            node: The node in the computation graph.
        
        Returns:
            Gradient with respect to input, computed via IFFT.
        """
        return ifft(out_grad, device=self.device, dtype=self.dtype)

def fft(x: Tensor, shape=None, device=None, dtype="float32") -> Tensor:
    """
    Convenience function to apply 2D FFT to a tensor.
    
    Args:
        x: Input tensor of shape (batch, channels, height, width).
        shape: Optional output shape for FFT.
        device: Device for computation.
        dtype: Data type for output.
    
    Returns:
        FFT-transformed tensor with complex values in last dimension.
    """
    return FFT(shape=shape, device=device, dtype=dtype)(x)


class IFFT(TensorOp):
    """
    Inverse Fast Fourier Transform operation for batched 2D arrays with channels.
    
    This operation computes the inverse 2D FFT to transform frequency-domain
    representations back to spatial domain. It expects complex input with real
    and imaginary parts stored in the last dimension.
    
    Input: A complex tensor of shape (batch_size, cout, height, width//2+1, 2)
           - Last dimension contains [real, imaginary] components
           - Width is reduced due to rfft2's conjugate symmetry
           
    Output: A real tensor of shape (batch_size, cout, height, width)
            - Reconstructed spatial domain representation
            - Imaginary parts should be ~0 for real-valued inputs
    
    Args:
        shape: Optional output shape for IFFT. Should match original spatial dimensions.
        device: Device to place the output tensor on.
        dtype: Data type for the output tensor.
    """
    def __init__(self, shape: Optional[tuple] = None, device=None, dtype="float32"):
        """
        Initialize 2D IFFT operation on spatial dimensions (axes 2 and 3).
        
        Args:
            shape: Target spatial dimensions (height, width) for output.
            device: Computation device (CPU/CUDA).
            dtype: Output data type.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.shape = shape

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Compute 2D IFFT on frequency-domain representation.
        
        Args:
            x: Complex input array of shape (batch, cout, height, width//2+1, 2)
               where last dimension contains [real, imaginary] components.
        
        Returns:
            Real-valued spatial array of shape (batch, cout, height, width).
        """
        batch_size, cout, height, width, complex = x.shape
        assert complex == 2, "Last dimension must be 2 (real/imag pairs)"
        
        # Extract real and imaginary components from the last dimension
        # compact() ensures data is contiguous in memory for efficient processing
        real = x[:, :, :, :, 0].compact().reshape((batch_size, cout, height, width))
        imag = x[:, :, :, :, 1].compact().reshape((batch_size, cout, height, width))
        
        # Reconstruct complex array for numpy's IFFT
        complex_arr = real.numpy() + 1j * imag.numpy()

        # Apply inverse real FFT along spatial dimensions
        # irfft2 reconstructs full spatial dimensions from reduced frequency representation
        ifft_result = np.fft.irfft2(complex_arr, axes=(2, 3), s=self.shape)
        
        # Take only the real part (imaginary part should be ~0 for real input)
        # Any imaginary component is numerical noise from floating-point arithmetic
        result = ifft_result.real.astype(np.float32)
        result = array_api.array(result, device=self.device)
        
        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of IFFT is FFT.
        
        The derivative of the inverse DFT with respect to its input is the
        forward DFT matrix. This is the adjoint operation to FFT's gradient.
        
        Args:
            out_grad: Gradient flowing back from subsequent operations.
            node: The node in the computation graph.
        
        Returns:
            Gradient with respect to input, computed via FFT.
        """
        return fft(out_grad, device=self.device, dtype=self.dtype)

def ifft(x: Tensor, shape: Optional[tuple] = None, device=None, dtype="float32") -> Tensor:
    """
    Convenience function to apply 2D IFFT to a complex tensor.
    
    Args:
        x: Input complex tensor with shape (..., 2) for [real, imag].
        shape: Optional output spatial dimensions (height, width).
        device: Device for computation.
        dtype: Data type for output.
    
    Returns:
        Real-valued tensor in spatial domain.
    """
    return IFFT(shape=shape, device=device, dtype=dtype)(x)


class FFT1D(TensorOp):
    """
    Real 1D FFT over the sequence axis for time-series or sequential data.
    
    This operation is useful for processing sequences like audio signals or
    time-series data where frequency-domain analysis is beneficial. Uses rfft
    for efficiency with real-valued inputs.

    Input: Real tensor of shape (batch, seq_len, channels)
           - batch: number of sequences
           - seq_len: length of each sequence
           - channels: number of features per timestep
           
    Output: Complex tensor of shape (batch, n_freq, channels, 2)
            - n_freq: number of frequency bins (seq_len//2 + 1 for rfft)
            - Last dimension stores [real, imaginary] components
    
    Args:
        n: Length of FFT. If None, uses input sequence length.
           If n > seq_len, input is zero-padded. If n < seq_len, input is truncated.
        device: Device to place the output tensor on.
        dtype: Data type for the output tensor.
    """

    def __init__(self, n: Optional[int] = None, device=None, dtype: str = "float32"):
        """
        Initialize 1D FFT operation along the sequence axis.
        
        Args:
            n: Target FFT length. Controls frequency resolution.
            device: Computation device (CPU/CUDA).
            dtype: Output data type.
        """
        super().__init__()
        self.n = n
        self.device = device
        self.dtype = dtype

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Compute 1D FFT along sequence axis.
        
        Args:
            x: Input array of shape (batch, seq_len, channels).
        
        Returns:
            Complex array of shape (batch, n_freq, channels, 2) where
            n_freq = n//2 + 1 and last dim is [real, imag].
        """
        batch, seq_len, channels = x.shape
        # Use specified FFT length or default to input sequence length
        n = self.n or seq_len
        
        # Apply 1D real FFT along axis 1 (sequence/time dimension)
        # rfft computes only positive frequencies (exploits conjugate symmetry)
        fft_res = np.fft.rfft(x.numpy(), n=n, axis=1)
        
        # Separate into real and imaginary components
        real = fft_res.real.astype(np.float32)
        imag = fft_res.imag.astype(np.float32)
        
        # Pack into tensor with explicit complex representation
        out = array_api.empty((batch, real.shape[1], channels, 2), device=self.device)
        out[:, :, :, 0] = array_api.array(real, device=self.device)
        out[:, :, :, 1] = array_api.array(imag, device=self.device)
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of 1D FFT is 1D IFFT.
        
        Args:
            out_grad: Gradient from subsequent operations.
            node: Computation graph node.
        
        Returns:
            Gradient w.r.t. input via inverse FFT.
        """
        return ifft1d(out_grad, n=self.n, device=self.device, dtype=self.dtype)


def fft1d(x: Tensor, n: Optional[int] = None, device=None, dtype: str = "float32") -> Tensor:
    """
    Convenience function to apply 1D FFT to a sequential tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, channels).
        n: FFT length (controls frequency resolution).
        device: Device for computation.
        dtype: Data type for output.
    
    Returns:
        FFT-transformed tensor with frequency-domain representation.
    """
    return FFT1D(n=n, device=device, dtype=dtype)(x)


class IFFT1D(TensorOp):
    """
    Real 1D inverse FFT over the sequence axis.
    
    Transforms frequency-domain representations back to time/sequence domain.
    This is the inverse operation of FFT1D, reconstructing the original sequence
    from its frequency components.

    Input: Complex tensor of shape (batch, n_freq, channels, 2)
           - n_freq: number of frequency bins
           - Last dimension contains [real, imaginary] components
           
    Output: Real tensor of shape (batch, seq_len, channels)
            - seq_len: reconstructed sequence length
            - Real-valued time/sequence domain data
    
    Args:
        n: Output sequence length. If None, inferred from input.
           Controls the length of the reconstructed sequence.
        device: Device to place the output tensor on.
        dtype: Data type for the output tensor.
    """

    def __init__(self, n: Optional[int] = None, device=None, dtype: str = "float32"):
        """
        Initialize 1D IFFT operation along the sequence axis.
        
        Args:
            n: Target output sequence length.
            device: Computation device (CPU/CUDA).
            dtype: Output data type.
        """
        super().__init__()
        self.n = n
        self.device = device
        self.dtype = dtype

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Compute 1D IFFT along sequence axis.
        
        Args:
            x: Complex input array of shape (batch, n_freq, channels, 2)
               where last dimension contains [real, imaginary] components.
        
        Returns:
            Real array of shape (batch, seq_len, channels) in time/sequence domain.
        """
        batch, freq, channels, complex_dim = x.shape
        assert complex_dim == 2, "Last dimension must be 2 (real/imag pairs)"
        
        # Extract and convert complex components to contiguous numpy arrays
        real = x[:, :, :, 0].compact().numpy()
        imag = x[:, :, :, 1].compact().numpy()
        
        # Reconstruct complex representation for numpy IFFT
        complex_arr = real + 1j * imag
        
        # Determine output sequence length
        seq_len = self.n
        
        # Apply inverse real FFT to recover time-domain signal
        # irfft automatically handles the conjugate symmetry of rfft output
        ifft_res = np.fft.irfft(complex_arr, n=seq_len, axis=1)
        
        # Convert back to framework's array type
        result = array_api.array(ifft_res.astype(np.float32), device=self.device)
        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of 1D IFFT is 1D FFT.
        
        Args:
            out_grad: Gradient from subsequent operations.
            node: Computation graph node.
        
        Returns:
            Gradient w.r.t. input via forward FFT.
        """
        return fft1d(out_grad, n=self.n, device=self.device, dtype=self.dtype)


def ifft1d(x: Tensor, n: Optional[int] = None, device=None, dtype: str = "float32") -> Tensor:
    """
    Convenience function to apply 1D IFFT to a frequency-domain tensor.
    
    Args:
        x: Input complex tensor of shape (batch, n_freq, channels, 2).
        n: Output sequence length.
        device: Device for computation.
        dtype: Data type for output.
    
    Returns:
        Real-valued tensor in time/sequence domain.
    """
    return IFFT1D(n=n, device=device, dtype=dtype)(x)


class Pad(TensorOp):
    """
    Padding operation for 4D tensors with spatial dimensions.
    
    Adds zeros around the spatial dimensions (height and width) of input tensors.
    This is commonly used to:
    - Preserve spatial dimensions after convolution
    - Prepare data for FFT-based operations (power-of-2 sizes)
    - Implement border handling in image processing
    
    Input: A tensor of shape (batch_size, cin, height, width)
           - batch_size: number of samples
           - cin: number of input channels
           - height, width: spatial dimensions
           
    Output: A tensor of shape (batch_size, cin, height+2*h_padding, width+2*w_padding)
            - Padded with zeros on all sides of spatial dimensions
    
    Args:
        h_padding: Amount of padding to add on each side of height dimension.
        w_padding: Amount of padding to add on each side of width dimension.
    """
    def __init__(self, h_padding: int, w_padding: int):
        """
        Initialize padding operation for spatial dimensions.
        
        Args:
            h_padding: Padding amount for height (applied symmetrically).
            w_padding: Padding amount for width (applied symmetrically).
        """
        self.h_padding = h_padding
        self.w_padding = w_padding
        super().__init__()

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Apply zero-padding to spatial dimensions.
        
        Args:
            x: Input array of shape (batch_size, cin, height, width).
        
        Returns:
            Padded array of shape (batch_size, cin, height+2*h_pad, width+2*w_pad).
        """
        # Define padding specification for each dimension
        # Format: ((before_dim0, after_dim0), (before_dim1, after_dim1), ...)
        # No padding on batch (axis 0) and channel (axis 1) dimensions
        # Symmetric padding on height (axis 2) and width (axis 3)
        pad_width = (
            (0, 0),                              # No padding for batch dimension
            (0, 0),                              # No padding for channel dimension
            (self.h_padding, self.h_padding),    # Symmetric height padding
            (self.w_padding, self.w_padding),    # Symmetric width padding
        )
        padded_x = x.pad(pad_width)
        return padded_x

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of padding is cropping.
        
        When gradients flow back through padding, we remove the padded regions
        to match the original input shape.
        
        Args:
            out_grad: Gradient from subsequent operations (includes padded regions).
            node: Computation graph node.
        
        Returns:
            Gradient cropped to original input size.
        """
        return crop(out_grad, self.h_padding, self.w_padding)

def pad(x: Tensor, h_padding: int, w_padding: int) -> Tensor:
    """
    Convenience function to apply padding to a tensor.
    
    Args:
        x: Input tensor of shape (batch, channels, height, width).
        h_padding: Padding amount for height dimension.
        w_padding: Padding amount for width dimension.
    
    Returns:
        Zero-padded tensor with increased spatial dimensions.
    """
    return Pad(h_padding, w_padding)(x)


class Crop(TensorOp):
    """
    Cropping operation for 4D tensors with spatial dimensions.
    
    Removes pixels from the borders of spatial dimensions. This is the inverse
    operation of padding and is used to:
    - Extract regions of interest from images
    - Remove padding after FFT-based convolution
    - Implement gradient flow through padding operations
    
    Input: A tensor of shape (batch_size, cin, height, width)
           - batch_size: number of samples
           - cin: number of input channels
           - height, width: spatial dimensions before cropping
           
    Output: A tensor of shape (batch_size, cin, crop_height, crop_width)
            - crop_height, crop_width: specified by shape parameter
            - Region extracted starting at (h_cropping, w_cropping)
    
    Args:
        h_cropping: Number of rows to skip from top before extracting region.
        w_cropping: Number of columns to skip from left before extracting region.
        shape: Tuple (height, width) specifying the size of region to extract.
    """
    def __init__(self, h_cropping: int, w_cropping: int, shape: Optional[tuple] = None):
        """
        Initialize cropping operation for spatial dimensions.
        
        Args:
            h_cropping: Starting row index for cropped region.
            w_cropping: Starting column index for cropped region.
            shape: (height, width) of the output region to extract.
        """
        self.h_cropping = h_cropping
        self.w_cropping = w_cropping
        self.shape = shape
        super().__init__()

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Extract a rectangular region from spatial dimensions.
        
        Args:
            x: Input array of shape (batch_size, cin, height, width).
        
        Returns:
            Cropped array of shape (batch_size, cin, crop_h, crop_w) where
            crop_h and crop_w are determined by self.shape.
        """
        # Calculate the boundaries of the region to extract
        # Start at the cropping offsets and extend by the specified shape
        start_h = self.h_cropping
        end_h = self.h_cropping + self.shape[0]
        start_w = self.w_cropping
        end_w = self.w_cropping + self.shape[1]
        
        # Extract the region using array slicing
        # Keep all batches and channels, crop only spatial dimensions
        cropped_x = x[:, :, start_h:end_h, start_w:end_w]
        return cropped_x

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of cropping is padding with zeros.
        
        When gradients flow back through cropping, we pad the gradient to restore
        the original spatial dimensions. Regions that were cropped away receive
        zero gradient.
        
        Args:
            out_grad: Gradient from subsequent operations (cropped size).
            node: Computation graph node containing input information.
        
        Returns:
            Gradient padded to match original input size, with zeros in cropped regions.
        """
        # Calculate padding needed to restore original dimensions
        # Pad at the start (top/left) by the cropping offsets
        h_pad = self.h_cropping
        # Pad at the end (bottom/right) to reach original width
        w_pad = node.inputs[0].shape[3] - (self.w_cropping + self.shape[1])
        return pad(out_grad, h_pad, w_pad)


def crop(x: Tensor, h_cropping: int, w_cropping: int, shape: Optional[tuple] = None) -> Tensor:
    """
    Convenience function to apply cropping to a tensor.
    
    Args:
        x: Input tensor of shape (batch, channels, height, width).
        h_cropping: Starting row index for extraction.
        w_cropping: Starting column index for extraction.
        shape: (height, width) of the region to extract.
    
    Returns:
        Cropped tensor with specified spatial region.
    """
    return Crop(h_cropping, w_cropping, shape)(x)


class Pad1D(TensorOp):
    """
    Padding operation for 3D tensors with sequence dimension.
    
    Adds zeros around the sequence dimension (time axis) of input tensors.
    This is commonly used to:
    - Prepare sequences for FFT-based operations (power-of-2 sizes)
    - Avoid circular convolution artifacts
    - Implement border handling in sequence processing
    
    Input: A tensor of shape (batch_size, seq_len, channels)
           - batch_size: number of samples
           - seq_len: sequence length
           - channels: number of features per timestep
           
    Output: A tensor of shape (batch_size, seq_len+pad_left+pad_right, channels)
            - Padded with zeros on both sides of sequence dimension
    
    Args:
        pad_left: Amount of padding to add at the beginning of sequence.
        pad_right: Amount of padding to add at the end of sequence.
    """
    def __init__(self, pad_left: int, pad_right: int):
        """
        Initialize padding operation for sequence dimension.
        
        Args:
            pad_left: Padding amount at sequence start.
            pad_right: Padding amount at sequence end.
        """
        self.pad_left = pad_left
        self.pad_right = pad_right
        super().__init__()

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Apply zero-padding to sequence dimension.
        
        Args:
            x: Input array of shape (batch_size, seq_len, channels).
        
        Returns:
            Padded array of shape (batch_size, seq_len+pad_left+pad_right, channels).
        """
        # Define padding specification for each dimension
        # Format: ((before_dim0, after_dim0), (before_dim1, after_dim1), ...)
        # No padding on batch (axis 0) and channel (axis 2) dimensions
        # Padding on sequence (axis 1)
        pad_width = (
            (0, 0),                           # No padding for batch dimension
            (self.pad_left, self.pad_right),  # Padding for sequence dimension
            (0, 0),                           # No padding for channel dimension
        )
        padded_x = x.pad(pad_width)
        return padded_x

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of padding is cropping.
        
        When gradients flow back through padding, we remove the padded regions
        to match the original input shape.
        
        Args:
            out_grad: Gradient from subsequent operations (includes padded regions).
            node: Computation graph node.
        
        Returns:
            Gradient cropped to original input size.
        """
        # Get original sequence length from input
        original_seq_len = node.inputs[0].shape[1]
        return crop1d(out_grad, self.pad_left, original_seq_len)

def pad1d(x: Tensor, pad_left: int, pad_right: int) -> Tensor:
    """
    Convenience function to apply 1D padding to a tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, channels).
        pad_left: Padding amount at sequence start.
        pad_right: Padding amount at sequence end.
    
    Returns:
        Zero-padded tensor with increased sequence length.
    """
    return Pad1D(pad_left, pad_right)(x)


class Crop1D(TensorOp):
    """
    Cropping operation for 3D tensors with sequence dimension.
    
    Removes elements from the borders of sequence dimension. This is the inverse
    operation of padding and is used to:
    - Extract subsequences from sequences
    - Remove padding after FFT-based convolution
    - Implement gradient flow through padding operations
    
    Input: A tensor of shape (batch_size, seq_len, channels)
           - batch_size: number of samples
           - seq_len: sequence length before cropping
           - channels: number of features
           
    Output: A tensor of shape (batch_size, crop_len, channels)
            - crop_len: length of extracted subsequence
            - Region extracted starting at start_idx
    
    Args:
        start_idx: Index to start cropping from.
        crop_len: Length of the output sequence to extract.
    """
    def __init__(self, start_idx: int, crop_len: int):
        """
        Initialize cropping operation for sequence dimension.
        
        Args:
            start_idx: Starting index for cropped region.
            crop_len: Length of the output sequence to extract.
        """
        self.start_idx = start_idx
        self.crop_len = crop_len
        super().__init__()

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Extract a subsequence from the sequence dimension.
        
        Args:
            x: Input array of shape (batch_size, seq_len, channels, ...).
        
        Returns:
            Cropped array of shape (batch_size, crop_len, channels, ...).
        """
        # Calculate the boundaries of the region to extract
        start = self.start_idx
        end = self.start_idx + self.crop_len
        
        # Extract the region using array slicing
        # Build a slice tuple for any number of dimensions
        # Keep all dimensions except crop dimension 1 (sequence)
        ndim = len(x.shape)
        slice_tuple = tuple([slice(None), slice(start, end)] + [slice(None)] * (ndim - 2))
        cropped_x = x[slice_tuple]
        return cropped_x

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of cropping is padding with zeros.
        
        When gradients flow back through cropping, we pad the gradient to restore
        the original sequence length. Regions that were cropped away receive
        zero gradient.
        
        Args:
            out_grad: Gradient from subsequent operations (cropped size).
            node: Computation graph node containing input information.
        
        Returns:
            Gradient padded to match original input size, with zeros in cropped regions.
        """
        # Calculate padding needed to restore original dimensions
        # Pad at the start by the cropping offset
        pad_left = self.start_idx
        # Pad at the end to reach original length
        original_seq_len = node.inputs[0].shape[1]
        pad_right = original_seq_len - (self.start_idx + self.crop_len)
        return pad1d(out_grad, pad_left, pad_right)


def crop1d(x: Tensor, start_idx: int, crop_len: int) -> Tensor:
    """
    Convenience function to apply 1D cropping to a tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, channels).
        start_idx: Starting index for extraction.
        crop_len: Length of the sequence to extract.
    
    Returns:
        Cropped tensor with specified subsequence.
    """
    return Crop1D(start_idx, crop_len)(x)


class ComplexMultiply(TensorOp):
    """
    Complex multiplication of two tensors representing complex numbers.
    
    Performs element-wise complex multiplication in the frequency domain, which
    is equivalent to convolution in the spatial/time domain. This is a key operation
    for FFT-based convolution algorithms.
    
    Complex multiplication formula: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    
    Input: Two complex tensors with last dimension encoding [real, imaginary]:
           - a: shape (..., 2) - first complex tensor
           - b: shape (..., 2) - second complex tensor
           Both must have compatible shapes for broadcasting.
           
    Output: Complex tensor of same shape with element-wise complex product
    
    Args:
        device: Device to place the output tensor on.
    """
    def __init__(self, device=None):
        """
        Initialize complex multiplication operation.
        
        Args:
            device: Computation device (CPU/CUDA).
        """
        super().__init__()
        self.device = device

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        """
        Forward pass: Perform complex multiplication element-wise.
        
        Implements: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        
        Args:
            a: First complex array with last dimension [real, imaginary].
            b: Second complex array with last dimension [real, imaginary].
        
        Returns:
            Complex product with same shape.
        """
        # Build slicing tuple dynamically based on number of dimensions
        # Last dimension is always [real, imag]
        ndim = len(a.shape)
        real_slice = tuple([slice(None)] * (ndim - 1) + [0])
        imag_slice = tuple([slice(None)] * (ndim - 1) + [1])
        
        # Extract real and imaginary components from both inputs
        # Broadcast shapes if needed - one input might have batch dim 1
        a_real = a[real_slice]
        a_imag = a[imag_slice]
        b_real = b[real_slice]
        b_imag = b[imag_slice]
        
        # Apply complex multiplication formula
        # Real part: ac - bd (product of reals minus product of imaginaries)
        # Imaginary part: ad + bc (cross terms)
        result = array_api.empty(a.shape, device=self.device)
        result[real_slice] = a_real * b_real - a_imag * b_imag
        result[imag_slice] = a_real * b_imag + a_imag * b_real

        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Compute gradients for complex multiplication.
        
        For complex multiplication z = a * b:
        - ∂L/∂a = ∂L/∂z * conj(b)
        - ∂L/∂b = ∂L/∂z * conj(a)
        
        Where conj() is complex conjugate (negate imaginary part).
        
        Args:
            out_grad: Gradient from subsequent operations.
            node: Computation graph node with inputs a and b.
        
        Returns:
            Tuple of (grad_a, grad_b) with shapes matching inputs.
        """
        # Extract inputs from computation graph
        a, b = node.inputs
        
        # Compute gradients using complex conjugate multiplication
        # grad_a = out_grad * conj(b)
        # grad_b = out_grad * conj(a)
        grad_a = complex_multiply(out_grad, conjugate(b, device=self.device), device=self.device)
        grad_b = complex_multiply(out_grad, conjugate(a, device=self.device), device=self.device)

        return grad_a, grad_b


def complex_multiply(a: Tensor, b: Tensor, device=None) -> Tensor:
    """
    Convenience function for complex multiplication of two tensors.
    
    Args:
        a: First complex tensor with shape (..., 2).
        b: Second complex tensor with shape (..., 2).
        device: Device for computation.
    
    Returns:
        Element-wise complex product.
    """
    return ComplexMultiply(device)(a, b)

class Conjugate(TensorOp):
    """
    Complex conjugation of a tensor representing complex numbers.
    
    Negates the imaginary part of the complex tensor. This operation is often
    used in frequency-domain computations, such as in convolution theorems
    and gradient calculations.
    
    Input: Complex tensor with last dimension encoding [real, imaginary].
    
    Output: Complex tensor of same shape with imaginary part negated.
    
    Args:
        device: Device to place the output tensor on.
    """
    def __init__(self, device=None):
        """
        Initialize complex conjugation operation.
        
        Args:
            device: Computation device (CPU/CUDA).
        """
        super().__init__()
        self.device = device

    def compute(self, x: NDArray) -> NDArray:
        """
        Forward pass: Compute complex conjugate.
        
        Args:
            x: Input complex array with last dimension [real, imaginary].
        
        Returns:
            Complex conjugate array with imaginary part negated.
        """
        # Build slicing tuple dynamically based on number of dimensions
        # Last dimension is always [real, imag], so we access it with [:, ..., 0] and [:, ..., 1]
        ndim = len(x.shape)
        real_slice = tuple([slice(None)] * (ndim - 1) + [0])
        imag_slice = tuple([slice(None)] * (ndim - 1) + [1])
        
        real = x[real_slice]
        imag = x[imag_slice]
        
        conj_imag = -imag
        result = array_api.empty(x.shape, device=self.device)
        result[real_slice] = real
        result[imag_slice] = conj_imag
        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Gradient of conjugation is conjugation.
        
        The derivative of the conjugate operation is itself. Thus, the gradient
        flowing back through this operation is also conjugated.
        
        Args:
            out_grad: Gradient from subsequent operations.
            node: Computation graph node.
        
        Returns:
            Conjugated gradient.
        """
        return conjugate(out_grad, device=self.device)

def conjugate(x: Tensor, device=None) -> Tensor:
    """
    Convenience function to compute complex conjugate of a tensor.
    
    Args:
        x: Input complex tensor with last dimension [real, imaginary].
        device: Device for computation.
    
    Returns:
        Complex conjugate tensor.
    """
    return Conjugate(device)(x)


class CausalConv1DFFT(TensorOp):
    """
    Causal 1D convolution using FFT for efficient computation on sequence data.
    
    Implements causal (non-future-looking) convolution where output at time t
    only depends on inputs at times <= t. This is essential for autoregressive
    models and time-series prediction. Uses FFT-based convolution for O(n log n)
    complexity instead of O(n²) for direct convolution.
    
    The convolution theorem states: convolution in time domain = multiplication
    in frequency domain. This operation:
    1. Pads inputs to avoid circular convolution artifacts
    2. Transforms to frequency domain via FFT
    3. Multiplies frequency representations
    4. Transforms back via IFFT
    5. Extracts valid causal output
    
    Input: Two real tensors:
           - u: Input signal of shape (batch, seq_len, channels)
           - kernel_rev: Reversed kernel of shape (k_len, channels)
                        Expected with lag 0 at index 0 (same as existing causal_conv)
           
    Output: Causal convolution result of shape (batch, seq_len, channels)
            - Each output timestep only depends on current and past inputs
    
    Args:
        fft_len: Length for FFT computation. If None, auto-determined as seq_len + k_len - 1.
                 Larger values may improve performance by using power-of-2 FFT sizes.
        device: Device to place the output tensor on.
        dtype: Data type for the output tensor.
    """

    def __init__(self, fft_len: Optional[int] = None, device=None, dtype: str = "float32"):
        """
        Initialize causal 1D convolution with FFT.
        
        Args:
            fft_len: FFT length for computation. Affects zero-padding and performance.
            device: Computation device (CPU/CUDA).
            dtype: Output data type.
        """
        super().__init__()
        self.fft_len = fft_len
        self.device = device
        self.dtype = dtype

    def compute(self, u: NDArray, kernel_rev: NDArray) -> NDArray:
        """
        Forward pass: Compute causal convolution using FFT.
        
        Args:
            u: Input signal of shape (batch, seq_len, channels).
            kernel_rev: Reversed kernel of shape (k_len, channels).
        
        Returns:
            Causal convolution output of shape (batch, seq_len, channels).
        """
        batch, seq_len, channels = u.shape
        k_len, _ = kernel_rev.shape

        assert kernel_rev.shape[1] == channels

        # Determine FFT length: needs to be at least seq_len + k_len - 1
        # to avoid circular convolution (aliasing) artifacts
        fft_len = self.fft_len or (seq_len + k_len - 1)

        device = self.device or u.device

        # Pad signal and kernel along time axis to FFT length
        # This prevents circular wraparound effects in frequency domain multiplication
        u_padded = u.pad(((0, 0), (0, fft_len - seq_len), (0, 0)))
        k_padded = kernel_rev.pad(((0, fft_len - k_len), (0, 0)))

        # Transform to frequency domain
        # FFT along time axis=1, preserving batch and channel dimensions
        u_f = np.fft.rfft(u_padded.numpy(), n=fft_len, axis=1)          # (batch, n_freq, channels)
        k_f = np.fft.rfft(k_padded.numpy(), n=fft_len, axis=0)          # (n_freq, channels)
        k_f = k_f[None, :, :]                                           # (1, n_freq, channels) - add batch dim

        # Frequency domain multiplication = time domain convolution
        y_f = u_f * k_f                                      # broadcast over batch dimension
        
        # Transform back to time domain
        y = np.fft.irfft(y_f, n=fft_len, axis=1)
        
        # Extract only the first seq_len outputs (causal portion)
        # Discard the extra outputs from padding
        y = y[:, :seq_len, :]
        return array_api.array(y.astype(np.float32), device=device)

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Backward pass: Compute gradients for causal convolution.
        
        Uses the convolution theorem in reverse:
        - Gradient w.r.t. input: convolve output gradient with conjugate of kernel
        - Gradient w.r.t. kernel: convolve input with conjugate of output gradient
        
        Both operations are efficiently computed in frequency domain.
        
        Args:
            out_grad: Gradient from subsequent operations, shape (batch, seq_len, channels).
            node: Computation graph node with inputs u and kernel_rev.
        
        Returns:
            Tuple of (grad_u, grad_kernel) with shapes matching inputs.
        """
        # Extract inputs from computation graph
        u, kernel_rev = node.inputs
        batch, seq_len, channels = u.shape
        k_len, _ = kernel_rev.shape

        fft_len = self.fft_len or (seq_len + k_len - 1)

        device = self.device or u.device

        # Pad all arrays to FFT length for proper gradient computation
        # Convert to Tensors and use pad1d operation
        grad_padded = pad1d(out_grad, 0, fft_len - seq_len)
        u_padded = pad1d(u, 0, fft_len - seq_len)
        kernel_rev = reshape(kernel_rev, (1, k_len, channels))
        k_padded = pad1d(kernel_rev, 0, fft_len - k_len)

        # Transform all arrays to frequency domain
        G = fft1d(grad_padded, n=fft_len, device=device)      # (batch, n_freq, ch, 2) - output gradient
        U = fft1d(u_padded, n=fft_len, device=device)         # (batch, n_freq, ch, 2) - input signal
        K = fft1d(k_padded, n=fft_len, device=device)         # (1, n_freq, ch, 2) - kernel
        K = broadcast_to(K, (batch, K.shape[1], K.shape[2], 2))  # Broadcast over batch

        # Compute gradients in frequency domain using conjugate multiplication
        # Gradient w.r.t. input: output_grad * conj(kernel)
        grad_u_freq = complex_multiply(G, conjugate(K, device=device), device=device)
        # Gradient w.r.t. kernel: output_grad * conj(input)
        grad_k_freq = complex_multiply(G, conjugate(U, device=device), device=device)
        
        grad_u_full = ifft1d(grad_u_freq, n=fft_len, device=device, dtype=self.dtype)
        grad_k_full = ifft1d(grad_k_freq, n=fft_len, device=device, dtype=self.dtype)

        # Extract valid portions and aggregate kernel gradient across batch
        grad_u = crop1d(grad_u_full, 0, seq_len)               # Keep only original sequence length
        grad_k = summation(crop1d(grad_k_full, 0, k_len), axes=(0,))  # Sum over batch, keep only kernel length

        return grad_u, grad_k


def causal_conv1d_fft(u: Tensor, kernel_rev: Tensor, fft_len: Optional[int] = None) -> Tensor:
    """
    Convenience function to perform causal 1D convolution using FFT.
    
    Efficiently computes causal convolution for sequence modeling tasks.
    Useful for temporal convolutional networks, WaveNet-style models, and
    other architectures requiring causal dependencies.
    
    Args:
        u: Input signal tensor of shape (batch, seq_len, channels).
        kernel_rev: Reversed kernel tensor of shape (k_len, channels).
        fft_len: Optional FFT length for computation. Auto-determined if None.
    
    Returns:
        Causally convolved output of shape (batch, seq_len, channels).
    """
    return CausalConv1DFFT(fft_len=fft_len, device=u.device, dtype=u.dtype)(u, kernel_rev)
