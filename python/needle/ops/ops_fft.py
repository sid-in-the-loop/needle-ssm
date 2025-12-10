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
    
    Input: A real tensor of shape (batch_size, height, width, cin, cout)
    Output: A tensor of shape (batch_size, height, width, cin, cout, 2) 
    """
    def __init__(self, shape: Optional[tuple] = None, device=None, dtype="float32"):
        """
        2D FFT operation on spatial dimensions (axes 1 and 2).
        """
        super().__init__()
        self.shape = shape
        self.device = device
        self.dtype = dtype

    def compute(self, x: NDArray) -> NDArray:
        # Apply 2D FFT along spatial dimensions (axes 2 and 3) for each channel
        fft_result = np.fft.rfft2(x.numpy(), s=self.shape, axes=(2, 3))
        real = fft_result.real.astype(np.float32)
        img = fft_result.imag.astype(np.float32)
        N, C_in, P, Q = real.shape

        result = array_api.empty((N, C_in, P, Q, 2), device=self.device)
        result[:, :, :, :, 0] = array_api.array(real, device=self.device)
        result[:, :, :, :, 1] = array_api.array(img, device=self.device)

        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient of FFT is IFFT.
        The derivative of DFT with respect to input is the inverse DFT matrix.
        """
        return ifft(out_grad, device=self.device, dtype=self.dtype)

def fft(x: Tensor, shape=None, device=None, dtype="float32") -> Tensor:
    return FFT(shape=shape, device=device, dtype=dtype)(x)


class IFFT(TensorOp):
    """
    Inverse Fast Fourier Transform operation for batched 2D arrays with channels.
    
    Input: A tensor of shape (batch_size, height, width, cin, cout, 2) where 
    Output: A real tensor of shape (batch_size, height, width, cin, cout) 
    """
    def __init__(self, shape: Optional[tuple] = None, device=None, dtype="float32"):
        """
        2D IFFT operation on spatial dimensions (axes 1 and 2).
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.shape = shape

    def compute(self, x: NDArray) -> NDArray:
        """
        Compute 2D IFFT on batched arrays with channels.
        
        Args:
            x: NDArray of shape (batch_size, height, width, cout, 2)
        
        Returns:
            NDArray of shape (batch_size, height, width, channels) - real values
        """
        batch_size, cout, height, width, complex = x.shape
        assert complex == 2, "Last dimension must be 2 (real/imag pairs)"
        
        # Extract real and imaginary parts
        real = x[:, :, :, :, 0].compact().reshape((batch_size, cout, height, width))
        imag = x[:, :, :, :, 1].compact().reshape((batch_size, cout, height, width))
        
        complex_arr = real.numpy() + 1j * imag.numpy()

        ifft_result = np.fft.irfft2(complex_arr, axes=(2, 3), s=self.shape)
        
        # Take only the real part (imaginary part should be ~0 for real input)
        result = ifft_result.real.astype(np.float32)
        result = array_api.array(result, device=self.device)
        
        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient of IFFT is FFT.
        The derivative of inverse DFT with respect to input is the forward DFT matrix.
        """
        return fft(out_grad, device=self.device, dtype=self.dtype)

def ifft(x: Tensor, shape: Optional[tuple] = None, device=None, dtype="float32") -> Tensor:
    return IFFT(shape=shape, device=device, dtype=dtype)(x)


class FFT1D(TensorOp):
    """
    Real 1D FFT over the sequence axis.

    Input: (batch, seq_len, channels)
    Output: (batch, freq, channels, 2) where last dim is (real, imag)
    """

    def __init__(self, n: Optional[int] = None, device=None, dtype: str = "float32"):
        super().__init__()
        self.n = n
        self.device = device
        self.dtype = dtype

    def compute(self, x: NDArray) -> NDArray:
        batch, seq_len, channels = x.shape
        n = self.n or seq_len
        fft_res = np.fft.rfft(x.numpy(), n=n, axis=1)
        real = fft_res.real.astype(np.float32)
        imag = fft_res.imag.astype(np.float32)
        out = array_api.empty((batch, real.shape[1], channels, 2), device=self.device)
        out[:, :, :, 0] = array_api.array(real, device=self.device)
        out[:, :, :, 1] = array_api.array(imag, device=self.device)
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        return ifft1d(out_grad, n=self.n, device=self.device, dtype=self.dtype)


def fft1d(x: Tensor, n: Optional[int] = None, device=None, dtype: str = "float32") -> Tensor:
    return FFT1D(n=n, device=device, dtype=dtype)(x)


class IFFT1D(TensorOp):
    """
    Real 1D inverse FFT over the sequence axis.

    Input: (batch, freq, channels, 2)
    Output: (batch, seq_len, channels)
    """

    def __init__(self, n: Optional[int] = None, device=None, dtype: str = "float32"):
        super().__init__()
        self.n = n
        self.device = device
        self.dtype = dtype

    def compute(self, x: NDArray) -> NDArray:
        batch, freq, channels, complex_dim = x.shape
        assert complex_dim == 2, "Last dimension must be 2 (real/imag pairs)"
        real = x[:, :, :, 0].compact().numpy()
        imag = x[:, :, :, 1].compact().numpy()
        complex_arr = real + 1j * imag
        seq_len = self.n
        ifft_res = np.fft.irfft(complex_arr, n=seq_len, axis=1)
        result = array_api.array(ifft_res.astype(np.float32), device=self.device)
        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        return fft1d(out_grad, n=self.n, device=self.device, dtype=self.dtype)


def ifft1d(x: Tensor, n: Optional[int] = None, device=None, dtype: str = "float32") -> Tensor:
    return IFFT1D(n=n, device=device, dtype=dtype)(x)


class Pad(TensorOp):
    """
    Pad operation for 4D tensors.
    
    Input: A tensor of shape (batch_size, height, width, channels)
    Output: A tensor padded with zeros on height and width dimensions.
    """
    def __init__(self, h_padding: int, w_padding: int):
        """
        Padding operation on spatial dimensions (axes 1 and 2).
        """
        self.h_padding = h_padding
        self.w_padding = w_padding
        super().__init__()

    def compute(self, x: NDArray) -> NDArray:
        """
        Compute padding on batched arrays with channels.
        
        Args:
            x: NDArray of shape (batch_size, height, width, channels)
        
        Returns:
            NDArray of shape (batch_size, height + padding, width + padding, channels)
        """
        # Split padding symmetrically; place any odd remainder on the trailing side
        pad_width = (
            (0, 0),
            (0, 0),
            (self.h_padding, self.h_padding),
            (self.w_padding, self.w_padding),
        )
        padded_x = x.pad(pad_width)
        return padded_x

    def gradient(self, out_grad: Tensor, node: Tensor):
        return crop(out_grad, self.h_padding, self.w_padding)

def pad(x: Tensor, h_padding: int, w_padding: int) -> Tensor:
    return Pad(h_padding, w_padding)(x)


class Crop(TensorOp):
    """
    Crop operation for 4D tensors.
    
    Input: A tensor of shape (batch_size, height, width, channels)
    Output: A tensor cropped on height and width dimensions.
    """
    def __init__(self, h_cropping: int, w_cropping: int, shape: Optional[tuple] = None):
        """
        Cropping operation on spatial dimensions (axes 1 and 2).
        """
        self.h_cropping = h_cropping
        self.w_cropping = w_cropping
        self.shape = shape
        super().__init__()

    def compute(self, x: NDArray) -> NDArray:
        """
        Compute cropping on batched arrays with channels.
        
        Args:
            x: NDArray of shape (batch_size, height, width, channels)
        
        Returns:
            NDArray of shape (batch_size, height - h_cropping, width - w_cropping, channels)
        """
        # Split cropping symmetrically; place any odd remainder on the trailing side
        start_h = self.h_cropping
        end_h = self.h_cropping + self.shape[0]
        start_w = self.w_cropping
        end_w = self.w_cropping + self.shape[1]
        cropped_x = x[:, :, start_h:end_h, start_w:end_w]
        return cropped_x

    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient of Crop is padding the cropped area with zeros.
        """
        h_pad = self.h_cropping
        w_pad = node.inputs[0].shape[3] - (self.w_cropping + self.shape[1])
        return pad(out_grad, h_pad, w_pad)


def crop(x: Tensor, h_cropping: int, w_cropping: int, shape: Optional[tuple] = None) -> Tensor:
    return Crop(h_cropping, w_cropping, shape)(x)

class ComplexMultiply(TensorOp):
    """
    Complex multiplication of two tensors representing complex numbers.
    
    Input: Two tensors of shape (..., 2) where last dimension is (real, imag)
    Output: A tensor of shape (..., 2) representing the complex product.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def compute(self, x: NDArray, kernel: NDArray) -> NDArray:
        """
        Compute complex multiplication.
        
        Args:
            a: NDArray of shape (..., 2)
            b: NDArray of shape (..., 2)
        
        Returns:
            NDArray of shape (..., 2)
        """
        batch_size, cout, cin, height, width, cmplx = x.shape
        assert cmplx == 2, "Last dimension must be 2 (real/imag pairs)"

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        x_real = x[:, :, :, :, :, 0]
        x_imag = x[:, :, :, :, :, 1]
        k_real = kernel[:, :, :, :, :, 0]
        k_imag = kernel[:, :, :, :, :, 1]
        
        # (ac - bd) + (ad + bc)i
        result = array_api.empty((batch_size, cout, cin, height, width, 2), device=self.device)
        result[:, :, :, :, :, 0] = x_real * k_real - x_imag * k_imag
        result[:, :, :, :, :, 1] = x_real * k_imag + x_imag * k_real

        result = result.sum(axis=2)  # Sum over cin dimension

        return result

    def gradient(self, out_grad: Tensor, node: Tensor):
        x, kernel = node.inputs
        out_r = out_grad[:, :, :, :, 0].compact().numpy()
        out_i = out_grad[:, :, :, :, 1].compact().numpy()

        x_r = x[:, :, :, :, :, 0].compact().numpy()
        x_i = x[:, :, :, :, :, 1].compact().numpy()
        k_r = kernel[:, :, :, :, :, 0].compact().numpy()
        k_i = kernel[:, :, :, :, :, 1].compact().numpy()

        out_complex = out_r + 1j * out_i  # (batch, cout, height, width)
        x_complex = x_r + 1j * x_i        # (batch, cout, cin, height, width)
        k_complex = k_r + 1j * k_i        # (batch, cout, cin, height, width)

        grad_x_complex = out_complex[:, :, None, :, :] * np.conjugate(k_complex)
        grad_k_complex = out_complex[:, :, None, :, :] * np.conjugate(x_complex)

        grad_x = array_api.empty(x.shape, device=self.device)
        grad_k = array_api.empty(kernel.shape, device=self.device)

        grad_x[:, :, :, :, :, 0] = array_api.array(
            grad_x_complex.real.astype(np.float32), device=self.device
        )
        grad_x[:, :, :, :, :, 1] = array_api.array(
            grad_x_complex.imag.astype(np.float32), device=self.device
        )

        grad_k[:, :, :, :, :, 0] = array_api.array(
            grad_k_complex.real.astype(np.float32), device=self.device
        )
        grad_k[:, :, :, :, :, 1] = array_api.array(
            grad_k_complex.imag.astype(np.float32), device=self.device
        )

        return grad_x, grad_k


def complex_multiply(a: Tensor, b: Tensor, device=None) -> Tensor:
    return ComplexMultiply(device)(a, b)


class CausalConv1DFFT(TensorOp):
    """
    Causal 1D convolution using FFT on sequence axis.

    Expects kernel already ordered with lag 0 at index 0 (same as existing causal_conv).
    """

    def __init__(self, fft_len: Optional[int] = None, device=None, dtype: str = "float32"):
        super().__init__()
        self.fft_len = fft_len
        self.device = device
        self.dtype = dtype

    def compute(self, u: NDArray, kernel_rev: NDArray) -> NDArray:
        batch, seq_len, channels = u.shape
        k_len = kernel_rev.shape[0]
        assert kernel_rev.shape[1] == channels

        fft_len = self.fft_len or (seq_len + k_len - 1)

        device = self.device or u.device

        # Pad signal and kernel along time axis
        u_pad = np.pad(u.numpy(), ((0, 0), (0, fft_len - seq_len), (0, 0)))
        k_pad = np.pad(kernel_rev.numpy(), ((0, fft_len - k_len), (0, 0)))

        # FFT along time axis=1, keep channel dimension
        u_f = np.fft.rfft(u_pad, n=fft_len, axis=1)          # (batch, n_freq, channels)
        k_f = np.fft.rfft(k_pad, n=fft_len, axis=0)          # (n_freq, channels)
        k_f = k_f[None, :, :]                                # (1, n_freq, channels)

        y_f = u_f * k_f                                      # broadcast over batch
        y = np.fft.irfft(y_f, n=fft_len, axis=1)
        y = y[:, :seq_len, :]
        return array_api.array(y.astype(np.float32), device=device)

    def gradient(self, out_grad: Tensor, node: Tensor):
        u, kernel_rev = node.inputs
        batch, seq_len, channels = u.shape
        k_len = kernel_rev.shape[0]
        fft_len = self.fft_len or (seq_len + k_len - 1)

        device = self.device or u.device

        grad_np = np.pad(
            out_grad.numpy(), ((0, 0), (0, fft_len - seq_len), (0, 0))
        )
        u_np = np.pad(u.numpy(), ((0, 0), (0, fft_len - seq_len), (0, 0)))
        k_np = np.pad(kernel_rev.numpy(), ((0, fft_len - k_len), (0, 0)))

        G = np.fft.rfft(grad_np, n=fft_len, axis=1)      # (batch, n_freq, ch)
        U = np.fft.rfft(u_np, n=fft_len, axis=1)         # (batch, n_freq, ch)
        K = np.fft.rfft(k_np, n=fft_len, axis=0)         # (n_freq, ch)
        K = K[None, :, :]                                # (1, n_freq, ch)

        grad_u_full = np.fft.irfft(G * np.conjugate(K), n=fft_len, axis=1)
        grad_k_full = np.fft.irfft(G * np.conjugate(U), n=fft_len, axis=1)

        grad_u = grad_u_full[:, :seq_len, :]
        grad_k = grad_k_full[:, :k_len, :].sum(axis=0)

        from ..autograd import Tensor  # local import to avoid cycle

        grad_u_tensor = Tensor(
            grad_u.astype(np.float32),
            device=device,
            dtype=self.dtype,
            requires_grad=u.requires_grad,
        )
        grad_k_tensor = Tensor(
            grad_k.astype(np.float32),
            device=device,
            dtype=self.dtype,
            requires_grad=kernel_rev.requires_grad,
        )

        return grad_u_tensor, grad_k_tensor


def causal_conv1d_fft(u: Tensor, kernel_rev: Tensor, fft_len: Optional[int] = None) -> Tensor:
    return CausalConv1DFFT(fft_len=fft_len, device=u.device, dtype=u.dtype)(u, kernel_rev)
