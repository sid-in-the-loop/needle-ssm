from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from .ops_tuple import *
from .ops_mathematic import *

import numpy as np
from ..backend_selection import array_api

# TODO: Add support for fft / ifft ops to array_api. Replace complex number handling by numpy.

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
        # Gradient implementation can be added if needed
        raise NotImplementedError("Gradient for ComplexMultiply is not implemented.")


def complex_multiply(a: Tensor, b: Tensor, device=None) -> Tensor:
    return ComplexMultiply(device)(a, b)
