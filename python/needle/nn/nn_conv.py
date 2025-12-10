"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, BatchNorm2d, ReLU

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        ### BEGIN YOUR SOLUTION
        self.padding = kernel_size // 2
        kernel_sqaure = kernel_size ** 2
        self.weight = Parameter(
                init.kaiming_uniform(
                    fan_in=in_channels*kernel_sqaure, fan_out=out_channels*kernel_sqaure, 
                    shape=(kernel_size, kernel_size, in_channels, out_channels), 
                    dtype=dtype,
                    device=device,
                    requires_grad=True
                )
            )
        self.bias = None
        if bias:
            bias_bound = 1 / ((in_channels * kernel_sqaure) ** 0.5)
            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=-bias_bound, high=bias_bound,
                    dtype=dtype,
                    device=device,
                    requires_grad=True)
            )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x in NCHW format, outputs also in NCHW format
        
        # NCHW -> NHWC
        x = ops.transpose(
            ops.transpose(x, (1, 2)),
            (2,3)
        )
        x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            x = x + ops.broadcast_to(self.bias, x.shape)

        x = ops.transpose(
            ops.transpose(x, (1,3)),
            (2,3)
        ) # NHWC -> NCWH -> NCHW
        return x
        ### END YOUR SOLUTION

class ConvBN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__() # training <- true
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, bias, device=device, dtype=dtype)
        self.bn = BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class FastConv2d(Conv):
    """
    Convolution using Fast Fourier Transform.
    Accepts inputs in NCHW format, outputs also in NCHW format
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        # Get dimensions
        batch_size, in_channels, in_height, in_width = x.shape # NCHW
        _, _, _, out_channels = self.weight.shape # KH, KW, C_in, C_out

        pad = self.kernel_size // 2
        Ho = in_height - self.kernel_size + pad * 2 + 1
        Wo = in_width - self.kernel_size + pad * 2 + 1

        # Linear convolution result size after padding    
        x = ops.pad(x, pad, pad)
        P = in_height + pad * 2 + self.kernel_size - 1
        Q = in_width + pad * 2 + self.kernel_size - 1

        x = ops.fft(x, shape=(P, Q), device=x.device, dtype=x.dtype) 
        xh, xw = x.shape[2], x.shape[3]
        x = ops.reshape(x, (batch_size, 1, in_channels, xh, xw, 2))
        x = ops.broadcast_to(x, (batch_size, out_channels, in_channels, xh, xw, 2))

        kernel = self.weight
        kernel = ops.transpose(ops.transpose(kernel, (0, 2)), (1, 3))  # KH, KW, C_in, C_out -> C_in, C_out, KH, KW
        kernel = ops.transpose(kernel, (0, 1)) # C_in, C_out, KH, KW -> C_out, C_in, KH, KW

        kernel = ops.flip(kernel, axes=(2, 3))  # Flip kernel for cross-correlation
        kernel = ops.fft(kernel, shape=(P, Q), device=x.device, dtype=x.dtype)

        kh, kw = kernel.shape[2], kernel.shape[3]
        kernel = ops.reshape(kernel, (1, out_channels, in_channels, kh, kw, 2))
        kernel = ops.broadcast_to(kernel, (batch_size, out_channels, in_channels, kh, kw, 2))

        fft_out = ops.complex_multiply(x, kernel, device=x.device)        
        
        # Sum over input channels to get (batch, out_channels, height, width, 2)
        # fft_out has shape (batch, out_channels, in_channels, height, width, 2)
        # Need to sum over the in_channels dimension (axis=2)
        fft_out = ops.summation(fft_out, axes=(2,))  # Now shape is (batch, out_channels, height, width, 2)
        
        # Inverse FFT to get back to spatial domain
        ifft_out = ops.ifft(fft_out, shape=(P, Q), device=x.device, dtype=x.dtype)

        # Crop to get valid convolution output: (batch, in_h, in_w, out_c)
        # For valid convolution, output size is (in_h - k_h + 1, in_w - k_w + 1)
        # But we want same padding, so output should be same as input
        out = ops.crop(ifft_out, self.kernel_size - 1, self.kernel_size - 1, shape=(Ho, Wo))  # N, C_out, Ho, Wo

        # Add bias if present
        if self.bias is not None:
            out = out + ops.broadcast_to(
                ops.reshape(self.bias, (1, self.out_channels, 1, 1)),
                out.shape
            )
        
        # Handle stride
        if self.stride > 1:
            out = ops.undilate(out, (2, 3), self.stride - 1)

        return out
