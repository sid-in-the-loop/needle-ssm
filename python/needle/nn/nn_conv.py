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