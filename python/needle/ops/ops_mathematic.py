"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if self.scalar == 0:
            return out_grad * 0
        
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, -out_grad * lhs / rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axie = list(range(a.ndim))

        if self.axes == None:
            axie[-2], axie[-1] = axie[-1], axie[-2]
            return array_api.transpose(a, axie)
        
        axie[self.axes[0]], axie[self.axes[1]] = axie[self.axes[1]], axie[self.axes[0]]
        return array_api.transpose(a, axie)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print("forward reshape: ", a.shape, self.shape)

        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(out_grad.shape, node.inputs[0].shape)
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        
        input_shape = node.inputs[0].shape
        output_shape = self.shape

        if input_shape == output_shape:
            return out_grad

        # align shapes from the right by padding leading ones
        padded_input = (1,) * (len(output_shape) - len(input_shape)) + input_shape

        summation_axes = []
        for axis, (in_dim, out_dim) in enumerate(zip(padded_input, output_shape)):
            if in_dim == 1 and out_dim > 1:
                summation_axes.append(axis)

        reduced = summation(out_grad, tuple(summation_axes)) if summation_axes else out_grad
        reshaped = reshape(reduced, padded_input)
        return reshape(reshaped, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not isinstance(self.axes, (tuple, list)):
            return a.sum(self.axes)
        if len(self.axes) > 1:
            axes = list(self.axes)[::-1]
            increments = 0
            while len(axes) > 0:
                a = a.sum(axes[-1] - increments)
                axes.pop()
                increments += 1
            return a
        elif len(self.axes) < 1:
            return a.sum()
        else:
            return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        tmp_reshape = list(input_shape)
        if self.axes is None:
            # 求和成1个scalar
            return broadcast_to(out_grad, input_shape)
        
        if isinstance(self.axes, tuple):
            for summed_axe in self.axes:
                tmp_reshape[summed_axe] = 1
        if isinstance(self.axes, int):
            tmp_reshape[self.axes] = 1

        return broadcast_to(reshape(out_grad, tuple(tmp_reshape)), input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class Scan(TensorOp):
    """
    Compute inclusive prefix sum (scan) operation.
    For 1D array: out[i] = sum of a[0] to a[i]
    """
    def __init__(self, axis=0):
        self.axis = axis

    def compute(self, a):
        return a.scan(axis=self.axis)

    def gradient(self, out_grad, node):
        """
        Gradient of scan: reverse scan of the gradient
        """
        # Reverse the gradient and scan it, then reverse again
        from .ops_mathematic import flip
        grad_reversed = flip(out_grad, axes=(self.axis,))
        grad_scanned = grad_reversed.scan(axis=self.axis)
        return flip(grad_scanned, axes=(self.axis,))


def scan(a, axis=0):
    return Scan(axis)(a)


class CausalConv(TensorOp):
    """
    Parallel causal convolution operation.
    Computes all outputs simultaneously using backend kernel.
    """
    def __init__(self, batch, seq_len, channels):
        self.batch = batch
        self.seq_len = seq_len
        self.channels = channels

    def compute(self, u, kernel_rev):
        return u.causal_conv(kernel_rev, self.batch, self.seq_len, self.channels)

    def gradient(self, out_grad, node):
        # Gradient of causal convolution
        u, kernel_rev = node.inputs[0], node.inputs[1]
        from .ops_mathematic import flip
        u_grad = CausalConv(self.batch, self.seq_len, self.channels)(
            out_grad, flip(kernel_rev, axes=(0,))
        )
        kernel_grad = CausalConv(self.batch, self.seq_len, self.channels)(
            flip(out_grad, axes=(1,)), flip(u, axes=(1,))
        )
        kernel_grad = summation(kernel_grad, axes=(0,))
        return u_grad, kernel_grad


def causal_conv_backend(u, kernel_rev, batch, seq_len, channels):
    return CausalConv(batch, seq_len, channels)(u, kernel_rev)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_tmp_grad = matmul(out_grad, transpose(rhs))
        rhs_tmp_grad = matmul(transpose(lhs), out_grad)
        if lhs_tmp_grad.shape != lhs.shape: 
            # Need Reduce
            lhs_tmp_grad = summation(lhs_tmp_grad, axes=tuple(range(len(lhs_tmp_grad.shape) - 2)))
        if rhs_tmp_grad.shape != rhs.shape: 
            # Need Reduce
            rhs_tmp_grad = summation(rhs_tmp_grad, axes=tuple(range(len(rhs_tmp_grad.shape) - 2)))
        return lhs_tmp_grad, rhs_tmp_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)



# ---------- Maximum op (update) ----------
class Maximum(TensorOp):
    def compute(self, a, b):
        return array_api.maximum(a, b)

    def gradient(self, out_grad, node):
        # node.inputs are Tensors; get their underlying NDArray values
        a_t, b_t = node.inputs

        # realize backend data (NDArray) for comparisons
        a_arr = a_t.realize_cached_data()
        b_arr = b_t.realize_cached_data()

        # compute elementwise masks using NDArray operations (returns NDArray)
        # NDArray.__gt__ / __lt__ already return numeric 0/1 arrays per your NDArray impl
        mask_a_arr = a_arr > b_arr   # 1 where a > b, else 0
        mask_b_arr = b_arr > a_arr   # 1 where b > a, else 0

        # wrap masks back into Tensors on same device/dtype, requires_grad=False
        mask_a = Tensor(mask_a_arr, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        mask_b = Tensor(mask_b_arr, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)

        # Multiply masks with out_grad to route gradients
        return out_grad * mask_a, out_grad * mask_b


def maximum(a, b):
    return Maximum()(a, b)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0) # Multiply arguments element-wise
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        # return out_grad * Tensor(a > 0)
        mask = Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -(tanh(node.inputs[0])**2 - 1) * out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)

class Sigmoid(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return (1 + (-a).exp()) ** (-1)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_sigmoid = sigmoid(node.inputs[0])
        return node_sigmoid * (1 - node_sigmoid) * out_grad
        ### END YOUR SOLUTION

def sigmoid(a):
    return Sigmoid()(a)

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))

        out = array_api.empty(
            new_shape, dtype=args[0].dtype, device=args[0].device)

        slices = []
        for i in range(len(new_shape)):
            if i != self.axis:
                slices.append(slice(new_shape[i]))
            else:
                slices.append(0)
        for i in range(len(args)):
            slices[self.axis] = i
            # NOTE reshape
            out[tuple(slices)] = args[i].reshape((1, ) + shape)
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        slices = [slice(0, A.shape[i], 1) if i!=self.axis else 0 for i in range(len(A.shape))]
        tensors = []
        new_shape = tuple([A.shape[s] for s in range(len(A.shape)) if s != self.axis])
        for i in range(A.shape[self.axis]):
            slices[self.axis] = i
            tensors.append(A[tuple(slices)].reshape(new_shape))
        return tuple(tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(tuple(out_grad), self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad, self.axes))
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        if self.dilation == 0:
            return a

        for i, axe in enumerate(self.axes):
            # axe: 当前维度
            new_shape[axe] = new_shape[axe] * (1 + self.dilation)
        
        other = NDArray.make(new_shape, device=a.device)
        other.fill(0)
        sli = [slice(0, new_shape[i], self.dilation + 1) if i in self.axes else slice(0, new_shape[i])
            for i in range(a.ndim)]

        other[tuple(sli)] = a
        return other
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = undilate(out_grad, self.axes, self.dilation)
        return (out_grad,)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        sli = [slice(0, shape[i], self.dilation + 1) if i in self.axes else slice(0, shape[i])
            for i in range(len(shape))]

        return a[tuple(sli)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = dilate(out_grad, self.axes, self.dilation)
        return (out_grad,)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    """ convolution op should accept tensors in the NHWC format """
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        ### padding: a single int – in which case the same value is used for the height and width dimension
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape

        _A = A.pad((
            (0,0), 
            (self.padding, self.padding), 
            (self.padding, self.padding), 
            (0, 0))
         ) if self.padding > 0 else A

        
        Ns, Hs, Ws, Cs = _A.strides

        H_out = (H - K + 2 * self.padding) // self.stride + 1
        W_out = (W - K + 2 * self.padding) // self.stride + 1

        inner_dim = K * K * C_in

        _A = _A.as_strided(
            shape=(N, H_out, W_out, K, K, C_in),
            strides = (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs)
            ).reshape((-1, inner_dim))
        
        out = _A @ (B.reshape((-1, C_out)))
        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # A is the input to the forward pass
        # W is the convolution kernel
        A, W = node.inputs[0], node.inputs[1]
        ### BEGIN YOUR SOLUTION
        # out_grad.shape: (N, H_out, W_out, C_out)
        out_shape = out_grad.shape
        N, H_out, W_out, C_out = out_shape
        K, _, C_in, _ = W.shape
        
        W_tranpose = transpose(flip(W, (0, 1)), (2, 3)) # K, K, C_out, C_in

        _out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1) # 

        A_grad = conv(_out_grad,
                      W_tranpose,
                      stride=1,
                      padding=K - 1 - self.padding)
        
        # The gradients of W must be accumulated over the batches
        # turning batches into channels via transpose/permute
        _out_grad = transpose(transpose(_out_grad, (0, 2)), (0, 1)) # (H_out, W_out, N, c_out)

        A_transpose = transpose(A, (0, 3)) # C H W N
        W_grad = conv(A_transpose, _out_grad,
                      stride=1,
                      padding=self.padding)
        
        W_grad = W_grad.transpose((0, 2)).transpose((0, 1))
        return (A_grad, W_grad)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
