from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxZ = Z.max(axis=Z.ndim-1, keepdims=True) # (N, 1)
        exp_sum = array_api.sum(array_api.exp(Z - maxZ.broadcast_to(Z.shape)), axis=Z.ndim-1, keepdims=True) # (N, 1)
        logexpsum = array_api.log(exp_sum) + maxZ # (N, 1)
        return Z - logexpsum.broadcast_to(Z.shape) # (N, C)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        """
        SEE: 查看这里的解释, 用jacobian行列式推导正确的答案 https://stackoverflow.com/questions/35304393/trying-to-understand-code-that-computes-the-gradient-wrt-to-the-input-for-logsof
        """
        shape = list(node.shape)
        shape_reduce = list(shape)
        shape_reduce[-1] = 1
        return out_grad - (exp(node.detach()) * out_grad.sum(axes=node.ndim-1).reshape(tuple(shape_reduce)).broadcast_to(tuple(shape)))
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxZ = Z.max(axis=self.axes, keepdims=True)
        expSum = array_api.sum(array_api.exp(Z - maxZ.broadcast_to(Z.shape)), axis=self.axes) # if axes = 2, (a,b,c) -> (a,b) reduce了维度
        return array_api.log(expSum) + Z.max(axis=self.axes, keepdims=False)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        input_shape = z.shape
        tmp_reshape = list(input_shape)

        if self.axes is None:
            return out_grad * exp(z - node)
        elif isinstance(self.axes, tuple):
            for summed_axe in self.axes:
                tmp_reshape[summed_axe] = 1
        elif isinstance(self.axes, int):
            tmp_reshape[self.axes] = 1
        
        node_new = reshape(node, tuple(tmp_reshape))
        grad_new = reshape(out_grad, tuple(tmp_reshape))

        # return grad_new * exp(z - node_new)
        return grad_new.broadcast_to(z.shape) * exp(z - node_new.broadcast_to(z.shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    # axes 我们这里只支持指定 1 个维度
    return LogSumExp(axes=axes)(a)

