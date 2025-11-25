"""The module."""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""
    pass


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )
        self.need_bias = bias

        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
                .reshape((1, out_features))
            )

    def forward(self, X: Tensor) -> Tensor:
        logit = ops.matmul(X, self.weight)
        if self.need_bias:
            logit = ops.add(logit, self.bias.broadcast_to(logit.shape))
        return logit


class Flatten(Module):
    def forward(self, X):
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        num_of_classes = logits.shape[1]
        one_hot = init.one_hot(num_of_classes, y, device=logits.device, dtype=logits.dtype)
        z_y_sum = (logits * one_hot).sum()
        exp_sum = ops.logsumexp(logits, axes=(1,)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]

        if self.training:
            batch_mean = x.sum(axes=0) / batch_size
            x_mean = x - batch_mean.broadcast_to(x.shape)
            std = (x_mean ** 2).sum(axes=0) / batch_size

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * std.detach()
        else:
            x_mean = x - self.running_mean.broadcast_to(x.shape)
            std = self.running_var

        normed = x_mean / ((std + self.eps) ** 0.5).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        s = x.shape
        _x = (
            x.transpose((1, 2))
             .transpose((2, 3))
             .reshape((s[0] * s[2] * s[3], s[1]))
        )
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.reshape(x.sum(axes=-1) / self.dim, (-1, 1))
        x_mean = x - mean.broadcast_to(x.shape)
        std = (((x_mean ** 2).sum(axes=-1).reshape((-1, 1)) / self.dim) + self.eps) ** 0.5
        normed = x_mean / std.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        mask = init.randb(*x.shape, p=1 - self.p, dtype="float32", device=x.device)
        if self.training:
            x_mask = x * mask
            return x_mask / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
