"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = {}  

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.detach()
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            if self.momentum != 0.0:
                u_prev = self.u.get(id(p), 0)
                u_new = self.momentum * u_prev + (1 - self.momentum) * grad
                self.u[id(p)] = u_new
                update = u_new
            else:
                update = grad

            p.data = p.data - self.lr * update

    def clip_grad_norm(self, max_norm=0.25):
        # Not required for HW2
        pass



class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}  # first moment
        self.v = {}  # second moment

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.detach()
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            m_prev = self.m.get(id(p), 0)
            v_prev = self.v.get(id(p), 0)

            # update biased first and second moment 
            m_new = self.beta1 * m_prev + (1 - self.beta1) * grad
            v_new = self.beta2 * v_prev + (1 - self.beta2) * (grad * grad)

            self.m[id(p)] = m_new
            self.v[id(p)] = v_new

            # bias correction
            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)

