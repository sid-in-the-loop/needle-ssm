"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU, Tanh, Linear


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1+ops.exp(-x))**(-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        k = (1 / hidden_size)

        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low= -k ** 0.5,
                high= k ** 0.5,
                device=device,
                dtype=dtype
            ),
            requires_grad=True
        )

        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low= -k ** 0.5,
                high= k ** 0.5,
                device=device,
                dtype=dtype
            ),
            requires_grad=True
        )

        if self.bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low= -k ** 0.5,
                    high= k ** 0.5,
                    device=device,
                    dtype=dtype
                ),
                requires_grad=True
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low= -k ** 0.5,
                    high= k ** 0.5,
                    device=device,
                    dtype=dtype
                ),
                requires_grad=True
            )
        self.nonlinearity = Tanh() if nonlinearity == 'tanh' else ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        h = h if h is not None else init.zeros(
            bs,
            self.hidden_size,
            device=X.device,
            dtype=X.dtype,
            requires_grad=False
        )

        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
            out += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
                
        return self.nonlinearity(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for i in range(num_layers - 1):
            rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        self.rnn_cells = rnn_cells
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        Xs = ops.split(X, 0)
        hs = ops.split(h0, 0) if h0 is not None else [None] * self.num_layers
        out = []
        for t, x in enumerate(Xs):
            hiddens = []
            for l, model in enumerate(self.rnn_cells):
                x = model(x, hs[l])
                hiddens.append(x)
            out.append(x)
            hs = hiddens
        out = ops.stack(out, 0)
        hs = ops.stack(hs, 0)
        return out, hs
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        k = 1.0 / hidden_size
        bound = k ** 0.5

        self.W_ih = Parameter(
            init.rand(input_size, 4 * hidden_size,
                      low=-bound, high=bound,
                      device=device, dtype=dtype),
            requires_grad=True
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size,
                      low=-bound, high=bound,
                      device=device, dtype=dtype),
            requires_grad=True
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(4 * hidden_size,
                          low=-bound, high=bound,
                          device=device, dtype=dtype),
                requires_grad=True
            )
            self.bias_hh = Parameter(
                init.rand(4 * hidden_size,
                          low=-bound, high=bound,
                          device=device, dtype=dtype),
                requires_grad=True
            )

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, X, h=None):
        batch_sz = X.shape[0]
        H = self.hidden_size

        if h is None:
            h0 = init.zeros(batch_sz, H, device=X.device, dtype=X.dtype, requires_grad=False)
            c0 = init.zeros(batch_sz, H, device=X.device, dtype=X.dtype, requires_grad=False)
        else:
            h0, c0 = h

        gates = X @ self.W_ih + h0 @ self.W_hh

        if self.bias:
            gates = gates + self.bias_ih.broadcast_to(gates.shape) \
                          + self.bias_hh.broadcast_to(gates.shape)

        gates_3d = ops.reshape(gates, (batch_sz, 4, H))
        i, f, g, o = ops.split(gates_3d, 1)

        i = ops.reshape(i, (batch_sz, H))
        f = ops.reshape(f, (batch_sz, H))
        g = ops.reshape(g, (batch_sz, H))
        o = ops.reshape(o, (batch_sz, H))

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)

        c1 = f * c0 + i * g
        h1 = o * self.tanh(c1)

        return h1, c1



class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 bias=True, device=None, dtype="float32"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        cells = []
        cells.append(LSTMCell(input_size, hidden_size, bias=bias,
                              device=device, dtype=dtype))
        for _ in range(1, num_layers):
            cells.append(LSTMCell(hidden_size, hidden_size, bias=bias,
                                  device=device, dtype=dtype))
        self.lstm_cells = cells

    def forward(self, X, h=None):
        seq_len = X.shape[0]
        batch_sz = X.shape[1]
        H = self.hidden_size

        Xs = list(ops.split(X, 0))

        if h is None:
            h0 = init.zeros(self.num_layers, batch_sz, H,
                            device=X.device, dtype=X.dtype, requires_grad=False)
            c0 = init.zeros(self.num_layers, batch_sz, H,
                            device=X.device, dtype=X.dtype, requires_grad=False)
        else:
            h0, c0 = h

        # FIX: convert TensorTuple to Python list so assignment works
        hs = list(ops.split(h0, 0))
        cs = list(ops.split(c0, 0))

        outputs = []

        for t in range(seq_len):
            x_t = Xs[t]
            for l, cell in enumerate(self.lstm_cells):
                h_l, c_l = cell(x_t, (hs[l], cs[l]))
                hs[l] = h_l
                cs[l] = c_l
                x_t = h_l
            outputs.append(x_t)

        output = ops.stack(outputs, 0)
        h_n = ops.stack(hs, 0)
        c_n = ops.stack(cs, 0)

        return output, (h_n, c_n)





class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            ),
            requires_grad=True
        )
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape[0], x.shape[1]
        # 1. one-hot
        x = init.one_hot(self.num_embeddings, x, dtype=self.dtype, device=self.device, requires_grad=True)
        # 2. linear
        return ops.matmul(
            x.reshape((-1, self.num_embeddings)),
            self.weight
            ).reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION