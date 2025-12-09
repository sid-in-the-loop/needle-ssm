# # -------------------------
# # Sentinel dataset + training for Needle (append to file)
# # -------------------------
# import argparse
# from needle.autograd import Tensor
# from needle import ops
# import needle as ndl
# import needle.nn as nn
# import numpy as np
# import time
# from tqdm import tqdm

# from python.needle.nn.nn_sequence import RNN as RNN_impl  # change if your RNN lives elsewhere
# from python.needle.nn.nn_ssm import S4 as S4_impl  # adjust if different


# # Simple Needle Dataset API compatible with the rest of the codebase
# class SentinelDataset:
#     """
#     Generates sequences of integers (tokens). Last token id (vocab_size-1) is the sentinel.
#     Can work in two modes:
#       - on_the_fly: generate a new random sequence for every __getitem__ call (good for train)
#       - fixed: pre-generate sequences once (good for reproducible evaluation)
#     """
#     def __init__(self, sequence_length: int, dataset_size: int, vocab_size: int = 50,
#                  sentinel_prob: float = 0.5, fixed: bool = False, seed: int = 42):
#         self.sequence_length = int(sequence_length)
#         self.dataset_size = int(dataset_size)
#         self.vocab_size = int(vocab_size)
#         self.sentinel_prob = float(sentinel_prob)
#         self.sentinel_id = self.vocab_size - 1
#         self.fixed = fixed
#         if fixed:
#             rng = np.random.RandomState(seed)
#             self._seqs = []
#             self._labels = []
#             for _ in range(self.dataset_size):
#                 seq = rng.randint(0, self.vocab_size - 1, size=self.sequence_length).astype(np.int64)
#                 if rng.rand() < self.sentinel_prob:
#                     pos = rng.randint(0, self.sequence_length)
#                     seq[pos] = self.sentinel_id
#                     label = 1
#                 else:
#                     label = 0
#                 self._seqs.append(seq)
#                 self._labels.append(label)

#     def __len__(self):
#         return self.dataset_size

#     def __getitem__(self, idx):
#         if self.fixed:
#             seq = self._seqs[idx]
#             label = self._labels[idx]
#         else:
#             # on-the-fly generation (non-deterministic)
#             seq = np.random.randint(0, self.vocab_size - 1, size=self.sequence_length).astype(np.int64)
#             if np.random.rand() < self.sentinel_prob:
#                 pos = np.random.randint(0, self.sequence_length)
#                 seq[pos] = self.sentinel_id
#                 label = 1
#             else:
#                 label = 0
#         # return numpy arrays (caller will wrap into ndl.Tensor)
#         return seq, np.array([label], dtype=np.int64)


# # ---- Model wrappers ----
# # Assumes you have a working Embedding, RNN, S4 classes available in your project.
# # If names differ, adjust the imports/constructors below.

# class RNNClassifierLM(nn.Module):
#     def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, device=ndl.cpu(), dtype="float32"):
#         super().__init__()
#         # Use needle.nn.Embedding if available; else adapt to your Embedding class
#         self.embedding = nn.Embedding(vocab_size, embedding_size, device=device, dtype=dtype)
#         # Use your RNN implementation (it expects input shape (seq_len, batch, emb_dim))
#         # I assume apps.models exports RNN (if not, import the RNN you implemented)
#         self.rnn = RNN_impl(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
#         self.linear = nn.Linear(hidden_size, 1, device=device, dtype=dtype)  # binary logit

#     def forward(self, x_seq):
#         # x_seq: (seq_len, batch) numpy or ndl.Tensor of ints
#         # Convert to embedding: embedding expects ndl.Tensor of ints (seq_len, batch)
#         if not isinstance(x_seq, Tensor):
#             x_seq = Tensor(x_seq, device=device)
#         emb = self.embedding(x_seq)  # (seq_len, batch, emb_dim)
#         rnn_out, _ = self.rnn(emb)   # (seq_len, batch, hidden)
#         # mean-pool over time (axis 0)
#         pooled = ops.summation(rnn_out, axes=0) / rnn_out.shape[0]  # (batch, hidden)
#         logits = self.linear(pooled)  # (batch, 1)
#         return logits  # (batch,1)


# class S4Classifier(nn.Module):
#     def __init__(self, vocab_size, embedding_size, s4_hidden, num_layers=1, device=ndl.cpu(), dtype="float32", seq_len=2048, batch_first=False):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_size, device=device, dtype=dtype)
#         # Use your S4 implementation (import from apps.models)
#         self.s4 = S4_impl(embedding_size, s4_hidden, num_layers=num_layers, device=device, dtype=dtype, sequence_len=seq_len, batch_first=batch_first)
#         self.linear = nn.Linear(embedding_size, 1, device=device, dtype=dtype)

#     def forward(self, x_seq):
#         # x_seq: (seq_len, batch)
#         if not isinstance(x_seq, Tensor):
#             x_seq = Tensor(x_seq, device=device)
#         emb = self.embedding(x_seq)  # (seq_len, batch, emb_dim)
#         # S4.forward in your code returns (x, zeros_like(x)) — adapt accordingly
#         s4_out, _ = self.s4(emb)  # expected (batch, seq_len, emb_dim) or (seq_len, batch, emb_dim) depending on impl
#         # normalize to (seq_len, batch, emb_dim) if necessary
#         if s4_out.shape[0] != emb.shape[0]:
#             # if your S4 is batch-first, transpose
#             s4_out = ops.transpose(s4_out, axes=(1,0,2))
#         pooled = ops.summation(s4_out, axes=0) / s4_out.shape[0]
#         logits = self.linear(pooled)  # (batch,1)
#         return logits


# # ---- Training / evaluation utilities ----

# def batch_generator(dataset, batch_size, shuffle=True):
#     idxs = np.arange(len(dataset))
#     if shuffle:
#         np.random.shuffle(idxs)
#     for i in range(0, len(dataset), batch_size):
#         batch_idx = idxs[i : i + batch_size]
#         seqs = []
#         labels = []
#         for j in batch_idx:
#             s, l = dataset[j]
#             seqs.append(s)
#             labels.append(l)
#         # seqs: list of (seq_len,) -> stack into (seq_len, batch)
#         seqs = np.stack(seqs, axis=1)  # (seq_len, batch)
#         labels = np.stack(labels, axis=0)  # (batch, 1)
#         yield seqs, labels

# def evaluate_model(model, dataset, batch_size=32, device=ndl.cpu()):
#     model.eval()
#     correct = 0
#     total = 0
#     for seqs, labels in batch_generator(dataset, batch_size, shuffle=False):
#         # convert to Tensor and send to device
#         seqs_t = Tensor(seqs, device=device)
#         labels_t = Tensor(labels, device=device)
#         logits = model(seqs_t)  # (batch,1)
#         probs = ops.sigmoid(logits)
#         preds = (probs.numpy() > 0.5).astype(np.int32).reshape(-1)
#         true = labels.reshape(-1)
#         correct += int((preds == true).sum())
#         total += preds.shape[0]
#     return correct / total

# def train_model(model, train_dataset, val_dataset, n_epochs=3, lr=1e-3, batch_size=32, optimizer_cls=ndl.optim.Adam, device=ndl.cpu()):
#     opt = optimizer_cls(model.parameters(), lr=lr)
#     for epoch in range(1, n_epochs+1):
#         model.train()
#         start = time.time()
#         total_loss = 0.0
#         total_samples = 0
#         for seqs, labels in tqdm(batch_generator(train_dataset, batch_size, shuffle=True)):
#             print("Sequence batch shape: ", seqs.shape)
#             seqs_t = Tensor(seqs, device=device)
#             labels_t = Tensor(labels.astype(np.float32), device=device)  # shape (batch,1)
#             logits = model(seqs_t)  # (batch,1)
#             # Binary cross entropy with logits: use ops.log and ops.sigmoid
#             # bce_logits = max(x,0) - x*target + log(1+exp(-abs(x)))
#             x = logits
#             t = labels_t
#             max_zero = ops.relu(x)
#             bce = max_zero - x * t + ops.log(1 + ops.exp(-ops.maximum(x, -x)))
#             loss = ops.summation(bce, axes=(0,1)) / bce.shape[0]  # average over batch
#             print("Predictions and loss: ", logits.numpy(), loss.numpy(), t.numpy())
#             opt.reset_grad()
#             loss.backward()
#             print("BACKWARD DONE")
#             opt.step()
#             print("OPTIMIZER STEP DONE")
#             total_loss += float(loss.numpy()) * seqs.shape[1]
#             print(total_loss)
#             total_samples += seqs.shape[1]
#         # end epoch
#         val_acc = evaluate_model(model, val_dataset, batch_size=batch_size)
#         print(f"[{model.__class__.__name__}] Epoch {epoch} train_loss={total_loss/total_samples:.4f} val_acc={val_acc:.4f} time={time.time()-start:.1f}s")

# # ---- main routine to run both models ----
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seq_len", type=int, default=10)
#     parser.add_argument("--vocab", type=int, default=50)
#     parser.add_argument("--train_size", type=int, default=100)
#     parser.add_argument("--val_size", type=int, default=50)
#     parser.add_argument("--batch", type=int, default=2)
#     parser.add_argument("--epochs", type=int, default=4)
#     parser.add_argument("--hidden", type=int, default=64)
#     parser.add_argument("--emb", type=int, default=32)
#     # parser.add_argument("--seq_len", type=int, default=2048)
#     # parser.add_argument("--vocab", type=int, default=50)
#     # parser.add_argument("--train_size", type=int, default=1000)
#     # parser.add_argument("--val_size", type=int, default=512)
#     # parser.add_argument("--batch", type=int, default=1)
#     # parser.add_argument("--epochs", type=int, default=4)
#     # parser.add_argument("--hidden", type=int, default=64)
#     # parser.add_argument("--emb", type=int, default=32)
#     args = parser.parse_args()

#     # create datasets: training is on-the-fly (not fixed), validation is fixed for reproducibility
#     train_ds = SentinelDataset(sequence_length=args.seq_len, dataset_size=args.train_size,
#                                vocab_size=args.vocab, sentinel_prob=0.5, fixed=False)
#     val_ds = SentinelDataset(sequence_length=args.seq_len, dataset_size=args.val_size,
#                              vocab_size=args.vocab, sentinel_prob=0.5, fixed=True, seed=1234)

#     # RNN classifier
#     print("Training RNN classifier")
#     device = ndl.cpu()
#     rnn_model = RNNClassifierLM(vocab_size=args.vocab, embedding_size=args.emb, hidden_size=args.hidden, num_layers=1, device=device)
#     train_model(rnn_model, train_ds, val_ds, n_epochs=args.epochs, lr=1e-3, batch_size=args.batch, optimizer_cls=ndl.optim.Adam)

#     # S4 classifier
#     print("Training S4 classifier")
#     s4_model = S4Classifier(vocab_size=args.vocab, embedding_size=args.emb, s4_hidden=args.hidden, num_layers=1, device=device, seq_len=args.seq_len, batch_first=False)
#     # train_model(s4_model, train_ds, val_ds, n_epochs=args.epochs, lr=1e-3, batch_size=args.batch, optimizer_cls=ndl.optim.Adam)

# -------------------------
# Sentinel dataset + training for Needle
# -------------------------
import argparse
from needle.autograd import Tensor
from needle import ops
import needle as ndl
import needle.nn as nn
import numpy as np
import time
from tqdm import tqdm

from python.needle.nn.nn_sequence import RNN as RNN_impl
from python.needle.nn.nn_ssm import S4 as S4_impl

# -------------------------
# Dataset
# -------------------------
class SentinelDataset:
    def __init__(self, sequence_length: int, dataset_size: int, vocab_size: int = 50,
                 sentinel_prob: float = 0.5, fixed: bool = False, seed: int = 42):
        self.sequence_length = sequence_length
        self.dataset_size = dataset_size
        self.vocab_size = vocab_size
        self.sentinel_prob = sentinel_prob
        self.sentinel_id = self.vocab_size - 1
        self.fixed = fixed
        if fixed:
            rng = np.random.RandomState(seed)
            self._seqs = []
            self._labels = []
            for _ in range(self.dataset_size):
                seq = rng.randint(0, self.vocab_size - 1, size=self.sequence_length).astype(np.int64)
                label = 0
                if rng.rand() < self.sentinel_prob:
                    pos = rng.randint(0, self.sequence_length)
                    seq[pos] = self.sentinel_id
                    label = 1
                self._seqs.append(seq)
                self._labels.append(label)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.fixed:
            seq = self._seqs[idx]
            label = self._labels[idx]
        else:
            seq = np.random.randint(0, self.vocab_size - 1, size=self.sequence_length).astype(np.int64)
            label = 0
            if np.random.rand() < self.sentinel_prob:
                pos = np.random.randint(0, self.sequence_length)
                seq[pos] = self.sentinel_id
                label = 1
        return seq, np.array([label], dtype=np.int64)

# -------------------------
# Batch generator
# -------------------------
def batch_generator(dataset, batch_size, shuffle=True):
    idxs = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, len(dataset), batch_size):
        batch_idx = idxs[i:i+batch_size]
        seqs, labels = [], []
        for j in batch_idx:
            s, l = dataset[j]
            seqs.append(s)
            labels.append(l)
        # seqs: (batch, seq_len) -> S4 expects batch_first
        seqs = np.stack(seqs, axis=0)  # (B,L)
        labels = np.stack(labels, axis=0)  # (B,1)
        yield seqs, labels

# -------------------------
# Models
# -------------------------
class RNNClassifierLM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, device=ndl.cpu(), dtype="float32"):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_size, device=device, dtype=dtype)
        self.rnn = RNN_impl(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, 1, device=device, dtype=dtype)

    def forward(self, x_seq):
        if not isinstance(x_seq, Tensor):
            x_seq = Tensor(x_seq, device=self.device)
        # RNN expects (seq_len, batch)
        x_seq = ops.transpose(x_seq, axes=(1,0))
        emb = self.embedding(x_seq)
        rnn_out, _ = self.rnn(emb)
        pooled = ops.summation(rnn_out, axes=0) / rnn_out.shape[0]
        logits = self.linear(pooled)
        return logits

class S4Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, s4_hidden, num_layers=1, device=ndl.cpu(), dtype="float32", seq_len=2048, batch_first=True):
        super().__init__()
        self.device = device
        self.batch_first = batch_first
        self.embedding = nn.Embedding(vocab_size, embedding_size, device=device, dtype=dtype)
        self.s4 = S4_impl(embedding_size, s4_hidden, num_layers=num_layers, device=device, dtype=dtype, sequence_len=seq_len, batch_first=batch_first)
        self.linear = nn.Linear(embedding_size, 1, device=device, dtype=dtype)

    def forward(self, x_seq):
        if not isinstance(x_seq, Tensor):
            x_seq = Tensor(x_seq, device=self.device)
        emb = self.embedding(x_seq)  # (B,L,E)
        if not self.batch_first:
            emb = ops.transpose(emb, axes=(1,0,2))
        s4_out, _ = self.s4(emb)  # (B,L,E)
        pooled = ops.summation(s4_out, axes=1) / s4_out.shape[1]  # mean over seq_len
        logits = self.linear(pooled)
        return logits

# -------------------------
# Training & evaluation
# -------------------------
def evaluate_model(model, dataset, batch_size=32, device=ndl.cpu()):
    model.eval()
    correct, total = 0, 0
    for seqs, labels in batch_generator(dataset, batch_size, shuffle=False):
        seqs_t = Tensor(seqs, device=device)
        labels_t = Tensor(labels, device=device)
        logits = model(seqs_t)
        probs = ops.sigmoid(logits)
        preds = (probs.numpy() > 0.5).astype(np.int32).reshape(-1)
        correct += int((preds == labels.reshape(-1)).sum())
        total += preds.shape[0]
    return correct / total

def train_model(model, train_dataset, val_dataset, n_epochs=3, lr=1e-3, batch_size=32, optimizer_cls=ndl.optim.Adam, device=ndl.cpu()):
    opt = optimizer_cls(model.parameters(), lr=lr)
    for epoch in range(1, n_epochs+1):
        model.train()
        total_loss, total_samples = 0.0, 0
        start = time.time()
        for seqs, labels in tqdm(batch_generator(train_dataset, batch_size, shuffle=True)):
            seqs_t = Tensor(seqs, device=device)
            labels_t = Tensor(labels.astype(np.float32), device=device)
            logits = model(seqs_t)
            # BCE loss with logits
            x = logits
            t = labels_t
            max_zero = ops.relu(x)
            bce = max_zero - x * t + ops.log(1 + ops.exp(-ops.maximum(x, -x)))
            loss = ops.summation(bce, axes=(0,1)) / bce.shape[0]
            opt.reset_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.numpy()) * seqs.shape[0]
            total_samples += seqs.shape[0]
        val_acc = evaluate_model(model, val_dataset, batch_size=batch_size, device=device)
        print(f"[{model.__class__.__name__}] Epoch {epoch} train_loss={total_loss/total_samples:.4f} val_acc={val_acc:.4f} time={time.time()-start:.1f}s")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--emb", type=int, default=32)
    args = parser.parse_args()

    train_ds = SentinelDataset(sequence_length=args.seq_len, dataset_size=args.train_size,
                               vocab_size=args.vocab, sentinel_prob=0.5, fixed=False)
    val_ds = SentinelDataset(sequence_length=args.seq_len, dataset_size=args.val_size,
                             vocab_size=args.vocab, sentinel_prob=0.5, fixed=True, seed=1234)

    device = ndl.cpu()

    print("Training RNN classifier")
    rnn_model = RNNClassifierLM(vocab_size=args.vocab, embedding_size=args.emb, hidden_size=args.hidden, device=device)
    train_model(rnn_model, train_ds, val_ds, n_epochs=args.epochs, batch_size=args.batch, optimizer_cls=ndl.optim.Adam, device=device)

    print("Training S4 classifier")
    s4_model = S4Classifier(vocab_size=args.vocab, embedding_size=args.emb, s4_hidden=args.hidden, num_layers=1, device=device, seq_len=args.seq_len, batch_first=True)
    train_model(s4_model, train_ds, val_ds, n_epochs=args.epochs, batch_size=args.batch, optimizer_cls=ndl.optim.Adam, device=device)
