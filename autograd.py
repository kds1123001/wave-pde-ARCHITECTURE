"""
Minimal reverse-mode autograd engine, numpy-backed.
Supports just enough ops to build SSM / wave-PDE models: matmul, elementwise
arithmetic, sigmoid/tanh, sum/mean, reshape, indexing, and softmax cross-entropy.

This is the practical stand-in for the "adjoint sensitivity" backward pass
discussed analytically in the paper draft: reverse-mode autodiff over the
discrete update equations IS the discretize-then-optimize adjoint method.
"""
import numpy as np

class Tensor:
    __slots__ = ("data", "grad", "_children", "_backward", "requires_grad", "_op")

    def __init__(self, data, children=(), op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._children = children
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self._op = op

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"

    # ---------- helpers for broadcasting-safe grad accumulation ----------
    @staticmethod
    def _unbroadcast(grad, shape):
        # sum-reduce grad to match target shape (undo numpy broadcasting)
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # ---------------------------- ops ----------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -1.0 * other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,), f"**{p}")
        def _backward():
            self.grad += self._unbroadcast((p * self.data ** (p - 1)) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _backward():
            # compute the raw (possibly over-batched) grads first, THEN reduce
            # down to the operand's actual shape -- doing the reduction after
            # an in-place += blows up when one operand isn't batched (e.g. a
            # fixed NxN operator multiplied against a batched BxNxD tensor)
            g_self = out.grad @ other.data.swapaxes(-1, -2)
            g_self = self._unbroadcast(g_self, self.data.shape)
            self.grad += g_self
            g_other = self.data.swapaxes(-1, -2) @ out.grad
            g_other = self._unbroadcast(g_other, other.data.shape)
            other.grad += g_other
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")
        def _backward():
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        return self.sum() * (1.0 / n)

    def reshape(self, *shape):
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        def _backward():
            self.grad += out.grad.reshape(old_shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,), "transpose")
        inv = np.argsort(axes)
        def _backward():
            self.grad += out.grad.transpose(*inv)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")
        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self.grad += g
        out._backward = _backward
        return out

    def clip(self, lo, hi):
        out = Tensor(np.clip(self.data, lo, hi), (self,), "clip")
        def _backward():
            mask = (self.data >= lo) & (self.data <= hi)
            self.grad += mask * out.grad
        out._backward = _backward
        return out

    def softmax_cross_entropy(self, labels):
        # self.data: (B, C) logits; labels: (B,) int array
        x = self.data - self.data.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        probs = ex / ex.sum(axis=-1, keepdims=True)
        B = x.shape[0]
        logp = np.log(probs[np.arange(B), labels] + 1e-12)
        loss_val = -logp.mean()
        out = Tensor(loss_val, (self,), "softmax_ce")
        def _backward():
            g = probs.copy()
            g[np.arange(B), labels] -= 1.0
            g /= B
            self.grad += g * out.grad
        out._backward = _backward
        return out, probs

    # ---------------------------- backward ----------------------------
    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


def concat_last(tensors):
    """Concatenate a list of Tensors along the last axis (manual op)."""
    datas = [t.data for t in tensors]
    out_data = np.concatenate(datas, axis=-1)
    out = Tensor(out_data, tuple(tensors), "concat")
    splits = np.cumsum([d.shape[-1] for d in datas])[:-1]
    def _backward():
        grads = np.split(out.grad, splits, axis=-1)
        for t, g in zip(tensors, grads):
            t.grad += g
    out._backward = _backward
    return out
