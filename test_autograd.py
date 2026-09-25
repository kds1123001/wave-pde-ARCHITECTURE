import numpy as np
from autograd import Tensor

np.random.seed(0)

def numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x.data[idx]
        x.data[idx] = orig + eps
        f1 = f().data.copy()
        x.data[idx] = orig - eps
        f2 = f().data.copy()
        x.data[idx] = orig
        g[idx] = (f1 - f2) / (2 * eps)
    return g

def check(name, f, tensors):
    out = f()
    out.backward()
    ok = True
    for t in tensors:
        t.grad[...] = 0
    out2 = f()
    out2.backward()
    for i, t in enumerate(tensors):
        analytic = t.grad.copy()
        t.grad[...] = 0
        numeric = numerical_grad(f, t)
        err = np.abs(analytic - numeric).max()
        rel = err / (np.abs(numeric).max() + 1e-8)
        status = "OK" if rel < 1e-3 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {name} wrt tensor{i}: max_abs_err={err:.2e} rel_err={rel:.2e} [{status}]")
    return ok

A = Tensor(np.random.randn(3, 4))
B = Tensor(np.random.randn(4, 5))
c = Tensor(np.random.randn(3, 5))

all_ok = True
all_ok &= check("matmul+sum", lambda: (A @ B).sum(), [A, B])
all_ok &= check("mul+sigmoid+sum", lambda: (A.sigmoid() * 2.0).sum(), [A])
all_ok &= check("tanh+sum", lambda: (A.tanh()).sum(), [A])
all_ok &= check("relu+sum", lambda: ((A - 0.1).relu()).sum(), [A])
all_ok &= check("pow+sum", lambda: ((A * A + 1.0) ** 0.5).sum(), [A])
all_ok &= check("matmul chain", lambda: (((A @ B) + c).tanh()).sum(), [A, B, c])

W = Tensor(np.random.randn(5, 3))
labels = np.array([0, 2, 1])
def ce_fn():
    logits = A @ B @ W
    loss, _ = logits.softmax_cross_entropy(labels)
    return loss
all_ok &= check("softmax_ce", ce_fn, [A])

print("\nALL OK" if all_ok else "\nSOME CHECKS FAILED")
