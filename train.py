import numpy as np
import time
from autograd import Tensor
from task import make_batch
from models import SSMBaseline, LocalWaveModel, MultigridWaveModel


H, W = 8, 8
N = H * W
DK = 6     
C = 4      
K = 4      
D_STATE = 12
BATCH = 24
STEPS = 400
LR = 0.02
WAVE_T = 6  

seed = 0
rng = np.random.default_rng(seed)


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


def flatten_query(query_pos):
    return query_pos[:, 0] * W + query_pos[:, 1]


def run_epoch(model, n_batches, train=True, opt=None, dist_range=None):
    correct, total, loss_sum = 0, 0, 0.0
    for _ in range(n_batches):
        min_d, max_d = (dist_range if dist_range else (1, max(H, W)))
        X, qpos, labels, dist = make_batch(H, W, DK, C, K, BATCH, rng, min_d, max_d)
        X_flat = X.reshape(BATCH, N, DK + C)
        qidx = flatten_query(qpos)

        logits = model.forward(X_flat, qidx)
        loss, probs = logits.softmax_cross_entropy(labels)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        preds = probs.argmax(axis=-1)
        correct += (preds == labels).sum()
        total += BATCH
        loss_sum += loss.data.item()
    return loss_sum / n_batches, correct / total


def train_model(name, model, n_steps=STEPS):
    opt = Adam(model.params(), lr=LR)
    t0 = time.time()
    for step in range(n_steps):
        loss, acc = run_epoch(model, 1, train=True, opt=opt, dist_range=(1, max(H, W)))
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  [{name}] step {step:4d}  loss {loss:.3f}  acc {acc:.3f}  ({time.time()-t0:.1f}s)")
    return model


max_dist_possible = max(H, W) - 1  # Chebyshev distance on an 8x8 grid of indices 0..7 tops out at 7

def eval_by_distance(name, model, buckets=((1, 2), (3, 4), (5, 6), (7, 7)), n_batches=20):
    print(f"  eval [{name}] accuracy by query-match distance:")
    results = {}
    for lo, hi in buckets:
        _, acc = run_epoch(model, n_batches, train=False, dist_range=(lo, hi))
        results[(lo, hi)] = acc
        print(f"    dist [{lo:2d},{hi:2d}] -> acc {acc:.3f}")
    return results


if __name__ == "__main__":
    d_in = DK + C

    print("training SSM baseline...")
    ssm = SSMBaseline(d_in, D_STATE, C, N)
    train_model("ssm", ssm)
    ssm_results = eval_by_distance("ssm", ssm)

    print("\ntraining local-only wave model...")
    local_wave = LocalWaveModel(d_in, D_STATE, C, H, W, WAVE_T)
    train_model("local-wave", local_wave)
    local_results = eval_by_distance("local-wave", local_wave)

    print("\ntraining multigrid wave model...")
    mg_wave = MultigridWaveModel(d_in, D_STATE, C, H, W, WAVE_T)
    train_model("mg-wave", mg_wave)
    mg_results = eval_by_distance("mg-wave", mg_wave)

    print("\n=== summary ===")
    print(f"{'dist bucket':<14}{'ssm':>8}{'local-wave':>14}{'mg-wave':>10}")
    for bucket in ssm_results:
        lo, hi = bucket
        print(f"[{lo:2d},{hi:2d}]".ljust(14)
              + f"{ssm_results[bucket]:>8.3f}"
              + f"{local_results[bucket]:>14.3f}"
              + f"{mg_results[bucket]:>10.3f}")
