import numpy as np
from train import (H, W, N, DK, C, K, D_STATE, BATCH, STEPS, LR, WAVE_T,
                    Adam, run_epoch, max_dist_possible)
from models import SSMBaseline, LocalWaveModel, MultigridWaveModel
import task

BUCKETS = ((1, 2), (3, 4), (5, 6), (7, 7))
N_SEEDS = 5

def train_and_eval(model_ctor, seed):
    import train as train_mod
    import models as models_mod
   
    train_mod.rng = np.random.default_rng(seed)
    models_mod.rng_init = np.random.default_rng(seed + 10_000)

    d_in = DK + C
    model = model_ctor(d_in)
    opt = Adam(model.params(), lr=LR)
    for step in range(STEPS):
        run_epoch(model, 1, train=True, opt=opt, dist_range=(1, max(H, W)))

    accs = {}
    for lo, hi in BUCKETS:
        _, acc = run_epoch(model, 15, train=False, dist_range=(lo, hi))
        accs[(lo, hi)] = acc
    return accs


def ctor_ssm(d_in):
    return SSMBaseline(d_in, D_STATE, C, N)

def ctor_local(d_in):
    return LocalWaveModel(d_in, D_STATE, C, H, W, WAVE_T)

def ctor_mg(d_in):
    return MultigridWaveModel(d_in, D_STATE, C, H, W, WAVE_T)


if __name__ == "__main__":
    results = {"ssm": [], "local-wave": [], "mg-wave": []}
    ctors = {"ssm": ctor_ssm, "local-wave": ctor_local, "mg-wave": ctor_mg}

    for seed in range(N_SEEDS):
        print(f"=== seed {seed} ===")
        for name, ctor in ctors.items():
            accs = train_and_eval(ctor, seed * 1000 + hash(name) % 97)
            results[name].append(accs)
            print(f"  {name}: " + ", ".join(f"[{lo},{hi}]={accs[(lo,hi)]:.3f}" for lo, hi in BUCKETS))

    print("\n=== mean +/- std over", N_SEEDS, "seeds ===")
    print(f"{'dist bucket':<14}{'ssm':>16}{'local-wave':>18}{'mg-wave':>16}")
    summary = {}
    for lo, hi in BUCKETS:
        row = f"[{lo:2d},{hi:2d}]".ljust(14)
        summary[(lo, hi)] = {}
        for name in ["ssm", "local-wave", "mg-wave"]:
            vals = np.array([r[(lo, hi)] for r in results[name]])
            summary[(lo, hi)][name] = (vals.mean(), vals.std())
            row += f"{vals.mean():.3f}+/-{vals.std():.3f}".rjust(18 if name != "ssm" else 16)
        print(row)

    np.save("multiseed_results.npy", results, allow_pickle=True)
    print("\nsaved raw results to multiseed_results.npy")
