# Wave-PDE Architecture

Testing whether a wave-PDE / multigrid architecture (proposed as an alternative to
transformer attention) actually works, instead of just arguing about it on paper.

No GPU or PyTorch available in the dev environment this was built in, so it's plain NumPy
plus a small hand-rolled reverse-mode autograd engine. See `results.md` for the write-up;
this file is just "how do I run it."

## Setup

```
pip install numpy
```

That's the only dependency.

## Run it

```bash
# sanity-check the autograd engine against finite differences
python code/test_autograd.py

# single-seed run: SSM baseline, local-only wave, one-shot-mix multigrid
python code/train.py

# 5-seed statistics (edit the `ctors` dict at the bottom to swap in whichever
# model variant from models.py you want, e.g. RecurrentCoarseMultigridWaveModel)
python code/multi_seed.py
```

`train.py` and `multi_seed.py` both print a results table at the end; `multi_seed.py` also
dumps raw per-seed numbers to `multiseed_results.npy`.

## What's in `models.py`

- `SSMBaseline` — bidirectional diagonal linear recurrence (S4D/LRU-style)
- `LocalWaveModel` — leapfrog wave PDE, local graph Laplacian only, no shortcuts
- `MultigridWaveModel` — local wave + one-shot coarse-grid pool/mix/unpool hop per step
- `LearnedPoolMultigridWaveModel` — same, but restriction/prolongation are learned instead
  of fixed average-pooling (tests the aliasing hypothesis)
- `RecurrentCoarseMultigridWaveModel` — the coarse level gets its own persistent leapfrog
  dynamics instead of a single dense mix (the variant that actually worked)

## Known limitations

See `results.md` §6. Short version: one grid size, one task, 400 training steps, 5 seeds.
Treat this as a toy-scale pilot, not a scaled-up claim.

## credits

kds1123001 aka yo boi 01101011 01100100 01110011 00110001 00110001 00110010 00110011 00110000 00110000 00110001
