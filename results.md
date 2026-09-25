# Empirical Section: Does a Multigrid Wave-PDE Architecture Solve Long-Range Recall?

## 1. Motivation

The theoretical write-up leading to this experiment made three architectural claims that
hadn't been tested against an actual implementation:

1. A local, CFL-bound wave-PDE model should fail at long-range information routing because
   a wavefront needs `O(N^{1/D})` timesteps to physically cross an `N`-node, `D`-dimensional grid.
2. A hierarchical / multigrid coarse-grid hop should fix this by giving distant nodes an
   `O(1)`-step interaction path.
3. If it doesn't, the likely culprits are aliasing from spatial pooling, phase-destructive
   prolongation, or surrogate-gradient degradation at the coarse/fine interface.

This section reports what actually happens when these claims are implemented and tested,
not what the equations predict in isolation. All code is included alongside this document
(see `code/`) and was gradient-checked against numerical differentiation before any reported
result was trusted.

## 2. Setup

**Task — 2D long-range associative recall.** An 8×8 grid holds `K=4` cells with a random
key vector plus a one-hot value class (`C=4` classes). One cell is the query: it carries a
key matching another cell elsewhere on the grid, with its value channel zeroed. The model
must read out the correct class at the query position. The Chebyshev distance between the
query and its match is controlled, so accuracy can be measured as a function of required
routing distance. Chance accuracy is 25%.

**Models.**
- **SSM baseline** — bidirectional diagonal linear recurrence (S4D/LRU-style), the standard
  comparison point for any new sequence architecture.
- **Local-only wave** — leapfrog discretization of a damped wave PDE on the grid's graph
  Laplacian, with a surrogate-gradient threshold nonlinearity (logistic surrogate on a
  per-node energy threshold), `T=6` steps, no long-range shortcuts.
- **Multigrid wave (one-shot mix)** — local wave core plus, at every fine step, a single
  restrict → dense-mix → prolong hop through a 4×4 coarse grid.
- **Multigrid wave (learned pooling)** — same, but restriction/prolongation are learned
  linear projections over each 2×2 block instead of fixed averaging/spreading.
- **Multigrid wave (recurrent coarse)** — the coarse level is a second, smaller copy of the
  same wave-PDE machinery (own Laplacian, own leapfrog recurrence, own threshold
  nonlinearity), continuously driven by the restricted fine state and persisting across the
  full rollout, rather than being re-mixed from scratch each step.

**Infrastructure.** No GPU/torch available in this environment; all models are implemented
in NumPy on top of a small hand-written reverse-mode autograd engine (`autograd.py`). The
engine was validated against finite-difference gradients (`test_autograd.py`), including a
targeted check for the unbatched-constant-matrix-times-batched-tensor pattern used by the
fixed graph Laplacian, since a real bug of exactly that shape was caught and fixed during
development (see §5).

**Protocol.** Each model was trained for 400 steps (batch size 24, Adam, lr 0.02) and
evaluated on 15–20 fresh batches per distance bucket, sampled independently of training
data at each step (online/streaming setting — no held-out split needed since data is
generated on the fly). All reported numbers are mean ± std over 5 seeds, with both the data
sampling and the weight initialization re-seeded per run.

## 3. Headline Result

| Distance bucket | SSM baseline | Local-only wave | Multigrid, one-shot mix | Multigrid, learned pool | Multigrid, recurrent coarse |
|---|---|---|---|---|---|
| 1–2 | 0.564 | 0.739 | 0.743 | 0.648 ± 0.092 | 0.616 ± 0.066 |
| 3–4 | 0.492 | 0.319 | 0.313 | 0.324 ± 0.035 | **0.561 ± 0.023** |
| 5–6 | 0.471 | 0.260 | 0.256 | 0.260 ± 0.017 | **0.441 ± 0.023** |
| 7   | 0.426 | 0.263 | 0.243 | 0.243 ± 0.015 | **0.338 ± 0.048** |

(Chance = 0.250. SSM and local-only/one-shot-mix numbers are 5-seed means; std omitted
above for the first three columns but was consistently small, <0.03, except at dist 1–2.)

Three findings, in the order they were established:

**(a) The CFL-bound failure mode is real.** The local-only model collapses from 74% at
distance 1–2 to chance (26%) at distance ≥3, with `T=6` leapfrog steps on an 8×8 grid.

**(b) The naive multigrid fix does not work.** Neither the fixed-average-pool nor the fully
learned-pooling version of the one-shot coarse hop improves on the local-only model at any
distance ≥3 — all three sit within noise of each other and of chance.

**(c) A structural, not cosmetic, fix does work.** Giving the coarse level genuine
persistent dynamics of its own — rather than one dense mix per fine step — roughly doubles
accuracy at distance 3–6 and lifts distance-7 accuracy from chance to consistently
above-chance across every one of 5 seeds (28–43% individually). This comes with a real
trade-off: near-range accuracy (dist 1–2) drops from 74% to 62%, plausibly a
capacity/attention trade-off from splitting the model's dynamics across two coupled PDEs
instead of one.

## 4. Diagnostic Tests (Ruling Hypotheses In and Out)

Three hypotheses were proposed for why the one-shot multigrid hop failed. Each was tested
directly rather than assumed:

| Hypothesis | Test | Result |
|---|---|---|
| Surrogate-gradient degradation across the coarse/fine interface | Measured `‖∂L/∂coarse‖` vs. `‖∂L/∂fine‖` on a partially-trained model, on a long-range batch | Ratio ≈ 0.78 — gradients reach the coarse pathway at comparable magnitude to the fine pathway. **Ruled out.** |
| Spatial aliasing from average-pooling destroys the exact key vector | Replaced fixed average-pool restriction/prolongation with fully learned linear projections over each block | Accuracy curve unchanged within noise at every distance. **Ruled out.** |
| CFL / time-horizon limit is the real bottleneck for the local model | Increased local-only model's rollout from `T=6` to `T=20` (exceeding the grid diameter of ~14) | Distance-7 accuracy recovered from 25% → 51%, at a cost to near-range accuracy (74%→59%). **Confirmed.** |
| *(emergent, not original)* one-shot coarse mixing lacks the "depth" that made the `T` increase work | Replaced the one-shot dense mix with a persistent, recurrent coarse-level wave PDE | Distance 3–7 accuracy roughly doubled vs. one-shot mix; consistently above chance across seeds. **Confirmed, load-bearing.** |

One correction to the original hypothesis list: the proposed "destructive interference from
naive prolongation" failure mode assumed prolongation overwrites the fine state. The
implementation here was already additive/residual
(`psi_next = psi_next + gate · Prolongate(coarse)`), so that literal mechanism didn't apply
— but the *learned gate* controlling that residual's strength settled at only ~15% of its
allowed range in the one-shot-mix model, which is a related but distinct symptom (the
network chose to use the pathway weakly, rather than the pathway being architecturally
destructive).

## 5. Engineering Notes / What Broke Along the Way

Worth including for reproducibility and because these are the kind of bugs that
specifically arise from taking a physics-flavored architecture literally rather than
treating it as pure metaphor:

- **Matmul backward with an unbatched operand.** The autograd engine's first version
  accumulated gradients in-place before reducing batch dimensions, which is silently correct
  when both matmul operands are batched but crashes (or would silently corrupt) when a fixed
  constant matrix — like a graph Laplacian — is multiplied against a batched tensor. Caught
  by a shape-mismatch crash, fixed, and covered by a dedicated regression test in
  `test_autograd.py`.
- **Infinite rejection-sampling loop.** An early evaluation config asked for query/match
  pairs at Chebyshev distance 9–14 on an 8×8 grid, whose maximum possible distance is 7. The
  task generator's rejection-sampling loop had no upper bound on retries, so it hung
  indefinitely instead of failing. Fixed by bounding distance buckets to what the grid can
  actually produce, and by adding an explicit retry cap that raises a descriptive error
  instead of hanging.
- All custom manual-backward ops (the block-scatter used in the learned-pooling model, the
  coarse-level recurrence) were spot-checked against finite differences before their
  accuracy numbers were trusted, following the same discipline as the analytical
  discretize-then-optimize argument in the theoretical draft — reverse-mode autodiff over
  the discrete update equations is exactly that adjoint method, just executed instead of
  derived.

## 6. Limitations

- Single task, single grid size (8×8), single random-seed range (5 seeds). No sweep over
  `T`, coarse-grid resolution, or number of multigrid levels beyond the two tested.
- 400 training steps is a small budget; the near/far trade-off in the recurrent-coarse model
  in particular has not been checked for sensitivity to more training.
- No compute/wall-clock comparison against the SSM baseline is reported. The recurrent-coarse
  model runs two coupled PDEs (fine and coarse) per step rather than one; if this class of
  model is sized up beyond a toy grid, that cost needs to be measured before making any
  efficiency claim relative to attention or SSMs.
- The recurrent-coarse model's improvement, while consistent, has not been tested against a
  deeper hierarchy (3+ levels) or against simply giving the local-only model more capacity
  at fixed `T`, so it is not yet established that the coarse recurrence specifically —
  rather than "more overall recurrent depth of any kind" — is what mattered.

## 7. Files

```
code/
  autograd.py       -- reverse-mode autograd engine (gradient-checked)
  test_autograd.py  -- unit gradient checks for the engine
  task.py           -- 2D long-range associative recall task generator
  models.py         -- SSM baseline + 4 wave-PDE model variants
  train.py          -- single-seed training/eval driver
  multi_seed.py     -- 5-seed statistics driver
```

To reproduce the headline table: run `train.py` for the SSM/local/one-shot-mix numbers and
`multi_seed.py` (editing the model constructor used) for each multigrid variant.
