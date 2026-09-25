import numpy as np
from autograd import Tensor, concat_last

rng_init = np.random.default_rng(42)

def glorot(shape):
    fan_in = shape[0] if len(shape) == 2 else shape[-2]
    fan_out = shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng_init.uniform(-limit, limit, size=shape)


def build_laplacian(H, W):
    # standard 5-point stencil, zero-dirichlet boundary (missing neighbors just don't contribute)
    N = H * W
    L = np.zeros((N, N))
    def idx(r, c):
        return r * W + c
    for r in range(H):
        for c in range(W):
            i = idx(r, c)
            deg = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    L[i, idx(nr, nc)] = 1.0
                    deg += 1
            L[i, i] = -deg
    return L


def build_multigrid_ops(H, W, factor=2):
    # simple 2x2 average pooling for restriction, transpose (scaled) for prolongation
    Hc, Wc = H // factor, W // factor
    N, Nc = H * W, Hc * Wc
    P = np.zeros((Nc, N))
    for rc in range(Hc):
        for cc in range(Wc):
            ci = rc * Wc + cc
            for dr in range(factor):
                for dc in range(factor):
                    r, c = rc * factor + dr, cc * factor + dc
                    P[ci, r * W + c] = 1.0 / (factor * factor)
    U = P.T * (factor * factor)  # adjoint of the pooling, rescaled so it's a real "spread" op
    return P, U


class Linear:
    """just a plain affine layer, nothing fancy"""
    def __init__(self, d_in, d_out, bias=True):
        self.W = Tensor(glorot((d_in, d_out)))
        self.b = Tensor(np.zeros(d_out)) if bias else None

    def __call__(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


# ---------------------------------------------------------------------------
# Model 1: bidirectional diagonal linear recurrent SSM (S4D/LRU-ish), the
# baseline every real sequence architecture gets compared to these days.
# ---------------------------------------------------------------------------
class SSMBaseline:
    def __init__(self, d_in, d_state, C, seq_len):
        self.d_state = d_state
        self.in_proj = Linear(d_in, d_state)
        # per-channel decay, parameterized so it always lands in (0,1)
        self.a_fwd_raw = Tensor(rng_init.uniform(-2, 2, size=(1, d_state)))
        self.a_bwd_raw = Tensor(rng_init.uniform(-2, 2, size=(1, d_state)))
        self.readout1 = Linear(2 * d_state, 4 * d_state)
        self.readout2 = Linear(4 * d_state, C)
        self.seq_len = seq_len

    def params(self):
        return (self.in_proj.params() + self.readout1.params() + self.readout2.params()
                + [self.a_fwd_raw, self.a_bwd_raw])

    def forward(self, X_np, query_idx):
        B, N, _ = X_np.shape
        x = Tensor(X_np)
        u = self.in_proj(x)  # (B, N, d)

        a_fwd = self.a_fwd_raw.sigmoid()  # (1, d), in (0,1)
        a_bwd = self.a_bwd_raw.sigmoid()

        h = Tensor(np.zeros((B, self.d_state)))
        fwd_states = []
        for t in range(N):
            h = h * a_fwd + u[:, t, :]
            fwd_states.append(h)

        h = Tensor(np.zeros((B, self.d_state)))
        bwd_states = [None] * N
        for t in reversed(range(N)):
            h = h * a_bwd + u[:, t, :]
            bwd_states[t] = h

        query_idx = np.asarray(query_idx)
        # gather each batch row's hidden state at its own query timestep,
        # keeping the autograd graph intact (can't just numpy-index the .data)
        rows = []
        for b in range(B):
            t = int(query_idx[b])
            f = fwd_states[t][b:b+1, :]
            g = bwd_states[t][b:b+1, :]
            rows.append(concat_last([f, g]))
        hidden = self._stack_batch(rows)
        return self.logits(hidden)

    @staticmethod
    def _stack_batch(tensors):
        data = np.concatenate([t.data for t in tensors], axis=0)
        out = Tensor(data, tuple(tensors), "stack_batch")
        def _backward():
            for i, t in enumerate(tensors):
                t.grad += out.grad[i:i+1]
        out._backward = _backward
        return out

    def logits(self, hidden):
        h = self.readout1(hidden).relu()
        return self.readout2(h)


# ---------------------------------------------------------------------------
# Model 2: local-only wave PDE (leapfrog, fixed graph Laplacian, surrogate
# threshold nonlinearity). No shortcut connections -- info can only move one
# grid cell per timestep, so this is the model that should choke on the
# CFL-bound long-range pairs.
# ---------------------------------------------------------------------------
class LocalWaveModel:
    def __init__(self, d_in, d, C, H, W, T):
        self.H, self.W, self.T, self.d = H, W, T, d
        self.L = Tensor(build_laplacian(H, W), requires_grad=False)  # fixed graph, not learned
        self.in_proj = Linear(d_in, d)
        self.nl_mix = Linear(d, d, bias=False)
        self.readout1 = Linear(d, 2 * d)
        self.readout2 = Linear(2 * d, C)

        # scalar PDE coefficients, squashed into stable ranges via sigmoid
        self.coeff_raw = Tensor(np.array([0.0]))   # -> c^2*dt^2, kept small for CFL stability
        self.damp_raw = Tensor(np.array([0.0]))    # -> gamma*dt
        self.tau = Tensor(np.array([0.3]))
        self.alpha = Tensor(np.array([4.0]))

    def params(self):
        return (self.in_proj.params() + self.nl_mix.params() + self.readout1.params()
                + self.readout2.params() + [self.coeff_raw, self.damp_raw, self.tau, self.alpha])

    def _step(self, psi, psi_prev, L):
        coeff = self.coeff_raw.sigmoid() * 0.18   # keep well under CFL limit for the 5-pt stencil
        damp = self.damp_raw.sigmoid() * 0.3
        lap = L @ psi
        # surrogate threshold firing, applied per-channel on a scalar "energy" proxy
        energy = (psi * psi).sum(axis=-1, keepdims=True)
        fire = ((energy - self.tau) * self.alpha).sigmoid()
        nl_term = fire * self.nl_mix(psi)
        psi_next = psi * 2.0 - psi_prev - damp * (psi - psi_prev) + lap * coeff + nl_term * 0.1
        return psi_next

    def forward(self, X_np, query_idx):
        B, N, _ = X_np.shape
        x = Tensor(X_np)
        psi0 = self.in_proj(x)  # (B, N, d), also the source injection
        psi_prev, psi = psi0, psi0
        L = self.L
        for _ in range(self.T):
            psi_next = self._step(psi, psi_prev, L)
            psi_prev, psi = psi, psi_next
        return self._gather_and_readout(psi, query_idx)

    def _gather_and_readout(self, psi, query_idx):
        B = psi.data.shape[0]
        rows = [psi[b:b+1, int(query_idx[b]), :] for b in range(B)]
        cat = SSMBaseline._stack_batch(rows)
        h = self.readout1(cat).relu()
        return self.readout2(h)


# ---------------------------------------------------------------------------
# Model 3: multigrid wave -- same local leapfrog core as above, but every
# step also does a coarse-grid hop (pool -> dense mix -> unpool) so distant
# cells can talk in O(1) steps instead of waiting for the wavefront to
# physically cross the grid.
# ---------------------------------------------------------------------------
class MultigridWaveModel(LocalWaveModel):
    def __init__(self, d_in, d, C, H, W, T, factor=2):
        super().__init__(d_in, d, C, H, W, T)
        self.P, self.U = build_multigrid_ops(H, W, factor)
        self.P = Tensor(self.P, requires_grad=False)
        self.U = Tensor(self.U, requires_grad=False)
        Nc = self.P.data.shape[0]
        self.coarse_mix = Linear(d, d, bias=True)
        self.coarse_gate_raw = Tensor(np.array([0.0]))  # how much the coarse hop feeds back into fine grid

    def params(self):
        return super().params() + self.coarse_mix.params() + [self.coarse_gate_raw]

    def _coarse_hop(self, psi):
        coarse = self.P @ psi                    # (B, Nc, d) -- restrict to coarse grid
        mixed = self.coarse_mix(coarse).tanh()    # dense all-to-all mixing on the small coarse grid
        fine_correction = self.U @ mixed          # prolongate back
        gate = self.coarse_gate_raw.sigmoid() * 0.3
        return fine_correction * gate

    def forward(self, X_np, query_idx):
        B, N, _ = X_np.shape
        x = Tensor(X_np)
        psi0 = self.in_proj(x)
        psi_prev, psi = psi0, psi0
        L = self.L
        for _ in range(self.T):
            psi_next = self._step(psi, psi_prev, L)
            psi_next = psi_next + self._coarse_hop(psi)
            psi_prev, psi = psi, psi_next
        return self._gather_and_readout(psi, query_idx)


# ---------------------------------------------------------------------------
# Model 4: same as MultigridWaveModel, but the restriction/prolongation are
# LEARNED instead of fixed average-pool/transpose. This tests the aliasing
# hypothesis directly: does average-pooling smear the exact key vector into
# noise, and does letting the net learn what to keep fix it?
# ---------------------------------------------------------------------------
class LearnedPoolMultigridWaveModel(LocalWaveModel):
    def __init__(self, d_in, d, C, H, W, T, factor=2):
        super().__init__(d_in, d, C, H, W, T)
        self.factor = factor
        self.Hc, self.Wc = H // factor, W // factor
        self.Nc = self.Hc * self.Wc
        block = factor * factor
        # restriction: concat the (factor x factor) block's raw features and
        # project down -- no averaging, the net decides what survives
        self.restrict = Linear(block * d, d)
        # prolongation: project coarse state back up to a full block, instead
        # of just copying/spreading the same value to every fine cell
        self.prolong = Linear(d, block * d)
        self.coarse_mix = Linear(d, d, bias=True)
        self.coarse_gate_raw = Tensor(np.array([0.0]))
        # fixed index map so we can gather/scatter blocks without a python loop per batch
        self._block_idx = self._make_block_index(H, W, factor)

    @staticmethod
    def _make_block_index(H, W, factor):
        Hc, Wc = H // factor, W // factor
        idx = np.zeros((Hc * Wc, factor * factor), dtype=np.int64)
        for rc in range(Hc):
            for cc in range(Wc):
                ci = rc * Wc + cc
                k = 0
                for dr in range(factor):
                    for dc in range(factor):
                        r, c = rc * factor + dr, cc * factor + dc
                        idx[ci, k] = r * W + c
                        k += 1
        return idx

    def params(self):
        return (super().params() + self.restrict.params() + self.prolong.params()
                + self.coarse_mix.params() + [self.coarse_gate_raw])

    def _coarse_hop(self, psi):
        B = psi.data.shape[0]
        d = self.d
        # gather each block's cells and concat their features -- (B, Nc, factor^2 * d)
        blocks = psi[:, self._block_idx, :]           # (B, Nc, block, d) via fancy indexing
        blocks_flat = blocks.reshape(B, self.Nc, -1)  # (B, Nc, block*d)
        coarse = self.restrict(blocks_flat)           # (B, Nc, d) -- learned, not averaged
        mixed = self.coarse_mix(coarse).tanh()
        expanded = self.prolong(mixed)                 # (B, Nc, block*d)
        expanded = expanded.reshape(B, self.Nc, self._block_idx.shape[1], d)
        # scatter back to fine grid positions (differentiable, see _scatter below)
        out = self._scatter(expanded, B, d)
        gate = self.coarse_gate_raw.sigmoid() * 0.3
        return out * gate

    def _scatter(self, expanded, B, d):
        idx = self._block_idx  # (Nc, block)
        N = self.H * self.W
        out_data = np.zeros((B, N, d))
        for ci in range(idx.shape[0]):
            for k, pos in enumerate(idx[ci]):
                out_data[:, pos, :] = expanded.data[:, ci, k, :]
        out = Tensor(out_data, (expanded,), "scatter")
        def _backward():
            g = np.zeros_like(expanded.data)
            for ci in range(idx.shape[0]):
                for k, pos in enumerate(idx[ci]):
                    g[:, ci, k, :] = out.grad[:, pos, :]
            expanded.grad += g
        out._backward = _backward
        return out

    def forward(self, X_np, query_idx):
        B, N, _ = X_np.shape
        x = Tensor(X_np)
        psi0 = self.in_proj(x)
        psi_prev, psi = psi0, psi0
        L = self.L
        for _ in range(self.T):
            psi_next = self._step(psi, psi_prev, L)
            psi_next = psi_next + self._coarse_hop(psi)
            psi_prev, psi = psi, psi_next
        return self._gather_and_readout(psi, query_idx)


# ---------------------------------------------------------------------------
# Model 5: coarse level gets its OWN persistent wave dynamics (own Laplacian,
# own leapfrog recurrence, own threshold nonlinearity), evolving across the
# whole T-step rollout and continuously driven by the restricted fine state.
# The bet from the diagnostics above: a single dense mix per step isn't
# enough "depth" for the coarse pathway to do anything useful; a coarse grid
# small enough to fully mix within T steps (its own diameter << T) should.
# ---------------------------------------------------------------------------
class RecurrentCoarseMultigridWaveModel(LocalWaveModel):
    def __init__(self, d_in, d, C, H, W, T, factor=2):
        super().__init__(d_in, d, C, H, W, T)
        self.Hc, self.Wc = H // factor, W // factor
        Lc = build_laplacian(self.Hc, self.Wc)
        self.Lc = Tensor(Lc, requires_grad=False)
        P_np, U_np = build_multigrid_ops(H, W, factor)
        self.P = Tensor(P_np, requires_grad=False)
        self.U = Tensor(U_np, requires_grad=False)

        # coarse level is a small copy of the same wave machinery, own params
        self.coarse_nl_mix = Linear(d, d, bias=False)
        self.coarse_coeff_raw = Tensor(np.array([0.0]))
        self.coarse_damp_raw = Tensor(np.array([0.0]))
        self.coarse_tau = Tensor(np.array([0.3]))
        self.coarse_alpha = Tensor(np.array([4.0]))
        self.inject_gate_raw = Tensor(np.array([0.0]))   # fine -> coarse driving strength
        self.feedback_gate_raw = Tensor(np.array([0.0])) # coarse -> fine feedback strength

    def params(self):
        return (super().params() + self.coarse_nl_mix.params() +
                [self.coarse_coeff_raw, self.coarse_damp_raw, self.coarse_tau,
                 self.coarse_alpha, self.inject_gate_raw, self.feedback_gate_raw])

    def _coarse_step(self, psi_c, psi_c_prev, source):
        coeff = self.coarse_coeff_raw.sigmoid() * 0.5   # coarse grid is tiny, can afford a looser CFL bound
        damp = self.coarse_damp_raw.sigmoid() * 0.3
        lap = self.Lc @ psi_c
        energy = (psi_c * psi_c).sum(axis=-1, keepdims=True)
        fire = ((energy - self.coarse_tau) * self.coarse_alpha).sigmoid()
        nl_term = fire * self.coarse_nl_mix(psi_c)
        inject = source * (self.inject_gate_raw.sigmoid() * 0.3)
        return psi_c * 2.0 - psi_c_prev - damp * (psi_c - psi_c_prev) + lap * coeff + nl_term * 0.1 + inject

    def forward(self, X_np, query_idx):
        B, N, _ = X_np.shape
        x = Tensor(X_np)
        psi0 = self.in_proj(x)
        psi_prev, psi = psi0, psi0

        coarse0 = self.P @ psi0
        psi_c_prev, psi_c = coarse0, coarse0

        L = self.L
        feedback_gate = self.feedback_gate_raw.sigmoid() * 0.3

        for _ in range(self.T):
            source = self.P @ psi  # restrict current fine state -- drives the coarse PDE
            psi_c_next = self._coarse_step(psi_c, psi_c_prev, source)

            fine_correction = (self.U @ psi_c) * feedback_gate
            psi_next = self._step(psi, psi_prev, L) + fine_correction

            psi_prev, psi = psi, psi_next
            psi_c_prev, psi_c = psi_c, psi_c_next

        return self._gather_and_readout(psi, query_idx)
