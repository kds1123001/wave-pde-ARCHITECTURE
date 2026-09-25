
import numpy as np

def make_batch(H, W, dk, C, K, batch_size, rng, min_dist=None, max_dist=None):
   
    B = batch_size
    X = np.zeros((B, H, W, dk + C), dtype=np.float64)
    query_pos = np.zeros((B, 2), dtype=np.int64)
    labels = np.zeros((B,), dtype=np.int64)
    dist = np.zeros((B,), dtype=np.int64)

    cells = [(r, c) for r in range(H) for c in range(W)]

    for b in range(B):
   
        attempts = 0
        while True:
            attempts += 1
            if attempts > 10000:
                raise ValueError(
                    f"can't satisfy distance range [{min_dist},{max_dist}] on a {H}x{W} grid "
                    f"(max possible Chebyshev distance is {max(H, W) - 1}) -- check bucket bounds"
                )
            qr, qc = cells[rng.integers(len(cells))]
            if min_dist is None and max_dist is None:
                mr, mc = cells[rng.integers(len(cells))]
            else:
                candidates = [
                    (r, c) for (r, c) in cells
                    if (r, c) != (qr, qc)
                    and min_dist <= max(abs(r - qr), abs(c - qc)) <= max_dist
                ]
                if not candidates:
                    continue
                mr, mc = candidates[rng.integers(len(candidates))]
            if (mr, mc) != (qr, qc):
                break

        used = {(qr, qc), (mr, mc)}
        key_match = rng.standard_normal(dk)
        key_match /= (np.linalg.norm(key_match) + 1e-8)
        cls = rng.integers(C)

      
        X[b, mr, mc, :dk] = key_match
        X[b, mr, mc, dk + cls] = 1.0

       
        X[b, qr, qc, :dk] = key_match

       
        n_distractors = max(K - 2, 0)
        tries = 0
        while n_distractors > 0 and tries < 50:
            r, c = cells[rng.integers(len(cells))]
            tries += 1
            if (r, c) in used:
                continue
            used.add((r, c))
            dkey = rng.standard_normal(dk)
            dkey /= (np.linalg.norm(dkey) + 1e-8)
            X[b, r, c, :dk] = dkey
            X[b, r, c, dk + rng.integers(C)] = 1.0
            n_distractors -= 1

        query_pos[b] = (qr, qc)
        labels[b] = cls
        dist[b] = max(abs(mr - qr), abs(mc - qc))

    return X, query_pos, labels, dist
