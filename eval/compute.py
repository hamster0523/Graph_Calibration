import numpy as np

def stepwise_judge_ece(confidences, judge_scores, n_bins=15):

    c = np.asarray(confidences, dtype=float)
    p = np.asarray(judge_scores, dtype=float)

    if c.shape != p.shape:
        raise ValueError(f"Shape mismatch: confidences {c.shape}, judge_scores {p.shape}")

    N = len(c)
    if N == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.digitize(c, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0

    for b in range(n_bins):
        mask = (bin_indices == b)
        n_b = mask.sum()
        if n_b == 0:
            continue  

        c_hat = c[mask].mean()
        p_hat = p[mask].mean()

        ece += (n_b / N) * abs(c_hat - p_hat)

    return float(ece)

