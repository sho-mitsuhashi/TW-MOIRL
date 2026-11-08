import argparse
import csv
from itertools import combinations

import numpy as np

GROUPS = ("Control", "Stimulant", "Rescue")


def statistic(c, s, r):
    return abs(np.mean(s) - np.mean(c)) - abs(np.mean(r) - np.mean(c))


def directional_test(values, labels):
    values, labels = np.asarray(values, dtype=np.float64), np.asarray(labels)
    if values.ndim != 1 or values.shape != labels.shape:
        raise ValueError("values and labels must be one-dimensional and of the same length")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    unknown = sorted(set(labels.tolist()) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown condition labels: {unknown}")
    c, s, r = (values[labels == g] for g in GROUPS)
    if min(len(c), len(s), len(r)) == 0:
        raise ValueError("each condition needs at least one value")
    t_obs = statistic(c, s, r)
    pooled = np.concatenate([s, r])
    null = []
    for idx in combinations(range(len(pooled)), len(s)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(idx)] = True
        null.append(statistic(c, pooled[mask], pooled[~mask]))
    n_ge = int(np.sum(np.array(null) >= t_obs - 1e-12))
    return {"T": float(t_obs), "p": n_ge / len(null), "n_ge": n_ge, "n_perm": len(null),
            "n": {"Control": len(c), "Stimulant": len(s), "Rescue": len(r)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preferences")
    args = parser.parse_args()
    with open(args.preferences) as f:
        rows = list(csv.DictReader(f))
    res = directional_test([float(r["w1"]) for r in rows], [r["condition"] for r in rows])
    print(f"T = {res['T']:.4f}, p = {res['p']:.4f} ({res['n_ge']}/{res['n_perm']})")


if __name__ == "__main__":
    main()
