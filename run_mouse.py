import argparse
import csv
import json
import os
import pickle

import numpy as np
import torch

from permutation_test import directional_test
from twmoirl.envs import mouse_grid
from twmoirl.model import fit_tw_moirl
from twmoirl.plotting import plot_bases, plot_preferences

CONDITIONS = {"sal-tas": "Control", "MAP-tas": "Stimulant", "MAP-Sul": "Rescue"}
GAMMA, BETA, N_VI = 0.99, 0.5, 150
D, N_ITERS, M_STEPS = 2, 100, 100
# (lambda_corr, E) grid of the 3-fold cross-validation and the selected setting (Supplementary Information)
CV_GRID = [(lam, e) for lam in (0.0, 0.1) for e in (10, 20, 50)]
LAM_CORR, E_STEPS = 0.1, 20


def load_trajectories(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    trajectories, labels = [], []
    for key, trajs in data.items():
        if key in CONDITIONS:
            trajectories += trajs
            labels += [CONDITIONS[key]] * len(trajs)
    return trajectories, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mouse_trajectories.pkl")
    parser.add_argument("--base_seed", type=int, default=10)
    parser.add_argument("--out")
    args = parser.parse_args()
    if not os.path.exists(args.data):
        raise SystemExit(f"{args.data} not found (available from the corresponding author upon request)")
    # same seed as in the cross-validation runs (base_seed + 100 * setting + fold); the full-data fit is fold 1
    seed = args.base_seed + 100 * (CV_GRID.index((LAM_CORR, E_STEPS)) + 1) + 1
    out = args.out or os.path.join("results", "mouse", f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    env = mouse_grid()
    trajectories, labels = load_trajectories(args.data)
    torch.manual_seed(seed)
    Theta, w, mu, log_sigma = fit_tw_moirl(trajectories, D, env["T"], env["phi"], GAMMA, BETA, N_VI,
                                           N_ITERS, E_STEPS, M_STEPS, lam_corr=LAM_CORR)

    np.savez(os.path.join(out, "params.npz"), Theta=Theta.numpy(), mu=mu.numpy(), log_sigma=log_sigma.numpy())
    with open(os.path.join(out, "preferences.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "condition", "w1", "w2"])
        for i, (label, wi) in enumerate(zip(labels, w.tolist())):
            writer.writerow([i, label, repr(wi[0]), repr(wi[1])])
    plot_bases(Theta, env, os.path.join(out, "bases.pdf"))
    plot_preferences(w[:, 0].numpy(), labels, os.path.join(out, "preferences.pdf"))

    res = directional_test(w[:, 0].numpy(), labels)
    res["seed"] = seed
    with open(os.path.join(out, "test.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"seed {seed}: T = {res['T']:.4f}, p = {res['p']:.4f} ({res['n_ge']}/{res['n_perm']})")


if __name__ == "__main__":
    main()
