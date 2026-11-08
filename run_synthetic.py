import argparse
import csv
import os

import torch

from twmoirl.envs import ENVS, generate_demos
from twmoirl.metrics import auc_inv, evd, r_pair
from twmoirl.model import fit_tw_moirl

GAMMA, BETA, N_VI = 0.95, 0.5, 150
# (E, I) selected by mean EVD over the ten seeds (Supplementary Information, Hyperparameter Selection)
SELECTED = {"GoalPuddle-5x5": (50, 60), "TrapGrid-9x9": (20, 100), "XYTradeoff-9x9": (50, 60)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="GoalPuddle-5x5", choices=list(ENVS))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--n_traj", type=int, default=20)
    parser.add_argument("--e_steps", type=int)
    parser.add_argument("--m_steps", type=int, default=100)
    parser.add_argument("--n_iters", type=int)
    parser.add_argument("--out", default="results/synthetic.csv")
    args = parser.parse_args()
    e_steps = args.e_steps or SELECTED[args.env][0]
    n_iters = args.n_iters or SELECTED[args.env][1]

    torch.manual_seed(args.seed)
    env = ENVS[args.env]()
    trajectories, w_true, theta_true = generate_demos(env, args.n_traj, GAMMA, BETA, N_VI)
    Theta, w, _, _ = fit_tw_moirl(trajectories, 2, env["T"], env["phi"], GAMMA, BETA, N_VI,
                                  n_iters, e_steps, args.m_steps, verbose=True)

    row = {
        "env": args.env,
        "seed": args.seed,
        "e_steps": e_steps,
        "m_steps": args.m_steps,
        "n_iters": n_iters,
        "evd": evd(trajectories, theta_true, Theta, w, env, GAMMA, BETA, N_VI),
        "r_pair": r_pair(w[:, 0].numpy(), w_true),
        "auc_inv": auc_inv(w[:, 0].numpy(), w_true),
    }
    print(row)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
