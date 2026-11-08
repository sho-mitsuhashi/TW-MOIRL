import numpy as np
import torch

from .mdp import boltzmann_policy, policy_evaluation


def evd(trajectories, theta_true, Theta, w, env, gamma, beta=0.5, n_vi=150):
    T, phi = env["T"], env["phi"]
    values = []
    for i, traj in enumerate(trajectories):
        r_true = torch.matmul(phi, theta_true[i])
        r_est = torch.matmul(phi, torch.einsum("d,df->f", w[i], Theta))
        v_true = policy_evaluation(boltzmann_policy(r_true, T, gamma, beta, n_vi), r_true, T, gamma)
        v_est = policy_evaluation(boltzmann_policy(r_est, T, gamma, beta, n_vi), r_true, T, gamma)
        s0 = traj[0][0]
        values.append(max(0.0, v_true[s0].item() - v_est[s0].item()))
    return float(np.mean(values))


def r_pair(w_hat, w_true):
    w_hat, w_true = np.asarray(w_hat, dtype=float), np.asarray(w_true, dtype=float)
    i, j = np.triu_indices(len(w_hat), k=1)
    d_hat, d_true = np.abs(w_hat[i] - w_hat[j]), np.abs(w_true[i] - w_true[j])
    if d_hat.std() < 1e-12 or d_true.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(d_hat, d_true)[0, 1])


def auc_inv(w_hat, w_true):
    w_hat, w_true = np.asarray(w_hat, dtype=float), np.asarray(w_true, dtype=float)
    upper = w_true > np.median(w_true)
    diff = w_hat[upper][:, None] - w_hat[~upper][None, :]
    auc = float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))
    return max(auc, 1.0 - auc)
