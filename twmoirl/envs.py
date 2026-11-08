import math

import numpy as np
import torch

from .mdp import boltzmann_policy

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _transitions(width, height, moves, absorbing=None):
    n_s = width * height
    T = torch.zeros(n_s, len(moves), n_s)
    for s in range(n_s):
        if s == absorbing:
            T[s, :, s] = 1.0
            continue
        y, x = divmod(s, width)
        for a, (dy, dx) in enumerate(moves):
            ny, nx = y + dy, x + dx
            T[s, a, ny * width + nx if 0 <= ny < height and 0 <= nx < width else s] = 1.0
    return T


def _env(width, height, feat, starts, goal, horizon, bases, moves=MOVES):
    return {
        "width": width,
        "height": height,
        "T": _transitions(width, height, moves, goal),
        "phi": feat.unsqueeze(1).repeat(1, len(moves), 1),
        "starts": starts,
        "goal": goal,
        "horizon": horizon,
        "bases": bases,
    }


def goal_puddle():
    goal, puddle = 14, {7, 11, 12, 13, 17}
    feat = torch.zeros(25, 3)
    for s in range(25):
        feat[s, 0 if s == goal else 1 if s in puddle else 2] = 1.0
    bases = (torch.tensor([1.0, -9.0, 0.0]), torch.tensor([10.0, 0.0, 0.0]))
    return _env(5, 5, feat, [10], goal, 10, bases)


def trap_grid():
    goal = 44
    feat = torch.zeros(81, 10)
    for s in range(81):
        y, x = divmod(s, 9)
        feat[s, 0] = float(s == goal)
        feat[s, 1] = abs(y - 4.0) + abs(x - 4.0)
        feat[s, 2] = float(x % 2 == 1)
        feat[s, 3] = float(y % 2 == 1)
        feat[s, 4] = float(abs(y - 4) + abs(x - 8))
        feat[s, 5] = 1.0
        feat[s, 6] = float(x % 2 == 1 and y % 2 == 1)
        feat[s, 7] = float(2 < x < 6)
        feat[s, 8] = float(2 < y < 6)
        feat[s, 9] = float(2 < x < 6 and 2 < y < 6)
    bases = (
        torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0]),
    )
    return _env(9, 9, feat, [27, 36, 45], goal, 30, bases)


def xy_tradeoff():
    feat = torch.zeros(81, 3)
    for s in range(81):
        y, x = divmod(s, 9)
        feat[s, 0] = math.sqrt(x)
        feat[s, 1] = math.sqrt(y)
        feat[s, 2] = 1.0
    bases = (torch.tensor([10.0, 0.0, 0.0]), torch.tensor([0.0, 10.0, 0.0]))
    return _env(9, 9, feat, [0], None, 8, bases)


def mouse_grid(size=10, n_rbf=8, sigma=2.0):
    centers = [(cy, cx) for cy in np.linspace(0, size - 1, n_rbf) for cx in np.linspace(0, size - 1, n_rbf)]
    feat = torch.zeros(size * size, n_rbf * n_rbf + 1)
    for s in range(size * size):
        y, x = divmod(s, size)
        for i, (cy, cx) in enumerate(centers):
            feat[s, i] = math.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
        feat[s, -1] = 1.0
    return _env(size, size, feat, None, None, None, None, MOVES + [(0, 0)])


ENVS = {"GoalPuddle-5x5": goal_puddle, "TrapGrid-9x9": trap_grid, "XYTradeoff-9x9": xy_tradeoff}


def rollout(theta, env, gamma, beta, n_vi):
    T = env["T"]
    pi = boltzmann_policy(torch.matmul(env["phi"], theta), T, gamma, beta, n_vi)
    s = env["starts"][torch.randint(len(env["starts"]), (1,)).item()]
    traj = []
    for _ in range(env["horizon"]):
        if s == env["goal"]:
            break
        a = torch.multinomial(pi[s], 1).item()
        traj.append((s, a))
        s = torch.multinomial(T[s, a], 1).item()
    return traj


def generate_demos(env, n, gamma, beta=0.5, n_vi=150):
    theta_1, theta_2 = env["bases"]
    trajectories, w_true, theta_true = [], [], []
    for _ in range(n):
        w = torch.rand(1).item()
        theta = w * theta_1 + (1.0 - w) * theta_2
        trajectories.append(rollout(theta, env, gamma, beta, n_vi))
        w_true.append(w)
        theta_true.append(theta)
    return trajectories, w_true, theta_true
