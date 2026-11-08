import torch


def soft_value_iteration(r, T, gamma, beta=0.5, n_iters=150, T_flat=None):
    single = r.dim() == 2
    if single:
        r = r.unsqueeze(0)
    n_s, n_a = r.shape[1], r.shape[2]
    if T_flat is None:
        T_flat = T.reshape(n_s * n_a, n_s).t().contiguous()
    q = torch.zeros_like(r)
    for _ in range(n_iters):
        v = torch.logsumexp(beta * q, dim=2) * (1.0 / beta)
        q.copy_(r).add_((v @ T_flat).reshape(-1, n_s, n_a), alpha=gamma)
    return q[0] if single else q


def boltzmann_policy(r, T, gamma, beta=0.5, n_iters=150):
    return torch.softmax(beta * soft_value_iteration(r, T, gamma, beta, n_iters), dim=1)


def policy_evaluation(pi, r, T, gamma, n_iters=100):
    q = torch.zeros_like(r)
    v = torch.zeros(r.shape[0], dtype=r.dtype)
    for _ in range(n_iters):
        v = (pi * q).sum(dim=1)
        q = r + gamma * torch.matmul(T, v)
    return v
