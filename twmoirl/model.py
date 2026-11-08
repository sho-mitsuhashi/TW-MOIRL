import torch
import torch.nn.functional as F

from .mdp import soft_value_iteration

# Bounds on variational log standard deviations for numerical stability.
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -6.0, 2.0


def sa_counts(trajectories, n_s, n_a):
    counts = torch.zeros(len(trajectories), n_s * n_a)
    for i, traj in enumerate(trajectories):
        sa = torch.tensor(traj, dtype=torch.long)
        counts[i] = torch.bincount(sa[:, 0] * n_a + sa[:, 1], minlength=n_s * n_a).float()
    return counts.view(-1, n_s, n_a)


def feature_covariance(counts, phi):
    n_f = phi.shape[-1]
    p = counts.sum(0).reshape(-1)
    p = p / p.sum().clamp_min(1.0)
    phi_sa = phi.reshape(-1, n_f)
    mean = (p.unsqueeze(1) * phi_sa).sum(dim=0)
    x = (phi_sa - mean.unsqueeze(0)) * torch.sqrt(p + 1e-12).unsqueeze(1)
    return x.t() @ x + 1e-8 * torch.eye(n_f)


def corr_penalty(Theta, cov):
    S = Theta @ cov @ Theta.t()
    d = torch.sqrt(torch.diag(S).clamp_min(1e-12))
    C = S / (d.unsqueeze(1) * d.unsqueeze(0) + 1e-12)
    off = C[~torch.eye(C.shape[0], dtype=torch.bool)]
    return (off ** 2).mean()


def kl_to_prior(mu, log_sigma, prior_log_sigma):
    log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    var = torch.exp(2.0 * log_sigma)
    prior_var = torch.exp(2.0 * prior_log_sigma)
    return (prior_log_sigma - log_sigma).sum(1) + 0.5 * ((var + mu ** 2) / prior_var).sum(1) - 0.5 * mu.shape[1]


def fit_tw_moirl(trajectories, D, T, phi, gamma, beta=0.5, n_vi=150, n_iters=100,
                 e_steps=20, m_steps=100, lr_q=0.05, lr_theta=0.01, n_mc=3,
                 prior_sigma=1.0, lam_kl=0.01, lam_corr=0.0, verbose=False):
    n_s, n_a, n_f = phi.shape
    N = len(trajectories)
    T_flat = T.reshape(n_s * n_a, n_s).t().contiguous()
    lengths = torch.tensor([len(t) for t in trajectories], dtype=torch.float32).clamp_min(1.0)
    counts = sa_counts(trajectories, n_s, n_a)
    cov = feature_covariance(counts, phi) if lam_corr > 0 and D > 1 else None

    Theta = torch.randn(D, n_f, requires_grad=True)
    mu = torch.zeros(N, D, requires_grad=True)
    log_sigma = torch.zeros(N, D, requires_grad=True)
    prior_log_sigma = torch.log(torch.ones(D) * float(prior_sigma))
    opt_q = torch.optim.Adam([mu, log_sigma], lr=lr_q)
    opt_theta = torch.optim.Adam([Theta], lr=lr_theta)

    def objective(Theta):
        eps = torch.randn(n_mc, N, D)
        u = mu.unsqueeze(0) + torch.exp(log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)).unsqueeze(0) * eps
        w = F.softmax(u, dim=-1).reshape(n_mc * N, D)
        r = torch.einsum("bd,df,saf->bsa", w, Theta, phi)
        q = soft_value_iteration(r, T, gamma, beta, n_vi, T_flat)
        log_pi = F.log_softmax(beta * q, dim=2).reshape(n_mc, N, n_s, n_a)
        ll = ((log_pi * counts.unsqueeze(0)).sum(dim=(2, 3)) / lengths.unsqueeze(0)).mean()
        kl = kl_to_prior(mu, log_sigma, prior_log_sigma).mean()
        obj = ll - lam_kl * kl
        if cov is not None:
            obj = obj - lam_corr * corr_penalty(Theta, cov)
        return obj, ll, kl

    for it in range(n_iters):
        for _ in range(e_steps):
            opt_q.zero_grad()
            (-objective(Theta.detach())[0]).backward()
            opt_q.step()
        for _ in range(m_steps):
            opt_theta.zero_grad()
            (-objective(Theta)[0]).backward()
            opt_theta.step()
        if verbose:
            with torch.no_grad():
                obj, ll, kl = objective(Theta)
            print(f"iter {it + 1:3d}  objective {obj.item():.4f}  loglik {ll.item():.4f}  kl {kl.item():.4f}")

    return Theta.detach(), F.softmax(mu.detach(), dim=1), mu.detach(), log_sigma.detach()
