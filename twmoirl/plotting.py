import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CONDITIONS = ["Control", "Stimulant", "Rescue"]


def plot_bases(Theta, env, path):
    phi_s = env["phi"][:, 0, :]
    fig, axes = plt.subplots(1, Theta.shape[0], figsize=(4 * Theta.shape[0], 4))
    for d, ax in enumerate(axes):
        r = torch.matmul(phi_s, Theta[d]).view(env["height"], env["width"]).numpy()
        im = ax.imshow(r, cmap="viridis", origin="upper", interpolation="bicubic")
        ax.set_title(f"Basis {d + 1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_preferences(values, labels, path):
    values, labels = np.asarray(values), np.asarray(labels)
    colors = plt.get_cmap("Set2").colors
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k, cond in enumerate(CONDITIONS):
        v = values[labels == cond]
        bp = ax.boxplot([v], positions=[k], widths=0.7, patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(colors[k])
        bp["boxes"][0].set_alpha(0.45)
        bp["medians"][0].set_color("black")
        ax.scatter(k + rng.uniform(-0.154, 0.154, size=v.size), v, s=32, facecolor=colors[k],
                   edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([f"{c}\n($n={np.sum(labels == c)}$)" for c in CONDITIONS])
    ax.set_ylabel(r"Inferred preference $\hat{w}_{i,1}$")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
