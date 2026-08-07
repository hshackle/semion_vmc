"""Produce the comparison figures (analogues of Fig. 6 and Fig. 8 of
arXiv:2206.14730) from the saved results.

Usage: python -m scripts.make_figures [lss] [quench]
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SECTORS = [(0, 0), (0, 1), (1, 2)]
SECTOR_LABELS = [r"$(k_x,k_y)=(0,0)$", r"$(k_x,k_y)=(0,2\pi/L)$",
                 r"$(k_x,k_y)=(2\pi/L,4\pi/L)$"]


def P_poisson(r):
    return 2.0 / (1.0 + r) ** 2


def P_goe(r):
    return 27.0 / 4.0 * (r + r * r) / (1 + r + r * r) ** 2.5


def P_gue(r):
    return 81.0 * np.sqrt(3.0) / (2 * np.pi) * (r + r * r) ** 2 / (1 + r + r * r) ** 4


def r_ratios(ev):
    d = np.diff(np.sort(ev))
    r = np.minimum(d[1:], d[:-1]) / np.maximum(d[1:], d[:-1])
    return r[np.isfinite(r)]


def fig_lss():
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 9.0), sharex=True)
    rr = np.linspace(1e-4, 1, 400)
    bins = np.linspace(0, 1, 21)
    centers = 0.5 * (bins[1:] + bins[:-1])
    mean_r = {}
    for ax, (jx, jy), lab in zip(axes, SECTORS, SECTOR_LABELS):
        ax.plot(rr, P_poisson(rr), "-", color="tab:blue", label="Poi")
        ax.plot(rr, P_goe(rr), "-", color="tab:orange", label="GOE")
        ax.plot(rr, P_gue(rr), "-", color="tab:green", label="GUE")
        for which, marker, color, mlab in (
                ("hcb", "o", "tab:red", "HCB"),
                ("semion", "^", "tab:purple", "Sem")):
            f = f"results/lss_{which}_k{jx}{jy}.npy"
            if not os.path.exists(f):
                continue
            ev = np.load(f)
            r = r_ratios(ev)
            hist, _ = np.histogram(r, bins=bins, density=True)
            ax.plot(centers, hist, marker, mfc="none", color=color, label=mlab)
            mean_r[(which, jx, jy)] = r.mean()
        ax.set_ylabel(r"$P(r)$")
        ax.set_ylim(0, 2.05)
        ax.text(0.98, 1.92, lab, ha="right", va="top")
    axes[0].legend(ncol=2, fontsize=8, loc="center right",
                   bbox_to_anchor=(1.0, 0.35))
    axes[-1].set_xlabel(r"$r$")
    axes[-1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig("results/fig_lss.png", dpi=200)
    print("wrote results/fig_lss.png")
    # summary: mean r values (Poisson: 0.38629, GOE: 0.5307, GUE: 0.5996)
    for k, v in mean_r.items():
        print(f"  mean r {k}: {v:.4f}")


def fig_quench():
    labels = {"fermion": "Fermions", "hcb": "HCBs", "semion": "Semions"}
    avail = [w for w in ("fermion", "hcb", "semion")
             if os.path.exists(f"results/quench_{w}.npz")]
    fig, axes = plt.subplots(1, len(avail), figsize=(3.2 * len(avail), 4.2),
                             sharey=True)
    if len(avail) == 1:
        axes = [axes]
    im = None
    for ax, w in zip(axes, avail):
        d = np.load(f"results/quench_{w}.npz")
        t, n = d["t"], d["n"]
        im = ax.pcolormesh(np.arange(n.shape[1]), t, n, cmap="inferno",
                           vmin=0, vmax=1, shading="nearest")
        ax.set_title(labels[w])
        ax.set_xlabel(r"$i$")
    axes[0].set_ylabel(r"$t$")
    fig.colorbar(im, ax=axes, label=r"$\langle n_i(t)\rangle$")
    fig.savefig("results/fig_quench.png", dpi=200)
    print("wrote results/fig_quench.png")


if __name__ == "__main__":
    args = sys.argv[1:] or ["lss", "quench"]
    if "lss" in args:
        fig_lss()
    if "quench" in args:
        fig_quench()
