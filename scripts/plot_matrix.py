"""Matriz de posiciones estilo "Predicted Final League Position" (PNG) desde outputs/web/<liga>.json.

Uso: python scripts/plot_matrix.py chi.1 [eng.1 ...]
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cadlag.data.store import ROOT

SURFACE = "#fcfcfb"
INK = "#1d1d1b"
MUTED = "#6b6b66"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seq", RAMP)


def plot(comp: str):
    f = json.loads((ROOT / "outputs" / "web" / f"{comp}.json").read_text())
    teams = f["teams"]
    n = len(teams)
    M = np.array([t["position_probs"] for t in teams]) * 100
    fig_h = 0.38 * n + 1.9
    fig, ax = plt.subplots(figsize=(0.52 * n + 5.2, fig_h), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    left = 4.6
    for i, t in enumerate(teams):
        y = i
        ax.text(-left, y, t["name"], ha="left", va="center", fontsize=8.5, color=INK)
        ax.text(-1.55, y, str(t["played"]), ha="center", va="center", fontsize=8, color=MUTED)
        ax.text(-0.85, y, str(t["points"]), ha="center", va="center", fontsize=8, color=INK, weight="bold")
        ax.text(-0.15, y, f"{t['xpts']:.0f}", ha="center", va="center", fontsize=8, color=MUTED)
        for k in range(n):
            p = M[i, k]
            if p < 0.05:
                continue
            ax.add_patch(plt.Rectangle((k + 0.55, y - 0.44), 0.9, 0.88, color=cmap(min(p / 60, 1) ** 0.75), lw=0))
            if p >= 0.1:
                txt = f"{p:.1f}" if p < 99.95 else "100"
                ax.text(k + 1, y, txt, ha="center", va="center", fontsize=6.3,
                        color="white" if p > 30 else INK)
    for k in range(n):
        ax.text(k + 1, -0.95, str(k + 1), ha="center", va="center", fontsize=8, color=INK, weight="bold")
    for x, lab in ((-1.55, "PJ"), (-0.85, "PTS"), (-0.15, "xPTS")):
        ax.text(x, -0.95, lab, ha="center", va="center", fontsize=7, color=MUTED)
    for z in f["zones"]:
        if z["key"] in ("champion",):
            continue
    ax.set_xlim(-left - 0.1, n + 0.6)
    ax.set_ylim(n - 0.4, -1.5)
    ax.axis("off")
    title = f"{f['name']} {f['season_label']} · probabilidad de terminar en cada posición (%)"
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=11.5, color=INK, weight="bold")
    sub = (f"{f['matches_played']} de {f['matches_total']} partidos jugados · {f['n_sims']:,} simulaciones "
           f"de la predictiva posterior · cadlag").replace(",", ".")
    fig.text(0.012, 0.985 - 0.32 / fig_h, sub, ha="left", va="top", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig_h))
    out = ROOT / "outputs" / "web" / f"{comp}_matriz.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        plot(c)
