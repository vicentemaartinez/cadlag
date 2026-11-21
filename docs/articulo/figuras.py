"""Figuras del documento técnico (docs/articulo/fig). Uso: python docs/articulo/figuras.py

Todas salen de outputs/ (análisis y pronósticos ya calculados); ninguna vuelve a ajustar nada.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = Path(__file__).resolve().parent / "fig"
FIG.mkdir(exist_ok=True)

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyreheros-*.otf"):
    font_manager.fontManager.addfont(str(f))
BLUE, ORANGE, PURPLE, GRAY, INK, RED = "#2F6BD0", "#E8673A", "#7B5CC4", "#9AA0AA", "#2A2D35", "#8C2B2A"
mpl.rcParams.update({
    "font.family": "TeX Gyre Heros", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "axes.edgecolor": "#C9CED8", "axes.linewidth": 0.6, "axes.facecolor": "#FAFAFB",
    "axes.grid": True, "grid.color": "#E3E6EC", "grid.linewidth": 0.6, "xtick.color": "#555B66",
    "ytick.color": "#555B66", "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlelocation": "left", "axes.formatter.use_locale": False,
})
NAMES = {"eng.1": "Premier", "esp.1": "LaLiga", "ita.1": "Serie A", "ger.1": "Bundesliga", "fra.1": "Ligue 1",
         "chi.1": "Chile", "bra.1": "Brasileirão", "arg.1": "Argentina", "eng.2": "Championship",
         "esp.2": "LaLiga 2", "ita.2": "Serie B", "ger.2": "2. Bundesliga", "fra.2": "Ligue 2", "bra.2": "Série B"}


def comma(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(name)


# 1. Matriz de posiciones de la Premier ------------------------------------------------------
def matriz():
    d = json.load(open(OUT / "web" / "eng.1.json"))
    T = sorted(d["teams"], key=lambda t: t["expected_position"])
    M = np.array([t["position_probs"] for t in T])
    n = len(T)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("heat", ["#FBF3EE", "#E9A98F", "#C0533F", "#7A1F1E"])
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.grid(False)
    ax.imshow(np.sqrt(M), cmap=cmap, aspect="auto", vmin=0, vmax=np.sqrt(0.6))
    for i in range(n):
        for j in range(n):
            if M[i, j] >= 0.005:
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.2,
                        color="white" if M[i, j] > 0.2 else INK)
    ax.set_yticks(range(n), [f"{t['name']}  {t['points']}" for t in T], fontsize=6.5)
    ax.set_xticks(range(n), [str(k + 1) for k in range(n)], fontsize=6.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for x in (3.5, 16.5):
        ax.axvline(x, color="white", lw=1.6)
    ax.set_title(f"Premier League 2026-27 tras {d['matches_played']} de {d['matches_total']} partidos: "
                 "P(posición final), en %", pad=18)
    save(fig, "matriz_premier")


# 2. Plug-in frente a predictiva posterior -----------------------------------------------------
def varianza():
    rows = []
    for L in ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"]:
        s = json.load(open(OUT / "backtest" / f"{L}_summary.json"))["seasons_summary"]
        for r in s:
            rows.append(dict(liga=L, frac=r["frac"], modo=r["mode"], z2=r["z2_pts"], cov=r["cover90_pts"], log=r["log_pos"]))
    D = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for modo, c, lab in [("plugin", ORANGE, "plug-in (fuerzas fijas en su media)"), ("posterior", BLUE, "predictiva posterior")]:
        g = D[D.modo == modo].groupby("frac")
        for L, dd in D[D.modo == modo].groupby("liga"):
            axs[0].plot(dd.frac * 100, dd.z2, color=c, lw=0.6, alpha=0.35)
            axs[1].plot(dd.frac * 100, dd["cov"] * 100, color=c, lw=0.6, alpha=0.35)
        axs[0].plot(g.z2.mean().index * 100, g.z2.mean(), color=c, lw=2, marker="o", ms=3.5, label=lab)
        axs[1].plot(g["cov"].mean().index * 100, g["cov"].mean() * 100, color=c, lw=2, marker="o", ms=3.5)
    axs[0].axhline(1, color=INK, lw=0.8, ls="--")
    axs[1].axhline(90, color=INK, lw=0.8, ls="--")
    axs[0].set(title="Error cuadrático estandarizado de los puntos", xlabel="% de la temporada jugada", ylabel=r"$E[z^2]$ (ideal 1)")
    axs[1].set(title="Cobertura del intervalo de 90 %", xlabel="% de la temporada jugada", ylabel="%")
    axs[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "varianza")


# 3. Dos latentes frente al núcleo: partidos y temporadas -----------------------------------
def comparacion():
    # partidos (R13, R17, R19): log-verosimilitud del marcador por partido, desde 2016 (Chile 2020)
    part = [("Premier", 0.0093, 0.0023), ("Chile", 0.0042, 0.0017), ("Ligue 1", 0.0052, 0.0024),
            ("Brasileirão", 0.0025, 0.0015), ("Serie A", 0.0016, 0.0023), ("LaLiga", 0.0004, 0.0015),
            ("Bundesliga", -0.0017, 0.0024), ("Championship", 0.0075, 0.0015), ("Argentina", 0.0035, 0.0019)]
    w = np.array([1 / s ** 2 for _, _, s in part]); m = np.array([x for _, x, _ in part])
    pooled = (float((w * m).sum() / w.sum()), float(1 / np.sqrt(w.sum())))
    sc = json.load(open(OUT / "analysis" / "season_compare.json"))
    order = ["eng.1", "chi.1", "fra.1", "bra.1", "ita.1", "esp.1", "ger.1", "eng.2", "arg.1"]
    temp = [(NAMES[k], sc[k]["delta_log_pos"], sc[k]["se"]) for k in order]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, data, tot, title, unit in [
        (axs[0], part, pooled, "Partidos: log-verosimilitud del marcador", "nats por partido"),
        (axs[1], temp, (sc["todas"]["delta_log_pos"], sc["todas"]["se"]), "Temporadas: −log P(posición final)", "nats por equipo y momento")]:
        ys = np.arange(len(data))[::-1] + 1.5
        for y, (nm, x, s) in zip(ys, data):
            ax.errorbar(x, y, xerr=2 * s, fmt="o", color=BLUE, ms=3.5, lw=1, capsize=0)
        ax.errorbar(tot[0], 0.3, xerr=2 * tot[1], fmt="D", color=PURPLE, ms=4.5, lw=1.6)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(list(ys) + [0.3], [d[0] for d in data] + ["Combinado"])
        ax.set_title(title, fontsize=8.3)
        ax.set_xlabel(f"ganancia de dos latentes ({unit})")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "dos_latentes")
    return pooled


# 4. Offsets entre ligas en el tiempo -----------------------------------------------------------
def ligas():
    o = pd.read_csv(OUT / "analysis" / "crossleague_offsets.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    cols = {"esp.1": BLUE, "ger.1": "#5A9BD8", "ita.1": "#86B3E8", "fra.1": "#A9C7EE", "bra.1": ORANGE,
            "arg.1": "#F0A06B", "chi.1": RED, "eng.2": GRAY}
    for L, c in cols.items():
        d = o[o.league == L].sort_values("season")
        ax.plot(d.season, d.net, color=c, lw=1.6, label=NAMES[L])
        if L in ("bra.1", "chi.1", "esp.1"):
            ax.fill_between(d.season, d.net - d.net_sd, d.net + d.net_sd, color=c, alpha=0.13, lw=0)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    ax.text(2009.2, 0.03, "Premier (referencia)", fontsize=7, color=INK)
    ax.set(xlabel="temporada", ylabel="fuerza neta frente a la Premier", title="Offsets de liga (bandas: ±1 desviación posterior)")
    ax.legend(ncol=4, loc="lower left", fontsize=7)
    ax.set_xticks(range(2010, 2027, 2))
    save(fig, "offsets_ligas")


# 5. Suspenso: entropía del campeón ------------------------------------------------------------
def suspenso():
    d = json.load(open(OUT / "analysis" / "suspense_curves.json"))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    cols = {"bra.1": ORANGE, "chi.1": RED, "eng.1": BLUE, "esp.1": "#5A9BD8", "ita.1": "#86B3E8",
            "fra.1": PURPLE, "ger.1": GRAY}
    for L, c in cols.items():
        E = np.array([v["entropy"] for k, v in d.items() if k.startswith(L + ":")])
        x = np.linspace(0, 100, E.shape[1])
        ax.plot(x, E.mean(0), color=c, lw=1.8, label=NAMES[L])
    ax.set(xlabel="% de la temporada jugada", ylabel="bits", title="Entropía del campeón a lo largo de la temporada (promedio 2016-2025)")
    ax.legend(ncol=4)
    save(fig, "suspenso")


# 6. Modelo en vivo -----------------------------------------------------------------------------
def envivo():
    c = pd.read_csv(OUT / "analysis" / "inplay_coef_contagion.csv", index_col=0)
    keys = [("red_own", "roja propia"), ("red_opp", "roja del rival"), ("trail1", "perdiendo por 1"),
            ("trail2", "perdiendo por 2+"), ("lead1", "ganando por 1"), ("lead2", "ganando por 2+"),
            ("after_own_goal", "10' tras marcar"), ("after_conceded", "10' tras recibir")]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ys = np.arange(len(keys))[::-1]
    for y, (k, lab) in zip(ys, keys):
        b, s = c.loc[k, "coef"], c.loc[k, "se"]
        col = ORANGE if k.startswith("after") else BLUE
        ax.errorbar(np.exp(b), y, xerr=[[np.exp(b) - np.exp(b - 2 * s)], [np.exp(b + 2 * s) - np.exp(b)]],
                    fmt="o", color=col, ms=4, lw=1.2)
        ax.text(np.exp(b), y + 0.28, f"×{comma(np.exp(b))}", ha="center", fontsize=6.8, color=INK)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1, 1.25, 1.5], ["0,6", "0,8", "1", "1,25", "1,5"])
    ax.set_yticks(ys, [k[1] for k in keys])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="multiplicador de la tasa de goles del equipo (±2 e.e.)", title="Efectos de estado en vivo (21.730 partidos)")
    save(fig, "envivo")


# 7. Offsets continentales 2026 -----------------------------------------------------------------
def continentales():
    eu = [("LaLiga", -0.27), ("Bundesliga", -0.31), ("Serie A", -0.37), ("Ligue 1", -0.50), ("Portugal", -0.92),
          ("Bélgica", -0.95), ("Países Bajos", -0.99), ("Turquía", -1.03), ("Dinamarca", -1.11), ("Suiza", -1.16),
          ("Noruega", -1.22), ("Grecia", -1.22), ("Austria", -1.27), ("Suecia", -1.33), ("Escocia", -1.44)]
    sa = [("Argentina", -0.20), ("Ecuador", -0.56), ("Colombia", -0.59), ("Paraguay", -0.72), ("Uruguay", -0.78),
          ("Chile", -0.81), ("Bolivia", -1.16), ("Perú", -1.19), ("Venezuela", -1.37)]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw=dict(width_ratios=[1.45, 1]))
    for ax, data, c, ref in [(axs[0], eu, BLUE, "Europa (frente a la Premier)"), (axs[1], sa, ORANGE, "Sudamérica (frente al Brasileirão)")]:
        ys = np.arange(len(data))[::-1]
        ax.barh(ys, [x for _, x in data], color=c, height=0.62)
        for y, (_, x) in zip(ys, data):
            ax.text(x - 0.02, y, comma(x), ha="right", va="center", fontsize=6.6, color=INK)
        ax.set_yticks(ys, [n for n, _ in data])
        ax.set_xlim(-1.75, 0)
        ax.set_title(ref, fontsize=8.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("offset neto de la liga, 2026 (log-goles)")
    fig.tight_layout()
    save(fig, "offsets_copas")


# 8. Calibración del ascenso en la Championship --------------------------------------------------
def ascenso():
    p = [0.09, 0.22, 0.37, 0.60, 0.87]; f = [0.07, 0.21, 0.43, 0.67, 0.89]
    fig, ax = plt.subplots(figsize=(2.6, 2.5))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls="--")
    ax.plot(p, f, "o-", color=BLUE, ms=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="P(ascenso) pronosticada", ylabel="frecuencia observada",
           title="Championship 2016-2025")
    save(fig, "ascenso")


# 9. Trayectoria del campeón en Chile 2026 -------------------------------------------------------
def trayectoria():
    tr = json.load(open(OUT / "web" / "trajectory_chi.1.json"))
    P = np.array([p["p"] for p in tr["points"]]); x = np.array([p["played"] for p in tr["points"]])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    cols = [RED, BLUE, "#5A9BD8", ORANGE, PURPLE, GRAY]
    for k, t in enumerate(tr["teams"]):
        ax.step(x, P[:, k] * 100, where="post", color=cols[k % len(cols)], lw=1.5, label=t["name"])
    ax.set(xlabel=f"partidos jugados (de {tr['total']})", ylabel="P(campeón), %",
           title="Chile 2026: probabilidad de salir campeón después de cada partido")
    ax.legend(ncol=2, loc="center left")
    save(fig, "trayectoria_chile")


# 10. "¿Qué necesita?" -------------------------------------------------------------------------
def necesita():
    d = json.load(open(OUT / "web" / "eng.1.json"))
    T = {t["name"]: t for t in d["teams"]}
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for nm, c in [("Manchester City", BLUE), ("Arsenal", ORANGE), ("Liverpool", PURPLE)]:
        if nm in T and "title" in T[nm]["needs"]:
            cv = np.array(T[nm]["needs"]["title"]["curve"])
            axs[0].plot(cv[:, 0], cv[:, 1] * 100, color=c, lw=1.6, label=nm)
    for nm, c in [("Crystal Palace", GRAY), ("Ipswich Town", RED), ("Coventry City", "#5A9BD8")]:
        if nm in T and "survival" in T[nm]["needs"]:
            cv = np.array(T[nm]["needs"]["survival"]["curve"])
            axs[1].plot(cv[:, 0], cv[:, 1] * 100, color=c, lw=1.6, label=nm)
    axs[0].set(title="Título", xlabel="puntos que suma desde hoy", ylabel="P(objetivo | puntos), %", xlim=(30, 85))
    axs[1].set(title="Salvarse", xlabel="puntos que suma desde hoy", xlim=(0, 50))
    for ax in axs:
        ax.axhline(50, color=GRAY, lw=0.6, ls=":"); ax.axhline(90, color=GRAY, lw=0.6, ls=":")
        ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "necesita")


# 11. Filtro de Laplace frente a NUTS -----------------------------------------------------------
def nuts():
    d = json.load(open(OUT / "analysis" / "nuts_validation_eng.1_2025.json"))
    T = pd.DataFrame(d["teams"])
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
    lo, hi = T[["nuts_mean", "filter_mean"]].min().min() - 0.05, T[["nuts_mean", "filter_mean"]].max().max() + 0.05
    axs[0].plot([lo, hi], [lo, hi], color=GRAY, lw=0.8, ls="--")
    axs[0].errorbar(T.nuts_mean, T.filter_mean, xerr=T.nuts_sd, yerr=T.filter_sd, fmt="o", color=BLUE, ms=3.5,
                    lw=0.5, alpha=0.85, ecolor="#B9CCEE")
    axs[0].set(xlabel="NUTS (exacto)", ylabel="filtro de Laplace", title="Fuerza neta: media ± desviación")
    axs[1].plot([0.12, 0.18], [0.12, 0.18], color=GRAY, lw=0.8, ls="--")
    axs[1].plot(T.nuts_sd, T.filter_sd, "o", color=PURPLE, ms=4)
    axs[1].set(xlabel="NUTS", ylabel="filtro de Laplace", title="Desviación posterior")
    fig.tight_layout()
    save(fig, "nuts")


if __name__ == "__main__":
    nuts()
    matriz(); varianza(); print("combinado", comparacion()); ligas(); suspenso(); envivo(); continentales()
    ascenso(); trayectoria(); necesita()
