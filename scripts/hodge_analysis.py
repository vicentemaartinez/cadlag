"""Intransitividad en cada liga-temporada: proporción cíclica (curl) observada vs. nulo del modelo.

Para cada temporada completa: se calcula la descomposición de Hodge del flujo de diferencias
de gol, y se repite sobre B temporadas simuladas del modelo (mismos partidos, goles sorteados
de un modelo de fuerzas puras ajustado a esa misma temporada: Poisson log-lineal con ataque,
defensa y localía, por máxima verosimilitud). Es un bootstrap paramétrico: el nulo es "todo lo
que se ve se explica con fuerzas transitivas de esta temporada". (Una versión anterior usaba las
predicciones un paso adelante del filtro como nulo; eso subestima la dispersión real de fuerzas
y hace que la realidad parezca *más* transitiva que el nulo: un nulo mal calibrado.)
El p-valor es la fracción de réplicas nulas con proporción cíclica mayor o igual a la observada.

Uso: python scripts/hodge_analysis.py [B]
"""
import sys

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.leagues import CORE_LEAGUES
from cadlag.model.filter import prepare
from cadlag.model.hodge import build_flows, decompose
from cadlag.pipeline import load_or_fit
from cadlag.sim.season import sample_dc

def static_poisson(h, a, x, y, n, iters=50, ridge=1e-3):
    """MLE de log lam = mu + eta + att_h - def_a, log nu = mu + att_a - def_h (suma cero vía ridge)."""
    P = 2 * n + 2
    Xh = np.zeros((len(h), P)); Xa = np.zeros((len(h), P))
    r = np.arange(len(h))
    Xh[r, h] = 1; Xh[r, n + a] = -1; Xh[:, 2 * n] = 1; Xh[:, 2 * n + 1] = 1
    Xa[r, a] = 1; Xa[r, n + h] = -1; Xa[:, 2 * n] = 1
    th = np.zeros(P); th[2 * n] = 0.3
    for _ in range(iters):
        lh, la = np.exp(Xh @ th), np.exp(Xa @ th)
        g = Xh.T @ (x - lh) + Xa.T @ (y - la) - ridge * th
        H = Xh.T @ (Xh * lh[:, None]) + Xa.T @ (Xa * la[:, None]) + ridge * np.eye(P)
        st = np.linalg.solve(H, g); th += st
        if np.abs(st).max() < 1e-10:
            break
    return np.exp(Xh @ th), np.exp(Xa @ th)


B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
m, _ = store.load()
rng = np.random.default_rng(0)
rows = []
for comp in CORE_LEAGUES:
    ld = prepare(m, comp, include_playoffs=False)
    mod = load_or_fit(ld)
    _, _, o = mod.run()
    rho = mod.params()["rho"]
    d = ld.matches.assign(zh=o["zbar"][:, 0], za=o["zbar"][:, 1])
    for s, g in d.groupby("syear"):
        teams = sorted(set(g["hi"]) | set(g["ai"]))
        n = len(teams)
        if len(g) < 0.95 * n * (n - 1):
            continue  # temporada incompleta o con formato raro
        loc = {t: i for i, t in enumerate(teams)}
        h = g["hi"].map(loc).to_numpy()
        a = g["ai"].map(loc).to_numpy()
        e, Y, W = build_flows(h, a, (g["hg"] - g["ag"]).to_numpy(float))
        obs = decompose(n, e, Y, W)
        lam, nu = static_poisson(h, a, g["hg"].to_numpy(float), g["ag"].to_numpy(float), n)
        null = []
        for b in range(B):
            x, y = sample_dc(lam, nu, rho, rng)
            _, Yb, Wb = build_flows(h, a, (x - y).astype(float))
            null.append(decompose(n, e, Yb, Wb)["share_curl"])
        null = np.array(null)
        rows.append(dict(comp=comp, syear=int(s), n=n, share_curl=obs["share_curl"],
                         null_mean=null.mean(), null_q05=np.quantile(null, 0.05), null_q95=np.quantile(null, 0.95),
                         p_value=float((null >= obs["share_curl"]).mean())))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "hodge_seasons.csv", index=False)
# combinación de p-valores (Fisher) por liga y global
from scipy.stats import chi2
for comp, g in list(df.groupby("comp")) + [("TODAS", df)]:
    p = np.clip(g["p_value"].to_numpy(), 1 / (B + 1), 1)
    X = -2 * np.log(p).sum()
    print(f"{comp}: curl obs {g.share_curl.mean():.3f} vs nulo {g.null_mean.mean():.3f}; "
          f"Fisher p = {chi2.sf(X, 2 * len(p)):.3g} ({len(p)} temporadas)")
