"""¿Explican los estilos latentes algo que las fuerzas dinámicas no explican? Prueba fuera de muestra.

Para cada liga y temporada de prueba s: se entrena la interacción u_i^T v_j (MAP con ARD) sobre
los residuos del filtro en las temporadas s-3..s-1 y la primera mitad de s, y se evalúa la
log-verosimilitud del marcador en la segunda mitad de s, frente al filtro sin interacción.
Δ > 0 significa que los estilos mejoran el pronóstico. Se reporta la media, su error estándar
(Diebold–Mariano con partidos como unidades) y las escalas ARD.

Uso: python scripts/styles_analysis.py [K]
"""
import sys

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.leagues import CORE_LEAGUES
from cadlag.model import styles
from cadlag.model.filter import prepare
from cadlag.pipeline import load_or_fit

K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SCALES = (0.2, 0.5, 1.0)
m, _ = store.load()
rows, per_match, spec = [], [], []
for comp in CORE_LEAGUES:
    ld = prepare(m, comp, include_playoffs=False)
    mod = load_or_fit(ld)
    _, _, o = mod.run()
    rho = mod.params()["rho"]
    d = ld.matches.assign(zh=o["zbar"][:, 0], za=o["zbar"][:, 1])
    complete = [s for s, g in d.groupby("syear") if len(g) >= 0.95 * g["hi"].nunique() * (g["hi"].nunique() - 1)]
    for s in [x for x in complete if x >= 2015]:
        cur = d[d["syear"] == s].sort_values("date")
        half = len(cur) // 2
        train = pd.concat([d[(d["syear"] >= s - 3) & (d["syear"] < s)], cur.iloc[:half]])
        test = cur.iloc[half:]
        teams = sorted(set(train["hi"]) | set(train["ai"]) | set(test["hi"]) | set(test["ai"]))
        loc = {t: i for i, t in enumerate(teams)}
        f = lambda df: (df["hi"].map(loc).to_numpy(), df["ai"].map(loc).to_numpy(), df["zh"].to_numpy(),
                        df["za"].to_numpy(), df["hg"].to_numpy(float), df["ag"].to_numpy(float))
        hi, ai, zh, za, x, y = f(train)
        sv, null, pv = styles.spectral_test(hi, ai, x, y, np.exp(zh), np.exp(za), len(teams), rho, B=100)
        spec.append(dict(comp=comp, syear=int(s), sv1=sv[0], null_q95=float(np.quantile(null[:, 0], 0.95)),
                         p1=float(pv[0]), sv2=sv[1], p2=float(pv[1])))
        fits = {sc: styles.fit_map(hi, ai, zh, za, x, y, len(teams), K=K, scale=sc)[:2] for sc in SCALES}
        hi, ai, zh, za, x, y = f(test)
        l0 = styles.interaction_logscore(zh, za, x, y, None, None, hi, ai, rho)
        for sc, (U, V) in fits.items():
            dlt = l0 - styles.interaction_logscore(zh, za, x, y, U, V, hi, ai, rho)
            per_match.append(pd.DataFrame(dict(comp=comp, syear=s, scale=sc, delta=dlt)))
            rows.append(dict(comp=comp, syear=int(s), scale=sc, n_test=len(test), delta_mean=float(dlt.mean()),
                             gamma_norm=float(np.linalg.norm(U @ V.T))))
        print(rows[-3:], flush=True)
pm = pd.concat(per_match)
out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / f"styles_K{K}.csv", index=False)
sp = pd.DataFrame(spec)
sp.to_csv(out / "styles_spectral.csv", index=False)
from scipy.stats import chi2
for comp, g in list(sp.groupby("comp")) + [("TODAS", sp)]:
    pp = np.clip(g["p1"].to_numpy(), 0.01, 1)
    print(f"{comp}: sv1 obs {g.sv1.mean():.2f} vs nulo q95 {g.null_q95.mean():.2f}; Fisher p = {chi2.sf(-2 * np.log(pp).sum(), 2 * len(pp)):.3g}")
for (comp, sc), g in list(pm.groupby(["comp", "scale"])) + [(("TODAS", sc), g) for sc, g in pm.groupby("scale")]:
    se = g["delta"].std(ddof=1) / np.sqrt(len(g))
    print(f"{comp} s={sc}: Δ log-score = {g['delta'].mean() * 1000:+.2f} milinats/partido (EE {se * 1000:.2f}, t = {g['delta'].mean() / se:+.2f}, n = {len(g)})")
