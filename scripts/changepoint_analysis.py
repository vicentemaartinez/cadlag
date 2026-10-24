"""E7: cambios de régimen en los residuos de cada equipo y ¿el filtro llega tarde?

Para cada equipo (ligas núcleo, desde 2015) se corre BOCPD sobre e_k = GD_k - E[GD_k] y se
marca un cambio en el partido k si P(r_{k+5} <= 5) > 0,5 (detección con 5 partidos de retraso).
Luego se mide el residuo medio del filtro en los 10 partidos siguientes a la detección: si el
filtro absorbiera bien los cambios, debería ser ~0.

Uso: python scripts/changepoint_analysis.py
"""
import json

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.leagues import CORE_LEAGUES
from cadlag.model.changepoint import bocpd
from cadlag.model.filter import prepare
from cadlag.pipeline import load_or_fit

m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
DELAY, AFTER = 5, 10
det, base = [], []
for comp in CORE_LEAGUES:
    ld = prepare(m, comp, include_playoffs=False)
    mod = load_or_fit(ld)
    _, _, o = mod.run()
    d = ld.matches.assign(eg=np.exp(o["zbar"][:, 0]) - np.exp(o["zbar"][:, 1]))
    d = d[d["syear"] >= 2015]
    long = pd.concat([
        pd.DataFrame(dict(team=d["home"], date=d["date"], e=(d["hg"] - d["ag"]) - d["eg"])),
        pd.DataFrame(dict(team=d["away"], date=d["date"], e=(d["ag"] - d["hg"]) + d["eg"]))]).sort_values("date")
    for t, g in long.groupby("team"):
        x = g["e"].to_numpy()
        if len(x) < 40:
            continue
        r = bocpd(x, L=60.0, kappa0=1.0, alpha0=3.0, beta0=3.0 * x.var())
        R = r["run_posterior"]
        dates = g["date"].to_numpy()
        flagged = []
        for k in range(len(x) - DELAY):
            p = R[k + DELAY, :DELAY + 1].sum()
            if p > 0.5 and (not flagged or k - flagged[-1] > 20):
                flagged.append(k)
                fut = x[k + DELAY + 1:k + DELAY + 1 + AFTER]
                past = x[max(0, k - 20):k]
                det.append(dict(comp=comp, team=t, name=names.get(t), date=str(dates[k])[:10], p=p,
                                past_mean=past.mean() if len(past) else np.nan,
                                after_mean=fut.mean() if len(fut) else np.nan, n_after=len(fut)))
        base.append(x)
det = pd.DataFrame(det)
allres = np.concatenate(base)
out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
det.to_csv(out / "changepoints.csv", index=False)
# después de una detección, ¿el residuo sigue teniendo el signo del cambio?
d2 = det.dropna(subset=["after_mean"])
sign = np.sign(d2["after_mean"])  # dirección del nuevo régimen según lo que viene
lag = d2["after_mean"].abs().mean()
summary = dict(n_detections=len(det), teams=int(det["team"].nunique()) if len(det) else 0,
               residual_sd=float(allres.std()), mean_abs_after=float(lag),
               mean_after=float(d2["after_mean"].mean()),
               se_after=float(d2["after_mean"].std() / np.sqrt(len(d2))) if len(d2) > 1 else None,
               corr_past_after=float(np.corrcoef(d2["past_mean"].fillna(0), d2["after_mean"])[0, 1]) if len(d2) > 2 else None)
json.dump(summary, open(out / "changepoints_summary.json", "w"), indent=1, default=float)
print(summary)
print(det.sort_values("p", ascending=False).head(25).round(3).to_string())
