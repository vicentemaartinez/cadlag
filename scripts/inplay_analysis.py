"""E6: ajuste del modelo en vivo, prueba de contagio (Hawkes) y validación de probabilidades en vivo.

Uso: python scripts/inplay_analysis.py
Escribe outputs/analysis/inplay_*.csv / .json
"""
import json

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.data.events import load_events
from cadlag.leagues import CORE_LEAGUES
from cadlag.model import inplay
from cadlag.model.dixon_coles import outcome_probs
from cadlag.eval.scores import rps, log_score
from cadlag.model.filter import prepare
from cadlag.pipeline import load_or_fit

m, _ = store.load()
ev = load_events(None, m)
frames = []
for comp in CORE_LEAGUES:
    ld = prepare(m, comp, include_playoffs=False)
    mod = load_or_fit(ld)
    _, _, o = mod.run()
    d = ld.matches.assign(zh=o["zbar"][:, 0], za=o["zbar"][:, 1], comp=comp)
    frames.append(d[d["syear"] >= 2015][["id", "comp", "syear", "date", "hg", "ag", "zh", "za"]])
games = pd.concat(frames, ignore_index=True)
games = games[games["id"].isin(set(ev["id"]))]
# entrenamiento: hasta 2023; prueba de probabilidades en vivo: 2024 en adelante
train = games[games["syear"] <= 2023]
test = games[games["syear"] >= 2024]
print(f"{len(train)} partidos de entrenamiento, {len(test)} de prueba", flush=True)

seg = inplay.build_segments(train, ev)
print(f"{len(seg)} segmentos", flush=True)
coef, ll = inplay.fit_glm(seg)
coef0, ll0 = inplay.fit_glm(seg, feats=[f for f in inplay.FEATS if not f.startswith("after_")])
print(coef.round(4).to_string())
print(f"LR contagio multiplicativo (2 gl): {2 * (ll - ll0):.2f}", flush=True)

# Hawkes aditivo sobre una submuestra (el cálculo del kernel es O(segmentos x goles))
sub_ids = set(train.sample(4000, random_state=0)["id"])
seg_s = seg[seg["id"].isin(sub_ids)].reset_index(drop=True)
hk = inplay.hawkes_test(seg_s, coef0, ev[ev["id"].isin(sub_ids)])
print(hk.round(5).to_string(), flush=True)

# Validación: probabilidades 1X2 en vivo en el minuto 60 (estado real) frente a pre-partido
rows = []
evg = {k: g for k, g in ev[ev["side"].isin(["h", "a"]) & ev["t"].notna()].groupby("id")}
sample = test.sample(min(1500, len(test)), random_state=1)
for r in sample.itertuples():
    g = evg.get(r.id)
    sc, rd = [0, 0], [0, 0]
    if g is not None:
        for t, typ, side in zip(g["t"], g["type"], g["side"]):
            if inplay.tau_of(t) < 61:
                if typ in "GPO":
                    sc[0 if side == "h" else 1] += 1
                elif typ == "R":
                    rd[0 if side == "h" else 1] += 1
    P60 = inplay.live_distribution(r.zh, r.za, coef0, 61.0, tuple(sc), tuple(rd))
    P0 = inplay.live_distribution(r.zh, r.za, coef0, 0.0)
    o = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
    rows.append(dict(id=r.id, o=o, p60=outcome_probs(P60), p0=outcome_probs(P0)))
o = np.array([x["o"] for x in rows])
p60 = np.array([x["p60"] for x in rows]); p0 = np.array([x["p0"] for x in rows])
val = dict(n=len(o), rps_pre=float(rps(p0, o).mean()), rps_min60=float(rps(p60, o).mean()),
           log_pre=float(log_score(p0, o).mean()), log_min60=float(log_score(p60, o).mean()))
print(val)
out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
coef.to_csv(out / "inplay_coef_contagion.csv")
coef0.to_csv(out / "inplay_coef.csv")
hk.to_csv(out / "inplay_hawkes.csv", index=False)
json.dump(dict(lr_multiplicative=2 * (ll - ll0), validation=val, n_train=len(train), n_segments=len(seg)),
          open(out / "inplay_summary.json", "w"), indent=1, default=float)
