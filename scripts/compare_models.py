"""Experimento: pronóstico de la temporada actual con el núcleo y con el de dos latentes, en todas
las ligas, sin tocar lo que corre en producción.

Para el de dos latentes usa los hiperparámetros de producción si existen (models/<liga>_shots2_h.npy)
y si no los del backtest (ajustados con temporadas anteriores a 2016); con --ambos corre las dos
versiones donde existen ambas, para ver cuánto pesa esa diferencia.

Uso: python scripts/compare_models.py [--ambos]
Escribe outputs/analysis/model_comparison.csv
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.leagues import CORE_LEAGUES, LEAGUES
from cadlag.model import engine
from cadlag.pipeline import forecast_league

both = "--ambos" in sys.argv
m, teams = store.load()
bt = store.ROOT / "outputs" / "backtest"
tmp = Path(tempfile.mkdtemp())
for f in engine.MODELS.glob("*_h.npy"):
    shutil.copy(f, tmp / f.name)
engine.MODELS = tmp  # el experimento nunca escribe en models/

rows = []
for comp in CORE_LEAGUES:
    prod = store.ROOT / "models" / f"{comp}_shots2_h.npy"
    back = sorted(bt.glob(f"{comp}_shots2_h_*.npy"))
    variants = [("dc", None)]
    if prod.exists():
        variants.append(("shots2", prod))
    if back and (both or not prod.exists()):
        variants.append(("shots2_2016" if prod.exists() else "shots2", back[0]))
    zones = {z.key: z.positions for z in LEAGUES[comp].zones}
    for name, h in variants:
        kind = "dc" if name == "dc" else "shots2"
        if h is not None:
            shutil.copy(h, tmp / f"{comp}_shots2_h.npy")
        engine.kind_for = lambda c, k=kind: k
        f = forecast_league(comp, m, teams, n_sims=20000, seed=0)
        for t in f["teams"]:
            p = np.array(t["position_probs"])
            rows.append(dict(liga=comp, variante=name, hiper=("produccion" if h == prod else "backtest") if h else "-",
                             equipo=t["name"], titulo=p[0],
                             descenso=float(sum(p[k - 1] for k in zones.get("relegation", []))),
                             pos_esp=float((p * np.arange(1, len(p) + 1)).sum())))
        print(comp, name, "listo", flush=True)
out = store.ROOT / "outputs" / "analysis" / "model_comparison.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("→", out.relative_to(store.ROOT))
