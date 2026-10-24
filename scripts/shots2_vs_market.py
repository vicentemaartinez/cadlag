"""El modelo oficial (dos latentes) frente a Bet365 en exactamente los mismos partidos del backtest
del núcleo (los que tienen cuotas). Reemplaza las probabilidades del modelo en
outputs/backtest/<liga>_matches.parquet por las del de dos latentes (hiperparámetros del backtest,
fuera de muestra) y resume con las mismas funciones.

Uso: python scripts/shots2_vs_market.py      → outputs/analysis/shots2_vs_market.json
"""
import json

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.data.events import load_stats
from cadlag.eval import backtest as bt
from cadlag.leagues import CORE_LEAGUES
from cadlag.model.filter import prepare
from cadlag.model.filter_shots import with_shots
from cadlag.model.filter_shots2 import TwoLatentShots, outcome_probs_mc

m, _ = store.load()
stats = load_stats()
B = store.ROOT / "outputs" / "backtest"
out = {}
for comp in CORE_LEAGUES:
    tp = B / f"{comp}_matches.parquet"
    hs = sorted(B.glob(f"{comp}_shots2_h_*.npy"))
    if not tp.exists() or not hs:
        continue
    t = pd.read_parquet(tp)
    ld = with_shots(prepare(m, comp), stats)
    two = TwoLatentShots(ld, np.load(hs[0]))
    _, _, o = two.run()
    ids = ld.matches["id"].astype(str).to_numpy()
    sel = np.isin(ids, t["id"].astype(str).to_numpy())
    pr = outcome_probs_mc(o["zbar"][sel], o["S"][sel], two.params()["r"])
    P = pd.DataFrame(pr, columns=["z_h", "z_d", "z_a"]).assign(id=ids[sel], z_logp=o["logp_goals"][sel])
    t = t.assign(id=t["id"].astype(str)).merge(P, on="id", how="left")
    base = bt.summarize_matches(t)
    t2 = t.assign(p_h=t.z_h, p_d=t.z_d, p_a=t.z_a, logp_score=t.z_logp)
    two_sum = bt.summarize_matches(t2)
    rows = {k: dict(rps=float(v["rps"]), log=float(v["log"]), n=int(v["n"])) for k, v in base.iterrows()}
    rows["Dos latentes"] = {k: (float(v) if k != "n" else int(v)) for k, v in two_sum.loc["Modelo (predictiva posterior)", ["rps", "log", "n"]].items()}
    out[comp] = rows
    print(comp, {k: round(v["rps"], 4) for k, v in rows.items()}, flush=True)
(store.ROOT / "outputs" / "analysis" / "shots2_vs_market.json").write_text(json.dumps(out, indent=1))
