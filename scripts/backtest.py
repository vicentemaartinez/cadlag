"""Backtest walk-forward de una liga: partidos y temporadas.

Uso:  python scripts/backtest.py eng.1 2016 [n_sims]
Escribe outputs/backtest/<liga>_matches.parquet, _seasons.parquet y _summary.json.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.eval import backtest as bt
from cadlag.leagues import LEAGUES

comp, test_from = sys.argv[1], int(sys.argv[2])
n_sims = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 4000
out = store.ROOT / "outputs" / "backtest"
out.mkdir(parents=True, exist_ok=True)
m, _ = store.load()
t0 = time.time()
hf = out / f"{comp}_h_{test_from}.npy"
h = np.load(hf) if hf.exists() and "--refit" not in sys.argv else None
ld, mod, upto = bt.fit_for_test(m, comp, test_from, h=h)
np.save(hf, mod.h)
print(f"ajuste {time.time() - t0:.0f}s", mod.params(), flush=True)
t = bt.match_backtest(m, ld, mod, upto, comp)
t.drop(columns=["date"]).assign(date=t["date"].astype(str)).to_parquet(out / f"{comp}_matches.parquet")
sm = bt.summarize_matches(t)
print(sm.round(4).to_string(), flush=True)
last = int(m[(m.comp == comp) & (m.stage == "regular")].groupby("syear")["finished"].all().loc[lambda s: s].index.max())
seasons = list(range(test_from, last + 1))
t0 = time.time()
sdf, mats = bt.season_backtest(m, LEAGUES[comp], ld, mod, seasons, n_sims=n_sims)
print(f"temporadas {time.time() - t0:.0f}s", flush=True)
sdf.to_parquet(out / f"{comp}_seasons.parquet")
ss = bt.summarize_seasons(sdf)
print(ss.round(4).to_string())
json.dump(dict(comp=comp, test_from=test_from, seasons=seasons, params=mod.params(), h=list(map(float, mod.h)),
               matches=sm.reset_index().to_dict("records"), seasons_summary=ss.reset_index().to_dict("records")),
          open(out / f"{comp}_summary.json", "w"), indent=1, default=float)
