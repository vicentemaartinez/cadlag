"""E4: ¿mejoran los tiros al arco la predicción de goles? Backtest walk-forward pareado.

Para cada liga: hiperparámetros del modelo base y del modelo con tiros ajustados solo con datos
anteriores a ``test_from``; luego ambos filtros recorren la historia y se compara la
log-verosimilitud predictiva **de goles** (marcador exacto) y el RPS de 1X2 en el período de prueba.

Uso: python scripts/shots_analysis.py eng.1:2016 chi.1:2020 ...
"""
import json
import sys

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.data.events import load_stats
from cadlag.eval.backtest import outcome_index
from cadlag.eval.scores import rps
from cadlag.model.dixon_coles import outcome_probs, predictive_grid
from cadlag.model.filter import DynamicDC, prepare
from cadlag.model.filter_shots import DynamicDCShots, with_shots

m, _ = store.load()
stats = load_stats()
out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
res = {}
for arg in sys.argv[1:]:
    comp, tf = arg.split(":")
    tf = int(tf)
    ld = with_shots(prepare(m, comp), stats)
    upto = int(np.argmax(ld.matches["syear"].to_numpy() >= tf))
    hb = store.ROOT / "outputs" / "backtest" / f"{comp}_h_{tf}.npy"
    base = DynamicDC(ld, np.load(hb) if hb.exists() else None)
    if not hb.exists():
        base.fit(upto=upto)
        np.save(hb, base.h)
    shots = DynamicDCShots(ld)
    shots.fit(upto=upto)
    _, _, ob = base.run()
    _, _, os_ = shots.run()
    test = np.arange(len(ld.matches)) >= upto
    d = ld.matches[test]
    o = outcome_index(d["hg"].to_numpy(), d["ag"].to_numpy())
    pb = outcome_probs(predictive_grid(ob["zbar"][test], ob["S"][test], base.params()["rho"]))
    ps = outcome_probs(predictive_grid(os_["zbar"][test], os_["S"][test], shots.params()["rho"]))
    dl = os_["logp_goals"][test] - ob["logp"][test]
    se = dl.std(ddof=1) / np.sqrt(len(dl))
    res[comp] = dict(test_from=tf, n=int(test.sum()), params_shots=shots.params(),
                     logscore_base=float(-ob["logp"][test].mean()), logscore_shots=float(-os_["logp_goals"][test].mean()),
                     delta_logp=float(dl.mean()), delta_se=float(se),
                     rps_base=float(rps(pb, o).mean()), rps_shots=float(rps(ps, o).mean()))
    print(comp, json.dumps(res[comp], default=float), flush=True)
json.dump(res, open(out / "shots_backtest.json", "w"), indent=1, default=float)
