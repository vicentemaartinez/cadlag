"""Pendiente 1: ¿separar creación y definición mejora la predicción de goles? Backtest pareado.

Mismo protocolo que shots_analysis.py: hiperparámetros con temporadas anteriores a test_from,
luego ambos filtros recorren la historia; se compara la log-verosimilitud predictiva del marcador
exacto y el RPS de 1X2 en el período de prueba, y se reporta qué tan persistente resulta la
definición (varianzas y φ del bloque f, g frente al de tiros).

Uso: python scripts/shots2_analysis.py eng.1:2016 esp.1:2016 chi.1:2020
"""
import json
import sys

import numpy as np

from cadlag.data import store
from cadlag.data.events import load_stats
from cadlag.eval.backtest import outcome_index
from cadlag.eval.scores import rps
from cadlag.model.dixon_coles import outcome_probs, predictive_grid
from cadlag.model.filter import DynamicDC, prepare
from cadlag.model.filter_shots import with_shots
from cadlag.model.filter_shots2 import TwoLatentShots, outcome_probs_mc

m, _ = store.load()
stats = load_stats()
out = store.ROOT / "outputs" / "analysis"
path = out / "shots2_backtest.json"
res = json.loads(path.read_text()) if path.exists() else {}
for arg in sys.argv[1:]:
    comp, tf = arg.split(":"); tf = int(tf)
    ld = with_shots(prepare(m, comp), stats)
    upto = int(np.argmax(ld.matches["syear"].to_numpy() >= tf))
    hb = store.ROOT / "outputs" / "backtest" / f"{comp}_h_{tf}.npy"
    base = DynamicDC(ld, np.load(hb) if hb.exists() else None)
    if not hb.exists():
        base.fit(upto=upto); np.save(hb, base.h)
    two = TwoLatentShots(ld)
    two.fit(upto=upto)
    _, _, ob = base.run()
    mT, PT, ot = two.run()
    test = np.arange(len(ld.matches)) >= upto
    d = ld.matches[test]
    o = outcome_index(d["hg"].to_numpy(), d["ag"].to_numpy())
    pb = outcome_probs(predictive_grid(ob["zbar"][test], ob["S"][test], base.params()["rho"]))
    hp = two.params()
    pt = outcome_probs_mc(ot["zbar"][test], ot["S"][test], hp["r"])
    dl = ot["logp_goals"][test] - ob["logp"][test]
    # persistencia: correlación entre temporadas implicada (φ) y varianza estacionaria de cada bloque
    N = ld.N; act = ld.active[-1]
    sd_now = lambda blk: float(np.std(mT[blk * N:(blk + 1) * N][act]))  # noqa: E731
    np.save(store.ROOT / "outputs" / "backtest" / f"{comp}_shots2_h_{tf}.npy", two.h)
    from cadlag.model.filter_shots2 import clean_sot
    seas = d["syear"].to_numpy()
    by_season = {int(s): float(dl[seas == s].mean()) for s in np.unique(seas)}
    res[comp] = dict(test_from=tf, n=int(test.sum()), share_with_sot=float(clean_sot(ld.matches)[2].mean()), delta_by_season=by_season,
                     logscore_base=float(-ob["logp"][test].mean()), logscore_two=float(-ot["logp_goals"][test].mean()),
                     delta_logp=float(dl.mean()), delta_se=float(dl.std(ddof=1) / np.sqrt(len(dl))),
                     rps_base=float(rps(pb, o).mean()), rps_two=float(rps(pt, o).mean()),
                     params=hp, sd_teams=dict(creacion=sd_now(0), prevencion=sd_now(1), definicion=sd_now(2), arquero=sd_now(3)))
    print(comp, json.dumps(res[comp], default=float), flush=True)
    path.write_text(json.dumps(res, indent=1, default=float))
