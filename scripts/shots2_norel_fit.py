"""Ablación del prior de descendidos (docs/teoria.md §23): ajusta el modelo oficial con temporadas
anteriores a ``test_from`` tratando a todos los equipos nuevos igual (sin máscara de descendidos) y
compara la log-verosimilitud predictiva del marcador fuera de muestra con la versión completa
(outputs/backtest/<liga>_shots2_h_<año>.npy, de shots2_analysis.py).

Uso: python scripts/shots2_norel_fit.py eng.2:2016   → outputs/backtest/<liga>_shots2norel_h_<año>.npy
"""
import json
import sys

import numpy as np

from cadlag.data import store
from cadlag.data.events import load_stats
from cadlag.model.filter import prepare
from cadlag.model.filter_shots import with_shots
from cadlag.model.filter_shots2 import TwoLatentShots

m, _ = store.load()
stats = load_stats()
B = store.ROOT / "outputs" / "backtest"
out = store.ROOT / "outputs" / "analysis" / "relegated_prior.json"
res = json.loads(out.read_text()) if out.exists() else {}
for arg in sys.argv[1:]:
    comp, tf = arg.split(":"); tf = int(tf)
    ld = with_shots(prepare(m, comp), stats)
    upto = int(np.argmax(ld.matches["syear"].to_numpy() >= tf))
    full = TwoLatentShots(ld, np.load(B / f"{comp}_shots2_h_{tf}.npy"))
    ld0 = with_shots(prepare(m, comp), stats)
    ld0.relegated = np.zeros_like(ld0.active)
    nore = TwoLatentShots(ld0)
    nore.fit(upto=upto)
    np.save(B / f"{comp}_shots2norel_h_{tf}.npy", nore.h)
    _, _, a = full.run()
    _, _, b = nore.run()
    test = np.arange(len(ld.matches)) >= upto
    # partidos de los equipos recién llegados en su primera temporada, donde el prior pesa
    si = ld.matches["si"].to_numpy(); hi = ld.matches["hi"].to_numpy(); ai = ld.matches["ai"].to_numpy()
    new = ld.promoted[si, hi] | ld.promoted[si, ai]
    rel = ld.relegated[si, hi] | ld.relegated[si, ai]
    d = a["logp_goals"] - b["logp_goals"]
    def ms(mask):
        x = d[mask]
        return dict(delta=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))), n=int(len(x)))
    p = full.params()
    res[comp] = dict(test_from=tf, todos=ms(test), con_recien_llegados=ms(test & new), con_descendidos=ms(test & rel),
                     prior=dict(a_rel=p["a_rel"], b_rel=p["b_rel"], v_rel=p["vrel"], a_prom=p["a_prom"], b_prom=p["b_prom"],
                                v_prom=p["vprom"]))
    print(comp, json.dumps(res[comp]), flush=True)
    out.write_text(json.dumps(res, indent=1))
