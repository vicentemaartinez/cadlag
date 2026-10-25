"""Modelo oficial (dos latentes) para las ligas europeas que solo se usan en la capa europea.

Para cada liga ajusta los hiperparámetros una sola vez, con las temporadas 2012 a 2015 y partiendo
de los de Portugal (el ajuste completo de dos latentes toma unos 40 minutos por liga en esta
máquina). Esos mismos se usan en los backtests (``outputs/backtest/<liga>_shots2_h_2016.npy``) y en
producción (``models/<liga>_shots2_h.npy``): no miran el período de prueba y, para un filtro que
solo alimenta la capa europea, no hace falta reajustarlos con todo. Los tiros al arco de ESPN faltan
en parte de estas ligas: el modelo usa la marginal de goles en esos partidos. Portugal se ajustó
con 2010-2015 completo.

Uso: python scripts/fit_extra_leagues.py [ligas...]   (por defecto ``leagues.EUROPE_EXTRA``)
"""
import sys

import numpy as np

from cadlag.data import store
from cadlag.data.events import load_stats
from cadlag.leagues import EUROPE_EXTRA
from cadlag.model.filter import prepare
from cadlag.model.filter_shots import with_shots
from cadlag.model.filter_shots2 import TwoLatentShots

comps = sys.argv[1:] or EUROPE_EXTRA
m, _ = store.load()
stats = load_stats()
B = store.ROOT / "outputs" / "backtest"
h0 = np.load(B / "por.1_shots2_h_2016.npy")
for comp in comps:
    if comp == "por.1":
        h = h0
    else:
        ld = with_shots(prepare(m, comp, min_syear=2012), stats)
        mod = TwoLatentShots(ld, h0)
        upto = int(np.searchsorted(ld.matches["syear"].to_numpy(), 2016))
        mod.fit(upto=upto, maxiter=40)
        h = mod.h
    np.save(B / f"{comp}_shots2_h_2016.npy", h)
    np.save(store.ROOT / "models" / f"{comp}_shots2_h.npy", h)
    print(f"{comp}: listo", flush=True)
