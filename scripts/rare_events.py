"""Eventos casi imposibles por muestreo por importancia (cadlag.sim.rare.importance_results).

1. Validación en la Premier contra fuerza bruta (600.000 temporadas): el favorito termina 5.º o
   peor, y 10.º o peor.
2. Tabla: por liga, que el favorito descienda y que el último de la tabla salga campeón.

Uso: python scripts/rare_events.py
Escribe outputs/analysis/rare_events.csv y rare_validation.json
"""
import json
import time

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.leagues import CORE_LEAGUES, LEAGUES
from cadlag.model.filter import prepare
from cadlag.pipeline import current_season, load_or_fit
from cadlag.sim import rare
from cadlag.sim.season import season_state, simulate

out = store.ROOT / "outputs" / "analysis"
m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))


def setup(comp):
    ld = prepare(m, comp); mod = load_or_fit(ld, verbose=False); hp = mod.params(); mT, PT, _ = mod.run()
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    st = season_state(m, comp, current_season(m, comp), tix)
    return LEAGUES[comp], st, mT, PT, hp, ld.N


lg, st, mT, PT, hp, N = setup("eng.1")
r0 = simulate(lg, st, mT, PT, hp, N, n_sims=8000, seed=1)
fav = int(np.argmax((r0.pos == 0).mean(0)))
val = []
for k in (4, 9):
    ev = lambda r, k=k: r.pos[:, fav] >= k  # noqa: E731
    t0 = time.time(); pn, en = rare.naive(lg, st, mT, PT, hp, N, ev, n_sims=600000, seed=3); tn = time.time() - t0
    t0 = time.time(); ri = rare.importance_results(lg, st, mT, PT, hp, N, ev, fav, seed=5); ti = time.time() - t0
    n_is = ri.n + 2000 * 35
    val.append(dict(evento=f"{names[st.teams[fav]]} termina {k + 1}.º o peor", directo_p=pn, directo_error=en, directo_n=600000,
                    directo_s=tn, is_p=ri.p, is_error=ri.rel_err, is_n=n_is, is_s=ti, c=ri.levels[0], theta=ri.levels[1],
                    # cuántas temporadas directas equivale cada temporada con importancia (a igual error)
                    eficiencia=(en ** 2 * 600000) / (ri.rel_err ** 2 * n_is)))
    print(val[-1], flush=True)
(out / "rare_validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1))

rows = []
for comp in CORE_LEAGUES:
    lg, st, mT, PT, hp, N = setup(comp)
    n = len(st.teams)
    n_rel = next((len(z.positions) for z in lg.zones if z.key == "relegation"), 0)
    r0 = simulate(lg, st, mT, PT, hp, N, n_sims=8000, seed=1)
    fav = int(np.argmax((r0.pos == 0).mean(0)))
    pts = np.zeros(n)
    for row in st.done.itertuples():
        pts[row.hi] += 3 * (row.hg > row.ag) + (row.hg == row.ag); pts[row.ai] += 3 * (row.ag > row.hg) + (row.hg == row.ag)
    last = int(np.argmin(pts))
    for label, t, sign, ev in [("el favorito desciende", fav, 1.0, lambda r, t=fav: r.pos[:, t] >= n - n_rel),
                               ("el último sale campeón", last, -1.0, lambda r, t=last: r.pos[:, t] == 0)]:
        res = [rare.importance_results(lg, st, mT, PT, hp, N, ev, t, sign=sign, seed=s) for s in (7, 8)]
        p = np.mean([x.p for x in res])
        rows.append(dict(liga=comp, evento=label, equipo=names[st.teams[t]], p=p,
                         p_semilla_1=res[0].p, p_semilla_2=res[1].p, error_relativo=float(np.mean([x.rel_err for x in res]) / np.sqrt(2)),
                         c=res[0].levels[0], theta=res[0].levels[1], uno_en=(1 / p) if p > 0 else None,
                         directo_8000=float(ev(r0).mean()), jugados=int(len(st.done)), por_jugar=int(len(st.todo))))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out / "rare_events.csv", index=False)
print("ok")
