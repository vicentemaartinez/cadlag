"""Repite con una grilla más amplia los eventos de rare_events.csv cuyas dos semillas discrepan más
de 3 veces (señal de que la mejor propuesta quedó en el borde de la grilla), y marca como
imposibles los que la cota por programación entera descarta.

Uso: python scripts/rare_refine.py
"""
import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.leagues import LEAGUES
from cadlag.model.filter import prepare
from cadlag.pipeline import current_season, load_or_fit
from cadlag.sim import rare
from cadlag.sim.analysis import position_bounds
from cadlag.sim.season import season_state

out = store.ROOT / "outputs" / "analysis" / "rare_events.csv"
D = pd.read_csv(out)
m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
inv = {v: k for k, v in names.items()}
D["estado"] = "ok"
for comp, g in D.groupby("liga"):
    ld = prepare(m, comp); mod = load_or_fit(ld, verbose=False); hp = mod.params(); mT, PT, _ = mod.run()
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    st = season_state(m, comp, current_season(m, comp), tix); lg = LEAGUES[comp]
    n = len(st.teams); n_rel = next((len(z.positions) for z in lg.zones if z.key == "relegation"), 0)
    bounds = None
    for i, r in g.iterrows():
        t = st.teams.index(next(x for x in st.teams if names.get(x) == r.equipo))
        down = r.evento.startswith("el favorito")
        if r.p == 0:
            if bounds is None:
                bounds = position_bounds(st).set_index("team")
            bw = bounds.loc[st.teams[t]]
            if (down and bw["worst_position"] <= n - n_rel) or (not down and bw["best_position"] > 1):
                D.loc[i, ["p", "estado"]] = [0.0, "imposible (cota entera)"]
                continue
        s1, s2 = r.p_semilla_1, r.p_semilla_2
        if min(s1, s2) > 0 and max(s1, s2) / min(s1, s2) <= 3:
            continue
        ev = (lambda rr, t=t: rr.pos[:, t] >= n - n_rel) if down else (lambda rr, t=t: rr.pos[:, t] == 0)
        res = [rare.importance_results(lg, st, mT, PT, hp, ld.N, ev, t, sign=1.0 if down else -1.0, seed=s,
                                       thetas=(0.3, 0.45, 0.6, 0.8, 1.0, 1.3), shifts=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
                                       pilot=2000, n_final=40000) for s in (11, 12, 13)]
        ps = np.array([x.p for x in res])
        ratio = ps.max() / ps.min() if ps.min() > 0 else np.inf
        D.loc[i, ["p", "p_semilla_1", "p_semilla_2", "c", "theta", "error_relativo"]] = [
            float(np.exp(np.log(np.maximum(ps, 1e-300)).mean())), ps[0], ps[1], res[0].levels[0], res[0].levels[1],
            float(np.std(np.log(np.maximum(ps, 1e-300)), ddof=1))]
        D.loc[i, "estado"] = "ok (grilla amplia)" if ratio <= 3 else f"solo orden de magnitud (semillas ×{ratio:.0f})"
        D.loc[i, "uno_en"] = 1 / D.loc[i, "p"]
        print(comp, r.equipo, ps, D.loc[i, "estado"], flush=True)
D.to_csv(out, index=False)
print(D[["liga", "evento", "equipo", "p", "uno_en", "estado"]].to_string())
