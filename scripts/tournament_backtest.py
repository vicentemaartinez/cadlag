"""Argentina: pronósticos de cada torneo con el formato actual (Apertura y Clausura 2025, Apertura
2026), fuera de muestra, al empezar y al 25, 50, 75 y 100 % de la fase de zonas.

Mide, para el núcleo y para el modelo oficial: −log P(campeón real), y la puntuación de Brier y el
log score de "clasifica a octavos" para los 30 equipos. Los hiperparámetros son los del backtest
(temporadas anteriores a 2016). Son solo tres torneos: es una verificación de que la llave y las
zonas se simulan bien, no una prueba con potencia estadística.

Uso: python scripts/tournament_backtest.py   → outputs/analysis/tournament_backtest.json
"""
import json

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.data.events import load_stats
from cadlag.leagues import LEAGUES
from cadlag.model.filter import DynamicDC, prepare
from cadlag.model.filter_shots import with_shots
from cadlag.model.filter_shots2 import TwoLatentShots, season_transition_np
from cadlag.eval.backtest import preseason_transition
from cadlag.sim.season import season_state, simulate
from cadlag.sim.split import _tourn

comp = "arg.1"; lg = LEAGUES[comp]; S = 4000
m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
B = store.ROOT / "outputs" / "backtest"
ld = prepare(m, comp); ld2 = with_shots(prepare(m, comp), load_stats())
models = {"dc": (DynamicDC(ld, np.load(B / f"{comp}_h_2016.npy")), ld),
          "shots2": (TwoLatentShots(ld2, np.load(B / f"{comp}_shots2_h_2016.npy")), ld2)}
tix = {t: i for i, t in enumerate(ld.team_ids)}
# campeones reales (la final de 2025-II se definió por penales: Estudiantes a Racing)
CHAMP = {(2025, 0): None, (2025, 1): "Estudiantes de La Plata", (2026, 0): None}
d = m[m.comp == comp].copy(); d["tourn"] = d["slug"].map(_tourn)
plan = []
for (y, t), champ in CHAMP.items():
    reg = d[(d.syear == y) & (d.stage == "regular") & (d.tourn == t) & d.finished].sort_values("date")
    fin = d[(d.syear == y) & (d.stage == "playoff") & (d.tourn == t) & d.slug.astype(str).str.endswith("final")].iloc[0]
    if champ is None:
        champ = names[fin.home] if fin.hg > fin.ag else names[fin.away]
    # quiénes clasificaron: los 16 que jugaron octavos
    r16 = d[(d.syear == y) & (d.stage == "playoff") & (d.tourn == t) & d.slug.astype(str).str.contains("round-of-16")]
    q = set(r16.home) | set(r16.away)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = int(round(f * len(reg)))
        asof = reg["date"].iloc[0] - pd.Timedelta(seconds=1) if k == 0 else reg["date"].iloc[k - 1]
        plan.append((y, t, f, asof, champ, q))
dn = ld.matches["date"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
js = [int(np.searchsorted(dn, a.tz_convert("UTC").tz_localize(None).to_datetime64(), side="right") - 1) for *_, a, _, _ in plan]
uj = sorted(set(js)); at = {j: i for i, j in enumerate(uj)}
states = {k: mod.run_checkpoints(uj)[:2] for k, (mod, _) in models.items()}
rows = []
for (y, t, f, asof, champ, q), j in zip(plan, js):
    st = season_state(m, comp, y, tix, asof=asof)
    for name, (mod, ldx) in models.items():
        mv, Pm = states[name][0][at[j]], states[name][1][at[j]]
        hp = mod.params()
        if not len(st.done):
            mv, Pm = (preseason_transition(mv, Pm, hp, ldx, y) if name == "dc" else season_transition_np(mv, Pm, hp, ldx, y))
        r = simulate(lg, st, mv, Pm, hp, ldx.N, n_sims=S, seed=0)
        X = r.extra
        ti = X["title"].mean(0); po = (X["zpos"] < 8).mean(0)
        ci = [names[x] for x in st.teams].index(champ)
        yq = np.array([x in q for x in st.teams], float)
        pc = np.clip(po, 1e-4, 1 - 1e-4)
        rows.append(dict(year=y, torneo=["Apertura", "Clausura"][t], frac=f, model=name, cur=st.split["cur"],
                         p_champ=float(ti[ci]), log_champ=float(-np.log(max(ti[ci], 1e-4))),
                         brier_po=float(((po - yq) ** 2).mean()),
                         log_po=float(-(yq * np.log(pc) + (1 - yq) * np.log(1 - pc)).mean()),
                         fav=names[st.teams[int(ti.argmax())]], p_fav=float(ti.max()), champ=champ))
        print(rows[-1], flush=True)
D = pd.DataFrame(rows)
out = {k: dict(log_champ=float(g.log_champ.mean()), p_champ=float(g.p_champ.mean()), brier_po=float(g.brier_po.mean()),
               log_po=float(g.log_po.mean())) for k, g in D.groupby("model")}
out["por_momento"] = D.pivot_table(index=["year", "torneo", "frac"], columns="model", values="p_champ").reset_index().to_dict("records")
(store.ROOT / "outputs" / "analysis" / "tournament_backtest.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "por_momento"}, indent=1))
