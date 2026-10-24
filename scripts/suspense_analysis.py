"""Competitividad y suspenso por liga-temporada.

Para cada temporada completa (desde ``desde``): en cada fecha (cada n/k partidos) se simula el
resto con la predictiva posterior y se registra la creencia sobre el campeón. Con esa secuencia:
  * entropía del campeón en cada momento (bits),
  * sorpresa total de Ely–Frankel–Kamenica: sum_t ||mu_t - mu_{t-1}||^2,
  * fecha en que el campeón supera 95 % ("definición"),
y de la tabla final: HHI normalizado, Gini, Noll–Scully con empates.

Uso: python scripts/suspense_analysis.py [desde] [checkpoints]
"""
import json
import sys

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.eval.backtest import final_positions, preseason_transition
from cadlag.leagues import CORE_LEAGUES, LEAGUES
from cadlag.model.filter import prepare
from cadlag.pipeline import load_or_fit
from cadlag.sim import analysis as an
from cadlag.sim.season import season_state, simulate

since = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
rows, curves = [], {}
for comp in CORE_LEAGUES:
    league = LEAGUES[comp]
    ld = prepare(m, comp)
    mod = load_or_fit(ld)
    hp = mod.params()
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    dn = ld.matches["date"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    for s in sorted(set(ld.matches["syear"])):
        if s < since:
            continue
        reg = m[(m["comp"] == comp) & (m["syear"] == s) & (m["stage"] == "regular") & m["finished"]].sort_values("date")
        allreg = m[(m["comp"] == comp) & (m["syear"] == s) & (m["stage"] == "regular")]
        if len(reg) < 0.97 * len(allreg) or len(reg) == 0 or (comp == "chi.1" and s == 2019):
            continue
        cks = np.linspace(0, len(reg), K + 1).astype(int)[:-1]
        plan = []
        for k in cks:
            asof = reg["date"].iloc[0] - pd.Timedelta(seconds=1) if k == 0 else reg["date"].iloc[k - 1]
            j = np.searchsorted(dn, asof.tz_convert("UTC").tz_localize(None).to_datetime64(), side="right") - 1
            plan.append((k, asof, j))
        Ms, Ps, _ = mod.run_checkpoints([p[2] for p in plan])
        beliefs = []
        teams_s = None
        for (k, asof, j), mv, Pm in zip(plan, Ms, Ps):
            if k == 0:
                mv, Pm = preseason_transition(mv, Pm, hp, ld, s)
            st = season_state(m, comp, s, tix, asof=asof)
            teams_s = st.teams
            r = simulate(league, st, mv, Pm, hp, ld.N, n_sims=2000, seed=k)
            beliefs.append((r.pos == 0).mean(0))
        tm, fpos, fpts = final_positions(m, league, comp, s, tix, mod, ld)
        champ = tm[int(np.argmin(fpos))]
        ci = teams_s.index(champ)
        B = np.vstack([beliefs, np.eye(len(teams_s))[ci]])  # creencia final = certeza
        ss = an.surprise_suspense(B)
        decided = next((float(c / len(reg)) for c, b in zip(cks, B[:-1]) if b.max() >= 0.95), 1.0)
        n = len(teams_s)
        draw_rate = float((reg["hg"] == reg["ag"]).mean())
        bal = an.balance_metrics(fpts, 2 * (n - 1), draw_rate)
        rows.append(dict(comp=comp, syear=int(s), champion=names.get(champ), entropy_start=ss["entropy"][0],
                         entropy_mid=ss["entropy"][K // 2], total_surprise=ss["total_surprise"],
                         decided_at=decided, p_champ_preseason=float(B[0][ci]), **bal))
        curves[f"{comp}:{s}"] = dict(entropy=ss["entropy"].tolist(), surprise=ss["surprise"].tolist())
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "suspense_seasons.csv", index=False)
json.dump(curves, open(out / "suspense_curves.json", "w"))
print(df.groupby("comp")[["entropy_start", "total_surprise", "decided_at", "noll_scully", "hhi_norm", "gini",
                          "p_champ_preseason"]].mean().round(3).to_string())
