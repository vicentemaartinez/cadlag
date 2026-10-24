"""Backtest pareado a nivel de temporada: núcleo (Dixon–Coles dinámico) frente a dos latentes.

Para cada liga y temporada de prueba, y en cuatro momentos (pretemporada, 25 %, 50 % y 75 % de los
partidos jugados), cada modelo pronostica la posición final de cada equipo con la predictiva
posterior; se compara −log P(posición que de verdad terminó). En ligas con ascenso se compara
además −log P(ascendió o no), con el ascenso real leído de la división superior de la temporada
siguiente (así cuentan los playoffs, que ESPN a veces define por penales). Los hiperparámetros son
los del backtest de partidos (ajustados solo con temporadas anteriores a la de prueba), así que
todo es fuera de muestra. El error estándar agrupa por temporada (los equipos de una misma
temporada no son independientes).

Variantes: "dc" (núcleo), "shots2" (oficial) y, si existe outputs/backtest/<liga>_shots2norel_h_*.npy,
"shots2_norel" (oficial sin el prior propio de los descendidos de la división superior).

Uso: python scripts/season_compare.py [n_sims] [ligas...]   → outputs/analysis/season_compare.{csv,json}
"""
import json
import sys

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.data.events import load_stats
from cadlag.eval.backtest import final_positions, preseason_transition
from cadlag.leagues import CORE_LEAGUES, LEAGUES
from cadlag.model.filter import DynamicDC, prepare
from cadlag.model.filter_shots import with_shots
from cadlag.model.filter_shots2 import TwoLatentShots, season_transition_np
from cadlag.sim.season import season_state, simulate

args = [a for a in sys.argv[1:]]
S = int(args.pop(0)) if args and args[0].isdigit() else 4000
COMPS = args or CORE_LEAGUES
FRACS = (0.0, 0.25, 0.5, 0.75)
TEST_FROM = {c: 2016 for c in CORE_LEAGUES} | {"chi.1": 2020}
m, _ = store.load()
stats = load_stats()
bt = store.ROOT / "outputs" / "backtest"
an = store.ROOT / "outputs" / "analysis"
csv = an / "season_compare.csv"
old = pd.read_csv(csv, dtype={"team": str}) if csv.exists() else pd.DataFrame()
if len(old):
    old = old[~old["liga"].isin(COMPS)]
rows = []


def promoted_set(comp, s):
    up = LEAGUES[comp].parent
    nxt = m[(m.comp == up) & (m.syear == s + 1)]
    cur = m[(m.comp == comp) & (m.syear == s)]
    return (set(nxt.home) | set(nxt.away)) & (set(cur.home) | set(cur.away))


for comp in COMPS:
    tf = TEST_FROM[comp]
    league = LEAGUES[comp]
    ld = prepare(m, comp)
    ld2 = with_shots(prepare(m, comp), stats)
    assert (ld.matches["id"].to_numpy() == ld2.matches["id"].to_numpy()).all()
    dc = DynamicDC(ld, np.load(bt / f"{comp}_h_{tf}.npy"))
    models = {"dc": (dc, ld)}
    models["shots2"] = (TwoLatentShots(ld2, np.load(bt / f"{comp}_shots2_h_{tf}.npy")), ld2)
    nr = bt / f"{comp}_shots2norel_h_{tf}.npy"
    if nr.exists():
        ld3 = with_shots(prepare(m, comp), stats)
        ld3.relegated = np.zeros_like(ld3.active)
        models["shots2_norel"] = (TwoLatentShots(ld3, np.load(nr)), ld3)
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    done = m[(m.comp == comp) & (m.stage == "regular")].groupby("syear")["finished"].all()
    seasons = [int(s) for s in done[done].index if s >= tf]
    has_up = bool(league.promotion_playoff) and league.parent
    if has_up:
        nxt = set(m[m.comp == league.parent].syear.unique())
        seasons = [s for s in seasons if s + 1 in nxt]
    dn = ld.matches["date"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    plan = []
    for s in seasons:
        reg = m[(m.comp == comp) & (m.syear == s) & (m.stage == "regular") & m.finished].sort_values("date")
        for f in FRACS:
            k = int(round(f * len(reg)))
            asof = reg["date"].iloc[0] - pd.Timedelta(seconds=1) if k == 0 else reg["date"].iloc[k - 1]
            j = int(np.searchsorted(dn, asof.tz_convert("UTC").tz_localize(None).to_datetime64(), side="right") - 1)
            plan.append((s, f, asof, j))
    uj = sorted({p[3] for p in plan})
    pos_of = {j: i for i, j in enumerate(uj)}
    states = {k: mod.run_checkpoints(uj)[:2] for k, (mod, _) in models.items()}
    n_rel = next((len(z.positions) for z in league.zones if z.key == "relegation"), 0)
    for s, f, asof, j in plan:
        st = season_state(m, comp, s, tix, asof=asof)
        teams, final, _ = final_positions(m, league, comp, s, tix, dc, ld)
        fin = dict(zip(teams, final))
        y = np.array([fin[t] for t in st.teams])
        n = len(st.teams)
        up = np.array([t in promoted_set(comp, s) for t in st.teams]) if has_up else None
        out = {}
        for name, (mod, ldx) in models.items():
            mv, Pm = states[name][0][pos_of[j]], states[name][1][pos_of[j]]
            hp = mod.params()
            if f == 0.0:
                mv, Pm = (preseason_transition(mv, Pm, hp, ldx, s) if name == "dc" else season_transition_np(mv, Pm, hp, ldx, s))
            r = simulate(league, st, mv, Pm, hp, ldx.N, n_sims=S, seed=0)
            Mx = r.position_matrix()
            Mc = np.clip(Mx, 1 / (S * 10), 1); Mc /= Mc.sum(1, keepdims=True)
            o = dict(log=-np.log(Mc[np.arange(n), y]), pchamp=Mx[:, 0], prel=Mx[:, n - n_rel:].sum(1) if n_rel else np.zeros(n))
            if up is not None:
                pu = np.clip(r.promoted.mean(0), 1 / (S * 10), 1 - 1 / (S * 10))
                o["pup"] = pu
                o["logup"] = -(up * np.log(pu) + (~up) * np.log(1 - pu))
            out[name] = o
        for i, t in enumerate(st.teams):
            row = dict(liga=comp, syear=s, frac=f, team=t, pos=int(y[i]))
            if up is not None:
                row["up"] = int(up[i])
            for name, o in out.items():
                sfx = {"dc": "dc", "shots2": "two", "shots2_norel": "norel"}[name]
                for k, v in o.items():
                    row[f"{k}_{sfx}"] = float(v[i])
            rows.append(row)
    D = pd.DataFrame([r for r in rows if r["liga"] == comp])
    g = D.assign(d=D.log_dc - D.log_two).groupby("syear")["d"].mean()
    print(f"{comp}: Δlog posición (núcleo − dos latentes; >0 = gana dos latentes) {g.mean():+.4f} ± {g.std(ddof=1) / np.sqrt(len(g)):.4f}"
          f"  ({len(g)} temporadas)", flush=True)
    if "logup_dc" in D:
        for a, b in (("dc", "two"), ("norel", "two")):
            if f"logup_{a}" in D:
                g = D.assign(d=D[f"logup_{a}"] - D.logup_two).groupby("syear")["d"].mean()
                print(f"  ascenso: Δlog ({a} − dos latentes) {g.mean():+.4f} ± {g.std(ddof=1) / np.sqrt(len(g)):.4f}", flush=True)
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(csv, index=False)

D = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
D["d"] = D.log_dc - D.log_two
res = {}
for key, G in [*D.groupby("liga"), ("todas", D)]:
    cl = G.groupby(["liga", "syear"])["d"].mean()
    byf = {str(f): float(x) for f, x in G.groupby("frac")["d"].mean().items()}
    res[key] = dict(delta_log_pos=float(cl.mean()), se=float(cl.std(ddof=1) / np.sqrt(len(cl))), n_seasons=int(len(cl)),
                    by_frac=byf, log_dc=float(G.log_dc.mean()), log_two=float(G.log_two.mean()))
    if key != "todas" and "logup_dc" in G and G["logup_dc"].notna().any():
        for a in ("dc", "norel"):
            c = f"logup_{a}"
            if c in G and G[c].notna().any():
                cu = G.assign(e=G[c] - G.logup_two).groupby("syear")["e"].mean()
                res[key][f"delta_log_up_{a}"] = float(cu.mean())
                res[key][f"se_up_{a}"] = float(cu.std(ddof=1) / np.sqrt(len(cu)))
(an / "season_compare.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["todas"], indent=1))
