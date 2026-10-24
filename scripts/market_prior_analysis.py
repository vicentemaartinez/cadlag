"""Pendiente 3: ¿el mercado de pases mejora la pretemporada?

Para cada club de las cinco grandes ligas y cada temporada s:
  x_i  = (valor de mercado que llega − valor que se va) / mediana del movimiento bruto de la liga
         en esa temporada (ventanas de verano e invierno, préstamos incluidos)
  y_i  = fuerza neta (α+δ) filtrada al final de s − media a priori de pretemporada de s
El modelo base ya hace regresión a la media (φ) y trata a los ascendidos con un prior común; y_i
es lo que el modelo no anticipó. Se ajusta y_i = γ x_i + ε con temporadas < 2016 y se evalúa
fuera de muestra (2016-2025): correlación, y pronósticos de pretemporada con la media desplazada
en γ x_i (mitad al ataque y mitad a la defensa) contra la posición y los puntos reales.

Uso: python scripts/market_prior_analysis.py
Escribe outputs/analysis/market_prior.json y market_prior_seasons.csv
"""
import json

import numpy as np
import pandas as pd

from cadlag.data import store, transfers
from cadlag.eval.backtest import final_positions, preseason_transition
from cadlag.eval.scores import rps
from cadlag.leagues import LEAGUES
from cadlag.model.filter import DynamicDC, prepare
from cadlag.sim.season import season_state, simulate

TOP5 = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"]
TEST_FROM, S = 2016, 4000
m, teams = store.load()
tr = transfers.load(m, teams, loans=True, dedupe=False)
tr = tr[tr["club_id"].notna()]
g = tr.assign(v=tr["market_value"].fillna(0.0)).groupby(["comp", "season", "club_id", "movement"])["v"].sum().unstack(fill_value=0.0)
g = g.reset_index().rename(columns={"in": "v_in", "out": "v_out"})
g["gross"] = g["v_in"] + g["v_out"]
g["x"] = (g["v_in"] - g["v_out"]) / g.groupby(["comp", "season"])["gross"].transform("median").clip(lower=1e6)
feat = {(r.comp, int(r.season), r.club_id): r.x for r in g.itertuples()}

rows, states = [], {}
for comp in TOP5:
    ld = prepare(m, comp)
    h = np.load(store.ROOT / "outputs" / "backtest" / f"{comp}_h_{TEST_FROM}.npy")
    mod = DynamicDC(ld, h); hp = mod.params()
    d = ld.matches
    seasons = [s for s in ld.seasons if s >= 2010 and s <= 2025]
    ends = {s: int(np.where(d["syear"].to_numpy() == s)[0].max()) for s in ld.seasons if (d["syear"] == s).any()}
    plan = [(s, ends[s - 1], ends[s]) for s in seasons if s - 1 in ends and s in ends]
    uniq = sorted({p for q in plan for p in q[1:]})  # sin repetidos: el fin de s es el previo de s+1
    Mu, Pu, _ = mod.run_checkpoints(uniq)
    pos = {p: k for k, p in enumerate(uniq)}
    tix = {t: i for i, t in enumerate(ld.team_ids)}; N = ld.N
    for k, (s, i0, i1) in enumerate(plan):
        m0, P0 = preseason_transition(Mu[pos[i0]], Pu[pos[i0]], hp, ld, s)
        m1 = Mu[pos[i1]]
        states[(comp, s)] = (m0, P0)
        for t in np.where(ld.active[ld.seasons.index(s)])[0]:
            tid = ld.team_ids[t]
            rows.append(dict(comp=comp, season=s, team=tid, x=feat.get((comp, s, tid), 0.0),
                             has=(comp, s, tid) in feat, promoted=bool(ld.promoted[ld.seasons.index(s), t]),
                             pre=m0[t] + m0[N + t], end=m1[t] + m1[N + t]))
    print(comp, "listo", flush=True)
D = pd.DataFrame(rows); D["y"] = D["end"] - D["pre"]
D.to_csv(store.ROOT / "outputs" / "analysis" / "market_prior_teams.csv", index=False)
import sys  # noqa: E402
if "--solo-datos" in sys.argv:
    sys.exit(0)
tr_, te_ = D[D.season < TEST_FROM], D[D.season >= TEST_FROM]
gamma = float((tr_.x * tr_.y).sum() / (tr_.x ** 2).sum())
res_tr = tr_.y - gamma * tr_.x
se = float(np.sqrt(res_tr.var() / (tr_.x ** 2).sum()))
r2_out = 1 - ((te_.y - gamma * te_.x) ** 2).sum() / (te_.y ** 2).sum()
corr_out = float(np.corrcoef(te_.x, te_.y)[0, 1])
gamma_prom = float((tr_[tr_.promoted].x * tr_[tr_.promoted].y).sum() / (tr_[tr_.promoted].x ** 2).sum())
print(f"γ = {gamma:.4f} ± {se:.4f}; fuera de muestra: corr {corr_out:.3f}, R² {r2_out:.3f}; ascendidos γ = {gamma_prom:.4f}", flush=True)

# --- pronóstico de pretemporada con y sin el desplazamiento ---------------------------
srows = []
for comp in TOP5:
    lg = LEAGUES[comp]; ld = prepare(m, comp); h = np.load(store.ROOT / "outputs" / "backtest" / f"{comp}_h_{TEST_FROM}.npy")
    mod = DynamicDC(ld, h); hp = mod.params(); N = ld.N; tix = {t: i for i, t in enumerate(ld.team_ids)}
    for s in range(TEST_FROM, 2026):
        if (comp, s) not in states:
            continue
        m0, P0 = states[(comp, s)]
        reg = m[(m.comp == comp) & (m.syear == s) & (m.stage == "regular") & m.finished].sort_values("date")
        st = season_state(m, comp, s, tix, asof=reg["date"].iloc[0] - pd.Timedelta(seconds=1))
        tm, fin, fpts = final_positions(m, lg, comp, s, tix, mod, ld)
        y = np.array([dict(zip(tm, fin))[t] for t in st.teams]); ypts = np.array([dict(zip(tm, fpts))[t] for t in st.teams])
        n = len(st.teams)
        prom = set(np.array(ld.team_ids)[ld.promoted[ld.seasons.index(s)]])
        for label in ("base", "mercado", "mercado_ascendidos"):
            mm = m0.copy()
            for t in st.teams:
                i = tix[t]; x = feat.get((comp, s, t), 0.0)
                g_ = 0.0 if label == "base" else (gamma if label == "mercado" else (gamma_prom if t in prom else 0.0))
                dx = g_ * x / 2
                mm[i] += dx; mm[N + i] += dx
            r = simulate(lg, st, mm, P0, hp, N, n_sims=S, seed=1)
            M = r.position_matrix(); Mc = np.clip(M, 1 / (S * 10), 1); Mc /= Mc.sum(1, keepdims=True)
            xp, sd = r.pts.mean(0), r.pts.std(0)
            pit = (r.pts < ypts).mean(0) + 0.5 * (r.pts == ypts).mean(0)
            pm = np.array([t in prom for t in st.teams])
            lp = -np.log(Mc[np.arange(n), y])
            srows.append(dict(comp=comp, season=s, modelo=label, log_pos=float(lp.mean()),
                              log_pos_prom=float(lp[pm].mean()) if pm.any() else np.nan,
                              rps_pos=float(rps(M, y).mean()), mae_pts=float(np.abs(xp - ypts).mean()),
                              z2=float((((ypts - xp) / sd) ** 2).mean()), cover90=float(((pit > .05) & (pit < .95)).mean())))
    print(comp, "temporadas listas", flush=True)
SR = pd.DataFrame(srows)
out = store.ROOT / "outputs" / "analysis"
SR.to_csv(out / "market_prior_seasons.csv", index=False)
summ = SR.groupby("modelo")[["log_pos", "log_pos_prom", "rps_pos", "mae_pts", "z2", "cover90"]].mean()
piv = SR.pivot_table(index=["comp", "season"], columns="modelo", values="log_pos")
dlog = (piv["mercado"] - piv["base"]); dlog_p = (piv["mercado_ascendidos"] - piv["base"])
res = dict(gamma=gamma, gamma_se=se, gamma_promoted=gamma_prom, n_train=len(tr_), n_test=len(te_),
           corr_out=corr_out, r2_out=float(r2_out), share_with_transfers=float(D.has.mean()),
           seasons=summ.to_dict(), delta_log_pos=float(dlog.mean()), delta_log_pos_se=float(dlog.std(ddof=1) / np.sqrt(len(dlog))),
           wins=int((dlog < 0).sum()), n_seasons=int(len(dlog)),
           delta_log_pos_ascendidos=float(dlog_p.mean()), delta_log_pos_ascendidos_se=float(dlog_p.std(ddof=1) / np.sqrt(len(dlog_p))),
           wins_ascendidos=int((dlog_p < 0).sum()))
print(json.dumps(res, indent=1, default=float))
(out / "market_prior.json").write_text(json.dumps(res, indent=1, default=float))
