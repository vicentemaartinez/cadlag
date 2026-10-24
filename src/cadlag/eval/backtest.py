"""Backtesting *walk-forward* sin fuga de información.

Protocolo
---------
1. Los hiperparámetros se ajustan **solo** con temporadas anteriores a ``test_from``.
2. El filtro recorre la historia completa con esos hiperparámetros fijos: la predicción de
   cada partido usa únicamente partidos anteriores (es un filtro, no un suavizador).
3. Partidos: se comparan la predictiva posterior, el plug-in, el mercado (Bet365 sin margen,
   multiplicativo y Shin) y una línea base de frecuencias históricas.
4. Temporadas: en varios puntos de cada temporada de prueba se simula el resto con los
   modos ``posterior`` y ``plugin``, y se puntúa contra la posición final real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import mirror
from ..leagues import League
from ..model.dixon_coles import outcome_probs, predictive_grid
from ..model.filter import DynamicDC, prepare
from ..sim.season import season_state, simulate
from . import odds as oddsmod
from .scores import brier, log_score, rps


def outcome_index(hg, ag):
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def fit_for_test(matches: pd.DataFrame, comp: str, test_from: int, maxiter: int = 150, h=None):
    ld = prepare(matches, comp)
    mod = DynamicDC(ld, h)
    upto = int(np.argmax(ld.matches["syear"].to_numpy() >= test_from))
    if h is None:
        mod.fit(upto=upto, maxiter=maxiter)
    return ld, mod, upto


def match_backtest(matches: pd.DataFrame, ld, mod, upto: int, comp: str):
    _, _, o = mod.run()
    rho = mod.params()["rho"]
    d = ld.matches.copy()
    test = np.arange(len(d)) >= upto
    zbar, S = o["zbar"][test], o["S"][test]
    g_post = predictive_grid(zbar, S, rho)
    g_plug = predictive_grid(zbar, np.zeros_like(S), rho)
    t = d[test].copy()
    t[["p_h", "p_d", "p_a"]] = outcome_probs(g_post)
    t[["q_h", "q_d", "q_a"]] = outcome_probs(g_plug)
    t["logp_score"] = -np.log(g_post[np.arange(len(t)), t["hg"].clip(0, 10).astype(int), t["ag"].clip(0, 10).astype(int)])
    t["lam_h"] = np.exp(zbar[:, 0])
    t["lam_a"] = np.exp(zbar[:, 1])
    t["o"] = outcome_index(t["hg"], t["ag"])
    base = d[~test]
    freq = np.bincount(outcome_index(base["hg"], base["ag"]), minlength=3) / len(base)
    t[["b_h", "b_d", "b_a"]] = freq
    if comp in mirror.DIVISION:
        lk = mirror.link(matches, comp)[["id", "OddHome", "OddDraw", "OddAway", "HomeElo", "AwayElo"]]
        t = t.merge(lk, on="id", how="left")
        ok = t[["OddHome", "OddDraw", "OddAway"]].notna().all(1) & (t[["OddHome", "OddDraw", "OddAway"]] > 1).all(1)
        od = t.loc[ok, ["OddHome", "OddDraw", "OddAway"]].to_numpy(float)
        t.loc[ok, ["m_h", "m_d", "m_a"]] = oddsmod.multiplicative(od)
        t.loc[ok, ["s_h", "s_d", "s_a"]] = oddsmod.shin(od)
    return t


def summarize_matches(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = {"Modelo (predictiva posterior)": ["p_h", "p_d", "p_a"], "Modelo (plug-in)": ["q_h", "q_d", "q_a"],
            "Frecuencias históricas": ["b_h", "b_d", "b_a"]}
    if "s_h" in t:
        sets["Bet365 (multiplicativo)"] = ["m_h", "m_d", "m_a"]
        sets["Bet365 (Shin)"] = ["s_h", "s_d", "s_a"]
        common = t[["s_h"]].notna().all(1)
    else:
        common = np.ones(len(t), bool)
    tt = t[common]
    o = tt["o"].to_numpy()
    for name, cols in sets.items():
        p = tt[cols].to_numpy(float)
        rows.append(dict(pronostico=name, n=len(tt), log=log_score(p, o).mean(), brier=brier(p, o).mean(),
                         rps=rps(p, o).mean()))
    return pd.DataFrame(rows).set_index("pronostico")


# ----------------------------------------------------------------------------------------
# Temporadas
# ----------------------------------------------------------------------------------------
def final_positions(matches, league: League, comp, syear, tix, mod, ld):
    st = season_state(matches, comp, syear, tix, asof=pd.Timestamp("2100-01-01", tz="UTC"),
                      complete_round_robin=False)
    D = 2 * ld.N + 2
    r = simulate(league, st, np.zeros(D), np.eye(D), mod.params(), ld.N, n_sims=1, mode="plugin")
    return st.teams, r.pos[0], r.pts[0]


def season_backtest(matches, league: League, ld, mod, seasons, fractions=(0.0, 0.25, 0.5, 0.75),
                    n_sims=4000, seed=0):
    comp = league.code
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    d = ld.matches
    plan = []
    for s in seasons:
        reg = matches[(matches["comp"] == comp) & (matches["syear"] == s) & (matches["stage"] == "regular")
                      & matches["finished"]].sort_values("date")
        if len(reg) == 0:
            continue
        for f in fractions:
            k = int(round(f * len(reg)))
            if k == 0:
                # pretemporada: el último partido de la temporada anterior en el filtro
                asof = reg["date"].iloc[0] - pd.Timedelta(seconds=1)
            else:
                asof = reg["date"].iloc[k - 1]
            dn = d["date"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
            j = np.searchsorted(dn, asof.tz_convert("UTC").tz_localize(None).to_datetime64(), side="right") - 1
            plan.append((s, f, asof, j))
    Ms, Ps, _ = mod.run_checkpoints([p[3] for p in plan])
    hp = mod.params()
    rows, mats = [], {}
    for (s, f, asof, j), mvec, Pmat in zip(plan, Ms, Ps):
        if f == 0.0:
            mvec, Pmat = preseason_transition(mvec, Pmat, hp, ld, s)
        st = season_state(matches, comp, s, tix, asof=asof)
        teams, final, fpts = final_positions(matches, league, comp, s, tix, mod, ld)
        fin = dict(zip(teams, final))
        fp = dict(zip(teams, fpts))
        y = np.array([fin[t] for t in st.teams])
        ypts = np.array([fp[t] for t in st.teams])
        n = len(st.teams)
        for mode in ("posterior", "plugin"):
            r = simulate(league, st, mvec, Pmat, hp, ld.N, n_sims=n_sims, mode=mode, seed=seed)
            M = r.position_matrix()
            mats[(s, f, mode)] = (st.teams, M, y)
            Mc = np.clip(M, 1 / (n_sims * 10), 1)
            Mc = Mc / Mc.sum(1, keepdims=True)
            for i, t in enumerate(st.teams):
                rows.append(dict(syear=s, frac=f, mode=mode, team=t, pos=int(y[i]),
                                 pts=float(ypts[i]), xpts=float(r.pts[:, i].mean()), sd_pts=float(r.pts[:, i].std()),
                                 pit=float((r.pts[:, i] < ypts[i]).mean() + 0.5 * (r.pts[:, i] == ypts[i]).mean()),
                                 log=-np.log(Mc[i, y[i]]),
                                 rps=rps(M[i:i + 1], y[i:i + 1])[0],
                                 p_champ=M[i, 0], champ=int(y[i] == 0),
                                 p_top4=M[i, :4].sum(), top4=int(y[i] < 4),
                                 p_rel=M[i, n - _n_rel(league):].sum(), rel=int(y[i] >= n - _n_rel(league))))
    return pd.DataFrame(rows), mats


def _n_rel(league: League) -> int:
    for z in league.zones:
        if z.key == "relegation":
            return len(z.positions)
    return 0


def preseason_transition(m, P, hp, ld, syear):
    """Aplica en numpy la transición entre temporadas (para pronósticos de pretemporada)."""
    N = ld.N
    si = ld.seasons.index(syear)
    act = ld.active[si].astype(float)
    pro = ld.promoted[si].astype(float)
    team = np.r_[np.ones(2 * N), 0.0, 0.0]
    scale = np.where(team > 0, hp["phi"], 1.0)
    m = m * scale
    P = P * scale[:, None] * scale[None, :] + np.diag(team * hp["s_season"] ** 2)
    r = np.r_[pro, pro, 0.0, 0.0]
    prior = np.r_[np.full(N, hp["a_prom"]), np.full(N, hp["d_prom"]), 0.0, 0.0]
    m = np.where(r > 0, prior, m)
    keep = 1 - r
    P = P * keep[:, None] * keep[None, :] + np.diag(r * hp["v_prom"])
    n = max(act.sum(), 1)
    ga = np.r_[act, np.zeros(N), -1.0, 0.0]
    gd = np.r_[np.zeros(N), act, 1.0, 0.0]
    ca = np.r_[act, np.zeros(N + 2)]
    cd = np.r_[np.zeros(N), act, 0.0, 0.0]
    Pi = np.eye(2 * N + 2) - np.stack([ga, gd], 1) @ np.stack([ca, cd], 1).T / n
    return Pi @ m, Pi @ P @ Pi.T


def summarize_seasons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["z2"] = ((df["pts"] - df["xpts"]) / df["sd_pts"].clip(lower=1e-6)) ** 2
    df["in90"] = ((df["pit"] > 0.05) & (df["pit"] < 0.95)).astype(float)
    g = df.groupby(["frac", "mode"])
    out = g.agg(log_pos=("log", "mean"), rps_pos=("rps", "mean"), z2_pts=("z2", "mean"), cover90_pts=("in90", "mean"))
    for ev in ("champ", "top4", "rel"):
        p = df[f"p_{ev}"].clip(1e-4, 1 - 1e-4)
        y = df[ev]
        df[f"brier_{ev}"] = (p - y) ** 2
        df[f"log_{ev}"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        out[f"brier_{ev}"] = g[f"brier_{ev}"].mean()
        out[f"log_{ev}"] = g[f"log_{ev}"].mean()
    return out
