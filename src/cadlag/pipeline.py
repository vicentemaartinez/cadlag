"""Pipeline de pronóstico: datos -> filtro -> simulación -> JSON para la web.

    cadlag forecast            # todas las ligas núcleo
    cadlag forecast chi.1      # una liga

Salida: ``outputs/web/<liga>.json`` y ``outputs/web/index.json``.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import store
from .leagues import CORE_LEAGUES, LEAGUES
from .model.dixon_coles import outcome_probs, predictive_grid
from .model.filter import DynamicDC, prepare
from .sim.season import season_state, simulate

MODELS = store.ROOT / "models"
WEB = store.ROOT / "outputs" / "web"


def current_season(matches: pd.DataFrame, comp: str) -> int:
    """La temporada más reciente con fixture publicado (los pendientes viejos —partidos
    suspendidos que nunca se jugaron— no cuentan)."""
    d = matches[(matches["comp"] == comp) & (matches["stage"] == "regular")]
    return int(d["syear"].max())


def load_or_fit(ld, refit: bool = False, verbose=True) -> DynamicDC:
    MODELS.mkdir(exist_ok=True)
    f = MODELS / f"{ld.comp}_h.npy"
    if f.exists() and not refit:
        return DynamicDC(ld, np.load(f))
    mod = DynamicDC(ld)
    mod.fit(verbose=verbose)
    np.save(f, mod.h)
    return mod


def team_ratings(m, P, ld, teams) -> list[dict]:
    N = ld.N
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    mu = m[2 * N]
    out = []
    for t in teams:
        i = tix[t]
        a, d = m[i], m[N + i]
        va, vd, cad = P[i, i], P[N + i, N + i], P[i, N + i]
        out.append(dict(attack=float(a), defense=float(d), attack_sd=float(np.sqrt(va)),
                        defense_sd=float(np.sqrt(vd)), net=float(a + d),
                        net_sd=float(np.sqrt(va + vd + 2 * cad)),
                        xgf_neutral=float(np.exp(mu + a)), xga_neutral=float(np.exp(mu - d))))
    return out


def upcoming(m, P, hp, ld, st, names, horizon_days=10) -> list[dict]:
    N = ld.N
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    todo = st.todo[st.todo["days_ahead"] <= horizon_days]
    if len(todo) == 0:
        todo = st.todo.head(10)
    rows, Z, SS = [], [], []
    q = np.r_[np.full(2 * N, hp["s_team"] ** 2), hp["s_mu"] ** 2, hp["s_eta"] ** 2]
    for _, r in todo.iterrows():
        h, a = tix[st.teams[r.hi]], tix[st.teams[r.ai]]
        A = np.zeros((2, 2 * N + 2))
        A[0, [h, N + a, 2 * N, 2 * N + 1]] = [1, -1, 1, 1 - float(r.neutral)]
        A[1, [a, N + h, 2 * N]] = [1, -1, 1]
        Pd = P + np.diag(q * r.days_ahead)
        Z.append(A @ m)
        SS.append(A @ Pd @ A.T)
        rows.append(dict(home=st.teams[r.hi], away=st.teams[r.ai], date=str(r.date)))
    if not rows:
        return []
    G = predictive_grid(np.array(Z), np.array(SS), hp["rho"])
    pr = outcome_probs(G)
    for k, row in enumerate(rows):
        g = G[k]
        best = np.unravel_index(np.argmax(g), g.shape)
        row.update(p_home=float(pr[k, 0]), p_draw=float(pr[k, 1]), p_away=float(pr[k, 2]),
                   xg_home=float((g.sum(1) * np.arange(g.shape[0])).sum()),
                   xg_away=float((g.sum(0) * np.arange(g.shape[1])).sum()),
                   likely_score=[int(best[0]), int(best[1])], p_likely=float(g[best]),
                   home_name=names.get(row["home"]), away_name=names.get(row["away"]))
    return rows


def standings(st, names) -> pd.DataFrame:
    n = len(st.teams)
    t = pd.DataFrame({"team": st.teams, "name": [names.get(x, x) for x in st.teams]})
    for c in ("pj", "g", "e", "p", "gf", "gc", "pts"):
        t[c] = 0
    for h, a, x, y in st.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
        for i, gf, gc in ((h, x, y), (a, y, x)):
            t.loc[i, "pj"] += 1
            t.loc[i, "gf"] += gf
            t.loc[i, "gc"] += gc
            t.loc[i, "g"] += gf > gc
            t.loc[i, "e"] += gf == gc
            t.loc[i, "p"] += gf < gc
            t.loc[i, "pts"] += 3 * (gf > gc) + (gf == gc)
    if st.adj is not None:
        t["pts"] = t["pts"] + st.adj.astype(int)
        t["adj"] = st.adj.astype(int)
    t["dg"] = t["gf"] - t["gc"]
    return t


def forecast_league(comp: str, matches=None, teams=None, n_sims=20000, refit=False, seed=0) -> dict:
    if matches is None:
        matches, teams = store.load()
    names = dict(zip(teams["team_id"], teams["name"]))
    league = LEAGUES[comp]
    from .model.engine import Engine
    eng = Engine(matches, comp, refit=refit)
    ld = eng.ld
    hp = eng.params()
    mT, PT = eng.run()
    syear = current_season(matches, comp)
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    st = season_state(matches, comp, syear, tix)
    if len(st.done) == 0:  # pretemporada
        mT, PT = eng.preseason(mT, PT, syear)
    res = simulate(league, st, mT, PT, hp, ld.N, n_sims=n_sims, mode="posterior", seed=seed, keep_rates=True)
    M = res.position_matrix()
    tab = standings(st, names)
    ratings = eng.ratings(mT, PT, st.teams)
    n = len(st.teams)
    # --- análisis: importancia, cotas matemáticas, "qué necesita" ---------------------------
    from .sim import analysis as an
    n_rel = next((len(z.positions) for z in league.zones if z.key == "relegation"), 0)
    imp = an.match_importance(res, n_rel, top=8) if len(st.todo) else pd.DataFrame()
    decisive = [dict(home=st.teams[int(r.home)], away=st.teams[int(r.away)],
                     home_name=names.get(st.teams[int(r.home)]), away_name=names.get(st.teams[int(r.away)]),
                     date=str(r.date), p_home=float(r.p_home), p_draw=float(r.p_draw), p_away=float(r.p_away),
                     mi_title=float(r.mi_title), mi_relegation=float(r.mi_relegation),
                     mi_promotion=float(r.mi_promotion))
                for r in imp.itertuples()] if len(imp) else []
    try:
        bounds = an.position_bounds(st).set_index("team")
    except Exception:  # el MIP es un extra: si falla, el pronóstico sigue
        bounds = None
    safe_pos = list(range(1, n - n_rel + 1))
    X = res.extra   # formato argentino: campeón del torneo en curso, zonas y descensos
    if X is not None:
        from .sim.split import tournament_table_now
        tpts, tpj = tournament_table_now(st)
    champ_entropy = an.entropy_bits(X["title"].mean(0) if X is not None else M[:, 0])
    rows = []
    for i, t in enumerate(st.teams):
        r = tab.iloc[i].to_dict()
        needs = {}
        for key, pc in an.need_curves(res, st, i, n_rel, league, mT, PT, hp, ld.N, seed=seed).items():
            needs[key] = dict(points_for_50=pc.attrs.get("need_50"), points_for_90=pc.attrs.get("need_90"),
                              curve=[[int(g), round(float(q), 4), round(float(pr), 5)] for g, q, pr in
                                     zip(pc["points_gained"], pc["p_objective"], pc["p_reach"])])
        if bounds is not None and t in bounds.index:
            bp, wp = bounds.loc[t, "best_position"], bounds.loc[t, "worst_position"]
            needs["best_position"], needs["worst_position"] = int(bp), int(wp)
            needs["title_clinched"] = bool(wp == 1) and X is None
            needs["title_eliminated"] = bool(bp > 1) and X is None
            if n_rel and X is None:
                needs["relegated"] = bool(bp > n - n_rel)
                needs["safe"] = bool(wp <= n - n_rel)
        zones = {z.key: float(M[i, [p - 1 for p in z.positions]].sum()) for z in league.zones}
        if X is not None:
            zones["title"] = float(X["title"][:, i].mean())
            zones["playoffs"] = float((X["zpos"][:, i] < 8).mean())
            zones["relegated"] = float(X["rel"][:, i].mean())
            if "lib" in X:
                zones["libertadores"] = float(X["lib"][:, i].mean())
                zones["sudamericana"] = float(X["sud"][:, i].mean())
        if res.promoted is not None:   # ascenso total: directo + playoffs
            zones["promoted"] = float(res.promoted[:, i].mean())
            zones["playoff_win"] = float((res.po_winner == i).mean())
        rows.append(dict(id=t, name=names.get(t, t), played=int(r["pj"]), won=int(r["g"]), drawn=int(r["e"]),
                         lost=int(r["p"]), gf=int(r["gf"]), ga=int(r["gc"]), points=int(r["pts"]),
                         adj=int(r.get("adj", 0)),
                         remaining=int(((st.todo["hi"] == i) | (st.todo["ai"] == i)).sum()),
                         **({"zone": "AB"[int(X["zone"][i])], "tpts": int(tpts[i]), "tpj": int(tpj[i])} if X is not None else {}),
                         xpts=float(res.pts[:, i].mean()),
                         pts_q=[float(x) for x in np.quantile(res.pts[:, i], [0.05, 0.5, 0.95])],
                         position_probs=[float(x) for x in M[i]],
                         expected_position=float((M[i] * np.arange(1, n + 1)).sum()),
                         zones=zones, rating=ratings[i], needs=needs))
    rows.sort(key=lambda r: (-r["points"], -(r["gf"] - r["ga"]), -r["gf"]))
    last_played = st.done["date"].max() if len(st.done) else None
    return dict(
        comp=comp, name=league.name, country=league.country, season=syear,
        season_label=str(syear) if league.season_style == "calendar" else f"{syear}-{str(syear + 1)[2:]}",
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        last_result=str(last_played) if last_played is not None else None,
        matches_played=int(len(st.done)), matches_total=int(len(st.done) + len(st.todo)),
        n_sims=n_sims, zones=[dict(key=z.key, label=z.label, positions=list(z.positions)) for z in league.zones],
        teams=rows, upcoming=eng.upcoming(mT, PT, st, names), decisive_matches=decisive,
        suspense=dict(champion_entropy_bits=champ_entropy, champion_max_bits=float(np.log2(n))),
        model=dict(kind=eng.kind, params=hp, **eng.levels(mT)),
    )


def forecast(comps=None, n_sims=20000, refit=False, cups=None) -> list[dict]:
    from .cup import CUPS, forecast_cup
    comps = CORE_LEAGUES if comps is None else comps
    cups = list(CUPS) if cups is None else cups
    matches, teams = store.load()
    WEB.mkdir(parents=True, exist_ok=True)
    index = []
    for c in list(comps) + list(cups):
        try:
            f = (forecast_cup(c, matches, teams, n_sims=n_sims) if c in CUPS
                 else forecast_league(c, matches, teams, n_sims=n_sims, refit=refit))
        except Exception as e:   # una copa sin fixture (fuera de temporada) no frena las ligas
            if c in CUPS:
                print(f"{c}: sin pronóstico ({e})", flush=True)
                continue
            raise
        (WEB / f"{c}.json").write_text(json.dumps(f, ensure_ascii=False, default=float))
        index.append({k: f[k] for k in ("comp", "name", "country", "season_label", "generated_at",
                                        "matches_played", "matches_total")})
        print(f"{c}: {f['matches_played']}/{f['matches_total']} partidos, simulado")
    (WEB / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    return index
