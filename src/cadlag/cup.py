"""Pronóstico de copas (por ahora la Champions League): capa europea -> simulación -> JSON.

Usa las fuerzas domésticas del modelo oficial de las cinco grandes ligas (recalculadas en cada
corrida), la capa europea (``model.europe``, hiperparámetros en ``models/europe_h.npy``) y el
simulador del formato con fase liga (``sim.ucl``). Escribe ``outputs/web/<copa>.json`` con la misma
forma que una liga, para que el sitio la muestre con la misma vista, y guarda las simulaciones para
el laboratorio (``export``).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .data import store
from .model import europe as eu
from .sim import ucl
from .sim.analysis import _mi_bits, _pava, _rank, _shrunk_rate, entropy_bits

CUPS = {"uefa.champions": dict(name="Champions League", country="Europa")}
MODELS = store.ROOT / "models"
ZONES = [dict(key="r16", label="Octavos", positions=list(range(1, 9))),
         dict(key="playoff", label="Playoff", positions=list(range(9, 25))),
         dict(key="out", label="Eliminado", positions=list(range(25, 37)))]
# etapa alcanzada (sim.ucl.knockout) -> columnas "llega al menos a"
REACH = {"r16": 2, "qf": 3, "sf": 4, "final": 5, "champion": 6}
_CACHE: dict = {}
FEW_DATA = 12   # partidos UEFA en tres temporadas bajo los cuales un equipo sin liga modelada lleva asterisco


def current_cup_season(matches: pd.DataFrame, comp: str) -> int:
    d = matches[(matches["comp"] == comp) & (matches["slug"].astype(str) == "league-phase")]
    return int(d["syear"].max())


def table_now(cs: ucl.CupState) -> pd.DataFrame:
    n = len(cs.teams)
    t = pd.DataFrame(0, index=range(n), columns=["pj", "g", "e", "p", "gf", "gc", "agf", "aw", "pts"])
    for h, a, x, y in cs.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
        for i, gf, gc, away in ((int(h), x, y, 0), (int(a), y, x, 1)):
            t.loc[i, "pj"] += 1; t.loc[i, "gf"] += gf; t.loc[i, "gc"] += gc
            t.loc[i, "g"] += gf > gc; t.loc[i, "e"] += gf == gc; t.loc[i, "p"] += gf < gc
            t.loc[i, "pts"] += 3 * (gf > gc) + (gf == gc)
            if away:
                t.loc[i, "agf"] += gf; t.loc[i, "aw"] += gf > gc
    t["dg"] = t["gf"] - t["gc"]
    return t


def strengths(matches, comp, cs, date, dom=None):
    """Fuerzas de los equipos de la copa en ``date`` con los hiperparámetros de producción."""
    if dom is None:
        dom = eu.domestic_states(matches, "prod")
    ed = eu.build(matches, dom, upto=date)
    ef = eu.EuroFilter(ed, eu.pad(np.load(MODELS / "europe_h.npy")))
    final, _ = ef.run()
    return eu.strengths_at(ef, final, dom, cs.teams, date, comp), ef, final


def need_curves(res: dict, cs: ucl.CupState, team: int, rng) -> dict:
    """'¿Qué necesita?' en la fase liga: P(top 8), P(top 24) y P(campeón) según los puntos que sume,
    con el mismo remuestreo condicional exacto que las ligas (``analysis.need_curves``): en cada
    simulación se vuelven a sortear solo sus partidos, con P_θ(o) ∝ P(o | fuerzas)·e^{θ·puntos}, sus
    rivales ganan o pierden esos mismos partidos y la tabla se reordena por puntos (desempate: el
    orden de esa simulación). Campeón: P(campeón | equipo, puesto) de todas las simulaciones."""
    hi, ai = cs.todo["hi"].to_numpy(), cs.todo["ai"].to_numpy()
    ks = np.where((hi == team) | (ai == team))[0]
    if not len(ks):
        return {}
    n = len(cs.teams); S = res["pts"].shape[0]
    opp = np.where(hi[ks] == team, ai[ks], hi[ks]); home = hi[ks] == team
    x, y = res["gx"][:, ks], res["gy"][:, ks]
    o0 = np.where(x > y, 0, np.where(x == y, 1, 2))
    PT = np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    mine0 = np.where(home[None, :], PT[0][o0], PT[1][o0]); theirs0 = 3.0 * (o0 != 1) + (o0 == 1) - mine0
    base = res["pts"].astype(float).copy()
    base[:, team] -= mine0.sum(1)
    np.subtract.at(base.T, opp, theirs0.T)
    p = res["q"][:, ks].astype(float)
    ptm = np.where(home[None, :, None], PT[0][None, None, :], PT[1][None, None, :])
    q_champ = _shrunk_rate(res["stage"] == 6, res["pos"], team, n)
    G_ = 3 * len(ks) + 1
    acc = {k: [np.zeros(G_), np.zeros(G_)] for k in ("top8", "top24", "champion")}
    for th in (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0):
        q = p * np.exp(th * ptm); q /= q.sum(-1, keepdims=True)
        o = (rng.random(q.shape[:2])[..., None] > np.cumsum(q, -1)).sum(-1)
        mine = np.where(home[None, :], PT[0][o], PT[1][o]); theirs = 3.0 * (o != 1) + (o == 1) - mine
        pts = base.copy(); pts[:, team] += mine.sum(1)
        np.add.at(pts.T, opp, theirs.T)
        g = mine.sum(1).astype(int)
        pos = _rank(pts, res["pos"])[:, team]
        for k, h in (("top8", pos < 8), ("top24", pos < 24), ("champion", q_champ[pos])):
            acc[k][0] += np.bincount(g, minlength=G_); acc[k][1] += np.bincount(g, weights=np.asarray(h, float), minlength=G_)
    now = table_now(cs)["pts"].to_numpy()[team]
    direct = np.bincount(np.clip(np.rint(res["pts"][:, team] - now).astype(int), 0, G_ - 1), minlength=G_) / S
    out = {}
    gg = np.arange(G_)
    for k, (cnt, hs) in acc.items():
        pr = np.where(cnt > 0, hs / np.maximum(cnt, 1), np.nan); ok = ~np.isnan(pr)
        pr = np.interp(gg, gg[ok], pr[ok]) if ok.any() else np.zeros(G_)
        pr = np.clip(_pava(pr, np.maximum(cnt, 1e-3)), 0, 1)
        n50 = next((int(v) for v, q in zip(gg, pr) if q >= 0.5), None); n90 = next((int(v) for v, q in zip(gg, pr) if q >= 0.9), None)
        out[k] = dict(points_for_50=n50, points_for_90=n90,
                      curve=[[int(v), round(float(q), 4), round(float(r), 5)] for v, q, r in zip(gg, pr, direct)])
    return out


def forecast_cup(comp: str = "uefa.champions", matches=None, teams=None, n_sims: int = 20000, seed: int = 0,
                 dom=None, now=None) -> dict:
    if matches is None:
        matches, teams = store.load()
    names = dict(zip(teams["team_id"], teams["name"]))
    syear = current_cup_season(matches, comp)
    cs = ucl.cup_state(matches, comp, syear, names)
    now = now or pd.Timestamp.now(tz="UTC")
    st, ef, final = strengths(matches, comp, cs, now, dom)
    res = ucl.simulate_cup(cs, st.draw, S=n_sims, seed=seed, r=st.r)
    _CACHE[comp] = (cs, st, res)
    n = len(cs.teams); S = n_sims
    pos, stage = res["pos"], res["stage"]
    M = np.stack([np.bincount(pos[:, i], minlength=n) / S for i in range(n)])
    tab = table_now(cs)
    rng = np.random.default_rng(seed)
    hp = ef.params()
    tb = st.table(names)
    mu = float(st.mg[0])
    # pocos datos: sin liga modelada y con pocos partidos UEFA en las últimas tres temporadas
    em = ef.ed.matches[(ef.ed.matches["season"] >= syear - 2) & (ef.ed.arr["w"] > 0)]
    n_uefa = pd.concat([em["home"], em["away"]]).value_counts()
    rows = []
    for i, t in enumerate(cs.teams):
        r = tab.iloc[i]
        zones = {z["key"]: float(M[i, [p - 1 for p in z["positions"]]].sum()) for z in ZONES}
        zones.update({k: float((stage[:, i] >= v).mean()) for k, v in REACH.items()})
        sd = np.sqrt(np.diag(st.P[i]))
        att, dfn = float(tb.att[i]), float(tb.dfn[i])
        rows.append(dict(id=t, name=names.get(t, t), played=int(r.pj), won=int(r.g), drawn=int(r.e), lost=int(r.p),
                         gf=int(r.gf), ga=int(r.gc), points=int(r.pts),
                         remaining=int(((cs.todo["hi"] == i) | (cs.todo["ai"] == i)).sum()),
                         xpts=float(res["pts"][:, i].mean()),
                         pts_q=[float(v) for v in np.quantile(res["pts"][:, i], [0.05, 0.5, 0.95])],
                         position_probs=[float(v) for v in M[i]],
                         expected_position=float((M[i] * np.arange(1, n + 1)).sum()),
                         stage_probs=[float((stage[:, i] == k).mean()) for k in range(7)],
                         zones=zones,
                         rating=dict(attack=att, defense=dfn, attack_sd=float(sd[0]), defense_sd=float(sd[1]),
                                     net=att + dfn, net_sd=float(tb.sd_net[i]),
                                     xgf_neutral=float(np.exp(mu + att)), xga_neutral=float(np.exp(mu - dfn)),
                                     league=eu.DOMESTIC[st.L[i]] if st.core[i] else None,
                                     n_uefa=int(n_uefa.get(t, 0))),
                         few_data=bool(not st.core[i] and n_uefa.get(t, 0) < FEW_DATA),
                         needs=need_curves(res, cs, i, rng)))
    # orden de la tabla: puntos, diferencia, goles, goles de visita, victorias, victorias de visita
    order = sorted(range(n), key=lambda i: tuple(-tab.iloc[i][c] for c in ("pts", "dg", "gf", "agf", "g", "aw")))
    rows = [rows[i] for i in order]
    # partidos decisivos de la fase liga: campeón y quién queda fuera (24 primeros)
    gx, gy = res["gx"], res["gy"]
    champ = stage.argmax(1)
    out24 = pos >= 24
    dec = []
    for k in range(min(len(cs.todo), 40)):
        o = np.where(gx[:, k] > gy[:, k], 0, np.where(gx[:, k] == gy[:, k], 1, 2))
        i_t = _mi_bits(o, champ, 3, n)
        i_o = sum(_mi_bits(o, out24[:, t].astype(int), 3, 2) for t in range(n))
        i_8 = sum(_mi_bits(o, (pos[:, t] < 8).astype(int), 3, 2) for t in range(n))
        pr = np.bincount(o, minlength=3) / S
        row = cs.todo.iloc[k]
        h, a = cs.teams[int(row.hi)], cs.teams[int(row.ai)]
        dec.append(dict(home=h, away=a, home_name=names.get(h), away_name=names.get(a), date=str(row.date),
                        p_home=float(pr[0]), p_draw=float(pr[1]), p_away=float(pr[2]),
                        mi_title=float(i_t), mi_relegation=float(i_o), mi_promotion=float(i_8)))
    dec.sort(key=lambda d: -(d["mi_title"] + d["mi_relegation"]))
    last = cs.done["date"].max() if len(cs.done) else None
    offsets = {c: dict(attack=float(st.mg[2 + j]), defense=float(st.mg[2 + eu.NOFF + j]))
               for j, c in enumerate(eu.OFFSET_LEAGUES)}
    total = len(cs.done) + len(cs.todo)
    return dict(
        comp=comp, name=CUPS[comp]["name"], country=CUPS[comp]["country"], season=syear,
        season_label=f"{syear}-{str(syear + 1)[2:]}", cup=True,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        last_result=str(last) if last is not None else None,
        matches_played=int(len(cs.done)), matches_total=int(total), n_sims=n_sims, zones=ZONES,
        teams=rows, upcoming=[], decisive_matches=dec[:8],
        suspense=dict(champion_entropy_bits=entropy_bits((stage == 6).mean(0)), champion_max_bits=float(np.log2(n))),
        model=dict(kind="europa", params={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in hp.items()},
                   home_advantage=float(np.exp(st.mg[1])), goal_level=float(np.exp(mu)), offsets=offsets),
    )


def sims_blob(comp: str, K: int = 10):
    """Bytes por simulación: o[K] sqrtq[3K] pts[n] pos[n] stage[n] (ver ``export``)."""
    cs, st, res = _CACHE[comp]
    S = res["pos"].shape[0]
    k = min(K, len(cs.todo))
    gx, gy = res["gx"][:, :k], res["gy"][:, :k]
    O = np.where(gx > gy, 0, np.where(gx == gy, 1, 2)).astype(np.uint8)
    Q = np.clip(np.round(np.sqrt(res["q"][:, :k]) * 255), 0, 255).astype(np.uint8).reshape(S, 3 * k)
    parts = [O, Q, np.clip(res["pts"], 0, 255).astype(np.uint8), res["pos"].astype(np.uint8), res["stage"].astype(np.uint8)]
    fixtures = []
    for j in range(k):
        row = cs.todo.iloc[j]
        fixtures.append(dict(home=cs.teams[int(row.hi)], away=cs.teams[int(row.ai)], date=str(row.date)[:16],
                             p=[float((O[:, j] == c).mean()) for c in range(3)]))
    return np.concatenate(parts, axis=1).tobytes(), k, cs.teams, fixtures
