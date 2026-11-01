"""Pronóstico de la Libertadores y la Sudamericana: capa sudamericana -> simulación conjunta -> JSON."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import store
from .model import europe as eu
from .sim import conmebol as cm

MODELS = store.ROOT / "models"


def strengths_for(matches, dom, st: cm.ConmebolState, date, h=None, max_days: int = 120):
    """Fuerzas de todos los equipos de las dos copas en ``date``. El prior de un equipo sin historia
    depende de la copa que juega (Libertadores o Sudamericana)."""
    h = eu.pad(np.load(MODELS / eu.SOUTH.hyper_file) if h is None else np.asarray(h))
    ed = eu.build(matches, dom, max_days=max_days, upto=date, region=eu.SOUTH)
    ef = eu.EuroFilter(ed, h)
    final, _ = ef.run()
    lib = set()
    if cm.LIB in st.cups:
        for g in st.cups[cm.LIB].groups:
            lib |= {st.teams[i] for i in g}
    sl = eu.strengths_at(ef, final, dom, st.teams, date, cm.LIB, max_days=max_days)
    if cm.SUD in st.cups:
        ss = eu.strengths_at(ef, final, dom, st.teams, date, cm.SUD, max_days=max_days)
        for i, t in enumerate(st.teams):
            if t not in lib:
                sl.m[i], sl.P[i] = ss.m[i], ss.P[i]
    sl.ef, sl.final = ef, final
    return sl


NAMES = {cm.LIB: "Copa Libertadores", cm.SUD: "Copa Sudamericana"}
REACH = {cm.LIB: {"r16": 1, "qf": 2, "sf": 3, "final": 4, "champion": 5},
         cm.SUD: {"playoff": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5, "champion": 6}}
ROUND_NAMES = {"knockout-round-playoffs": "Playoff", "round-of-16": "Octavos", "quarterfinals": "Cuartos",
               "semifinals": "Semifinal", "final": "Final"}


def current_syear(matches) -> int:
    d = matches[(matches["comp"] == cm.LIB) & (matches["slug"].astype(str) == "group-stage")]
    return int(d["syear"].max())


def _group_table(cup: cm.Cup, i: int):
    pj = g = e = p = gf = gc = pts = 0
    for h, a, x, y in cup.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
        if i not in (h, a):
            continue
        f, c = (x, y) if h == i else (y, x)
        pj += 1; gf += f; gc += c; g += f > c; e += f == c; p += f < c; pts += 3 * (f > c) + (f == c)
    return dict(played=int(pj), won=int(g), drawn=int(e), lost=int(p), gf=int(gf), ga=int(gc), points=int(pts))


SIMS_FILE = "conmebol.sims.gz"


LAB_S = 10000   # simulaciones que van al sitio (~2 MB comprimidas)


def lab_blob(st: cm.ConmebolState, res: dict, S_lab: int = LAB_S) -> tuple[bytes, dict]:
    """Simulaciones para el laboratorio del sitio, compartidas por las dos copas. Por simulación:
    para cada copa, o[K] y 255·√q[3K] de sus próximos partidos de grupo; luego, para los n equipos
    (unión de las dos copas), pts[n] (puntos en su grupo), gpos[n] (puesto en su grupo, 0-3),
    lib[n] y sud[n] (etapa alcanzada en cada copa + 1; 0 = no la juega)."""
    n = len(st.teams)
    comps = [c for c in (cm.LIB, cm.SUD) if c in st.cups]
    res = {c: {k: v[:S_lab] for k, v in r.items()} for c, r in res.items()}
    S = res[comps[0]]["stage"].shape[0]
    parts, meta = [], dict(S=S, n=n, order=list(st.teams), cups={}, team_cup=[None] * n, team_group=[None] * n)
    off = 0
    for c in comps:
        o, q = res[c]["o"], res[c]["q"]
        K = o.shape[1]
        parts += [o.astype(np.uint8), np.clip(np.round(np.sqrt(q) * 255), 0, 255).astype(np.uint8).reshape(S, 3 * K)]
        meta["cups"][c] = dict(K=K, off=off)
        off += 4 * K
    pts = np.zeros((S, n)); gpos = np.zeros((S, n), np.uint8)
    for ci, c in enumerate(comps):
        cup = st.cups[c]; gp = res[c]["groups"]
        for g, idx in enumerate(cup.groups):
            for i in idx:
                pts[:, i] = res[c]["pts"][:, i]
                meta["team_cup"][i] = ci; meta["team_group"][i] = g
            for k in range(gp.shape[2]):
                gpos[np.arange(S), gp[:, g, k]] = k
    stages = [(res[c]["stage"].astype(np.int16) + 1).astype(np.uint8) if c in res else np.zeros((S, n), np.uint8)
              for c in (cm.LIB, cm.SUD)]
    parts += [np.clip(pts, 0, 255).astype(np.uint8), gpos, *stages]
    meta["W"] = off + 4 * n
    meta["base"] = off
    return np.concatenate(parts, axis=1).tobytes(), meta


def forecast(matches=None, teams=None, n_sims: int = 20000, seed: int = 0, dom=None, now=None) -> dict:
    """Pronóstico de las dos copas: {comp: JSON del sitio}. Deja además las simulaciones para el
    laboratorio en ``outputs/web/conmebol.sims.gz``."""
    if matches is None:
        matches, teams = store.load()
    names = dict(zip(teams["team_id"], teams["name"]))
    syear = current_syear(matches)
    now = now or pd.Timestamp.now(tz="UTC")
    st = cm.conmebol_state(matches, syear, names, asof=now if now < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1) else None)
    if dom is None:
        dom = eu.domestic_states(matches, "prod", region=eu.SOUTH)
    strg = strengths_for(matches, dom, st, now)
    res = cm.simulate(st, strg.draw, S=n_sims, seed=seed, r=strg.r)
    tb = strg.table(names)
    mu = float(strg.mg[0])
    out = {}
    import datetime as dt
    for comp, cup in st.cups.items():
        stg, gp = res[comp]["stage"], res[comp]["groups"]
        K = len(cm.STAGES[comp])
        members = sorted({i for g in cup.groups for i in g} | {int(i) for i in np.where((stg >= 0).any(0))[0]})
        rows = []
        for i in members:
            gi = next((k for k, g in enumerate(cup.groups) if i in g), None)
            gpos = [float((gp[:, gi, k] == i).mean()) for k in range(4)] if gi is not None else None
            sd = np.sqrt(np.diag(strg.P[i]))
            rows.append(dict(
                id=st.teams[i], name=names.get(st.teams[i], st.teams[i]),
                group="ABCDEFGH"[gi] if gi is not None else None, group_probs=gpos,
                entry=float((stg[:, i] >= 0).mean()),
                stage_probs=[float((stg[:, i] == k).mean()) for k in range(K)],
                zones={k: float((stg[:, i] >= v).mean()) for k, v in REACH[comp].items()},
                **(_group_table(cup, i) if gi is not None else dict(played=0, won=0, drawn=0, lost=0, gf=0, ga=0, points=0)),
                rating=dict(attack=float(tb.att[i]), defense=float(tb.dfn[i]), attack_sd=float(sd[0]), defense_sd=float(sd[1]),
                            net=float(tb.net[i]), net_sd=float(tb.sd_net[i]),
                            xgf_neutral=float(np.exp(mu + tb.att[i])), xga_neutral=float(np.exp(mu - tb.dfn[i])),
                            league=eu.SOUTH.domestic[strg.L[i]] if strg.core[i] else None)))
        # llaves con cruce conocido y todavía abiertas (o recién jugadas): P(pasa) de cada lado
        ties = []
        for rnd in cm.KO[comp]:
            known = cm._known_pairs(cup, rnd)
            if not known:
                continue
            for u, v in known:
                legs = cm._legs(cup, rnd, u, v)
                done = all(l is not None for l in legs[: (1 if rnd == "final" and syear >= 2019 else 2)])
                pu = float((stg[:, u] > stg[:, v]).mean() + 0.5 * (stg[:, u] == stg[:, v]).mean())
                ties.append(dict(round=ROUND_NAMES.get(rnd, rnd), home=st.teams[u], away=st.teams[v],
                                 home_name=names.get(st.teams[u]), away_name=names.get(st.teams[v]),
                                 legs=[list(map(int, l)) if l is not None else None for l in legs], done=bool(done),
                                 p_home=pu))
        champ = (stg == K - 1).mean(0)
        # final con los dos finalistas ya definidos aunque ESPN todavía la tenga como "por definir"
        fin = np.where((stg >= K - 2).mean(0) > 0.999)[0]
        if len(fin) == 2 and not any(t["round"] == "Final" for t in ties):
            u, v = int(fin[0]), int(fin[1])
            ties.append(dict(round="Final", home=st.teams[u], away=st.teams[v], home_name=names.get(st.teams[u]),
                             away_name=names.get(st.teams[v]), legs=[None], done=False, p_home=float(champ[u])))
        total = len(cup.done) + len(cup.todo)
        out[comp] = dict(
            comp=comp, name=NAMES[comp], country="Sudamérica", season=syear, season_label=str(syear), cup=True, kind="groups",
            generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            last_result=str(cup.done["date"].max()) if len(cup.done) else None,
            matches_played=int(len(cup.done)), matches_total=int(total), n_sims=n_sims,
            stages=cm.STAGES[comp], reach=list(REACH[comp]), teams=rows, ties=ties,
            groups=[dict(name="ABCDEFGH"[k], teams=[st.teams[i] for i in g]) for k, g in enumerate(cup.groups)],
            suspense=dict(champion_entropy_bits=float(-(champ[champ > 0] * np.log2(champ[champ > 0])).sum()),
                          champion_max_bits=float(np.log2(32))),
            model=dict(kind="sudamerica", home_advantage=float(np.exp(strg.mg[1])), goal_level=float(np.exp(mu)),
                       offsets={c: dict(attack=float(strg.mg[2 + j]), defense=float(strg.mg[2 + eu.SOUTH.noff + j]))
                                for j, c in enumerate(eu.SOUTH.offset_leagues)}))
    if not any(len(c.todo) for c in st.cups.values()):
        return out          # grupos terminados: sin laboratorio
    import gzip
    blob, meta = lab_blob(st, res)
    (store.ROOT / "outputs" / "web").mkdir(parents=True, exist_ok=True)
    (store.ROOT / "outputs" / "web" / SIMS_FILE).write_bytes(gzip.compress(blob, 9))
    for ci, comp in enumerate([c for c in (cm.LIB, cm.SUD) if c in st.cups]):
        cup = st.cups[comp]; K = meta["cups"][comp]["K"]
        o = res[comp]["o"]
        fx = [dict(home=st.teams[int(r.hi)], away=st.teams[int(r.ai)], hi=int(r.hi), ai=int(r.ai), date=str(r.date)[:16],
                   p=[float((o[:, k] == c).mean()) for c in range(3)]) for k, r in enumerate(cup.todo.head(K).itertuples())]
        out[comp]["sims"] = dict(file=SIMS_FILE, S=meta["S"], n=meta["n"], W=meta["W"], base=meta["base"], order=meta["order"],
                                 cup_index=ci, K=K, off=meta["cups"][comp]["off"], team_cup=meta["team_cup"],
                                 team_group=meta["team_group"], fixtures=fx)
    return out
