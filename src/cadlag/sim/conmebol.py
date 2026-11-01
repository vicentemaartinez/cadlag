r"""Copas de la Conmebol: Libertadores y Sudamericana, simuladas juntas.

Formato (desde 2023 en la Sudamericana; la Libertadores lo tiene desde 2017):

  * **Grupos**: ocho de cuatro a dos ruedas en cada copa. Desempate hasta 2025: puntos, diferencia
    de gol, goles, goles de visita; desde 2026 primero la minitabla (puntos, diferencia y goles entre
    los empatados) y después diferencia y goles totales. Tarjetas y sorteo al final (sorteo aquí).
  * **Libertadores**: los dos primeros de cada grupo van a octavos; el tercero, al playoff de la
    Sudamericana. Octavos: primeros contra segundos, sorteados sin restricciones, con la vuelta en
    cancha del primero.
  * **Sudamericana**: el primero de cada grupo va a octavos; el segundo juega el playoff contra un
    tercero de la Libertadores (vuelta en cancha del tercero de la Libertadores, que viene de una copa
    más exigente) y los ocho ganadores enfrentan en octavos a los primeros (vuelta en cancha del
    primero).
  * Cuartos y semis: el cuadro sale del sorteo de octavos; mientras no esté en los datos se sortea.
    La vuelta la juega de local el de mejor campaña (puntos, diferencia y goles en todo el torneo).
  * Las llaves a dos partidos no tienen alargue: global igualado, penales (moneda). Final única en
    cancha neutral, con alargue y penales. Hasta 2018 la final fue a ida y vuelta.

Las fases previas no se simulan: el pronóstico empieza cuando los grupos están completos. Lo ya
jugado se respeta, incluidas las llaves que se definieron por penales (``data/manual/shootouts.csv``,
porque ESPN no trae la tanda).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import store
from ..leagues import CRIT_CODE
from . import tiebreak
from .ucl import _nb, _rate, outcome_probs, two_legs

LIB, SUD = "conmebol.libertadores", "conmebol.sudamericana"
KO = {LIB: ["round-of-16", "quarterfinals", "semifinals", "final"],
      SUD: ["knockout-round-playoffs", "round-of-16", "quarterfinals", "semifinals", "final"]}
STAGES = {LIB: ["Grupos", "Octavos", "Cuartos", "Semifinal", "Final", "Campeón"],
          SUD: ["Grupos", "Playoff", "Octavos", "Cuartos", "Semifinal", "Final", "Campeón"]}
SHOOTOUTS = store.ROOT / "data" / "manual" / "shootouts.csv"


def _shootout_winners() -> pd.DataFrame:
    if not SHOOTOUTS.exists():
        return pd.DataFrame(columns=["comp", "date", "winner"])
    return pd.read_csv(SHOOTOUTS, comment="#")


@dataclass
class Cup:
    comp: str
    syear: int
    groups: list                    # listas de índices globales (en ``ConmebolState.teams``)
    done: pd.DataFrame              # grupos jugados: hi, ai, hg, ag
    todo: pd.DataFrame              # grupos por jugar: hi, ai, date
    ko: dict = field(default_factory=dict)   # ronda -> partidos (hi, ai, hg, ag, finished, date, id)


@dataclass
class ConmebolState:
    syear: int
    teams: list                     # unión de equipos de ambas copas
    cups: dict                      # comp -> Cup
    pen: dict = field(default_factory=dict)   # (comp, ronda, frozenset(u, v)) -> ganador por penales


def _groups_from(gs: pd.DataFrame) -> list:
    adj: dict = {}
    for h, a in zip(gs["hi"], gs["ai"]):
        adj.setdefault(h, set()).add(a); adj.setdefault(a, set()).add(h)
    seen, out = set(), []
    for i in sorted(adj):
        if i in seen:
            continue
        comp_, st = [], [i]
        while st:
            j = st.pop()
            if j in seen:
                continue
            seen.add(j); comp_.append(j); st += list(adj[j])
        out.append(sorted(comp_))
    return out


def conmebol_state(matches: pd.DataFrame, syear: int, names: dict, asof=None, comps=(LIB, SUD)) -> ConmebolState:
    d = matches[matches["comp"].isin(comps) & (matches["syear"] == syear)].copy()
    d["slug"] = d["slug"].astype(str).replace({"finals": "final"})
    if asof is not None:
        d.loc[d["date"] > asof, "finished"] = False
    gs = d[d["slug"] == "group-stage"]
    ids = {v: k for k, v in names.items()}
    teams = sorted(set(gs["home"]) | set(gs["away"]))
    tix = {t: i for i, t in enumerate(teams)}
    sh = _shootout_winners()
    cups, pen = {}, {}
    for comp in comps:
        g = gs[gs["comp"] == comp].copy()
        if not len(g):
            continue
        g["hi"] = g["home"].map(tix); g["ai"] = g["away"].map(tix)
        dc = d[d["comp"] == comp]
        ko, prev_end = {}, g["date"].max()
        for rnd in KO[comp]:
            k = dc[dc["slug"] == rnd].copy()
            end = k["date"].max() if len(k) else None
            if asof is not None and (prev_end is None or asof < prev_end):
                k = k.iloc[0:0]          # en un backtest, los cruces se conocen al terminar la ronda anterior
            prev_end = end if end is not None else prev_end
            k = k[k["home"].isin(tix) & k["away"].isin(tix)]
            if len(k):
                k["hi"] = k["home"].map(tix); k["ai"] = k["away"].map(tix)
                ko[rnd] = k.sort_values("date")
                for r in k[k["finished"] & (k["status"] == "FINAL_PEN")].itertuples():
                    day = r.date.strftime("%Y-%m-%d")
                    w = sh[(sh["comp"] == comp) & (sh["date"] == day)]
                    for nm_ in w["winner"]:
                        if ids.get(nm_) in (r.home, r.away):
                            pen[(comp, rnd, frozenset((int(r.hi), int(r.ai))))] = tix[ids[nm_]]
        cups[comp] = Cup(comp, syear, _groups_from(g), g[g["finished"]], g[~g["finished"]].sort_values("date", kind="stable").reset_index(drop=True), ko)
    return ConmebolState(syear, teams, cups, pen)


# ----------------------------------------------------------------------------------------
# Grupos
# ----------------------------------------------------------------------------------------
LAB_K = 16   # partidos de grupo por copa que se pueden fijar en el laboratorio del sitio


def group_phase(cup: Cup, F, rng, r):
    """(S, 8, 4): equipos de cada grupo por puesto; la campaña (S, n) para la localía; y para el
    laboratorio, los puntos de grupo (S, n), el resultado sorteado (0/1/2) de los próximos ``LAB_K``
    partidos y su P(L, E, V) dadas las fuerzas de cada simulación."""
    n = F["att"].shape[1]; S = F["mu"].shape[0]
    pts, gd, gf, w, agf = (np.zeros((S, n)) for _ in range(5))
    hp, hg, hf, ha = (np.zeros((S, n, n)) for _ in range(4))

    def add(h, a, x, y):
        ph = 3.0 * (x > y) + (x == y); pa = 3.0 * (y > x) + (x == y)
        pts[:, h] += ph; pts[:, a] += pa; gd[:, h] += x - y; gd[:, a] += y - x
        gf[:, h] += x; gf[:, a] += y; agf[:, a] += y; w[:, h] += x > y; w[:, a] += y > x
        hp[:, h, a] += ph; hp[:, a, h] += pa; hg[:, h, a] += x - y; hg[:, a, h] += y - x
        hf[:, h, a] += x; hf[:, a, h] += y; ha[:, a, h] += y

    for h, a, x, y in cup.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
        add(int(h), int(a), float(x), float(y))
    K = min(LAB_K, len(cup.todo))
    o = np.zeros((S, K), np.uint8); q = np.zeros((S, K, 3))
    for k, (h, a) in enumerate(cup.todo[["hi", "ai"]].itertuples(index=False)):
        lh, la = _rate(F, int(h), int(a), False)
        x, y = _nb(rng, lh, r).astype(float), _nb(rng, la, r).astype(float)
        if k < K:
            o[:, k] = np.where(x > y, 0, np.where(x == y, 1, 2)); q[:, k] = outcome_probs(lh, la, r)
        add(int(h), int(a), x, y)
    crit_names = (["PTS", "H2H_PTS", "H2H_GD", "H2H_GF", "GD", "GF", "LOTS"] if cup.syear >= 2026
                  else ["PTS", "GD", "GF", "AGF", "LOTS"])
    crit = np.array([CRIT_CODE[c] for c in crit_names], np.int64)
    out = np.zeros((S, len(cup.groups), 4), int)
    for g, idx in enumerate(cup.groups):
        ix = np.array(idx)
        pos = tiebreak.rank_all(pts[:, ix], gd[:, ix], gf[:, ix], w[:, ix], agf[:, ix],
                                hp[:, ix][:, :, ix], hg[:, ix][:, :, ix], hf[:, ix][:, :, ix], ha[:, ix][:, :, ix],
                                rng.random((S, len(ix))), crit, False)
        for k in range(len(ix)):
            out[:, g, k] = ix[np.argmax(pos == k, axis=1)]
    campaign = pts * 1e4 + gd * 1e2 + gf          # mejor campaña (para la localía en cuartos y semis)
    return out, campaign, dict(pts=pts, o=o, q=q)


# ----------------------------------------------------------------------------------------
# Llaves
# ----------------------------------------------------------------------------------------
def _known_pairs(cup: Cup, rnd: str):
    """Cruces de la ronda en los datos: [(u, v)] con u local en la ida (o None si falta alguno)."""
    k = cup.ko.get(rnd)
    if k is None:
        return None
    pairs = []
    for p in {frozenset((int(a), int(b))) for a, b in zip(k["hi"], k["ai"])}:
        u, v = tuple(p)
        g = k[((k["hi"] == u) & (k["ai"] == v)) | ((k["hi"] == v) & (k["ai"] == u))].sort_values("date")
        first = int(g["hi"].iloc[0])
        pairs.append((first, v if first == u else u))
    return pairs


def _legs(cup: Cup, rnd: str, u: int, v: int):
    k = cup.ko[rnd]
    g = k[((k["hi"] == u) & (k["ai"] == v)) | ((k["hi"] == v) & (k["ai"] == u))].sort_values("date")
    legs = []
    for row in g.itertuples():
        if not row.finished:
            legs.append(None)
        else:
            legs.append((row.hg, row.ag) if int(row.hi) == u else (row.ag, row.hg))
    return legs + [None] * (2 - len(legs))


def _tie(st: ConmebolState, cup: Cup, rnd, F, u, v, rng, r, single=False):
    """Llave u (ida de local) contra v; con un partido único (final) es neutral y con alargue."""
    S = F["mu"].shape[0]
    u = np.broadcast_to(u, (S,)).astype(int); v = np.broadcast_to(v, (S,)).astype(int)
    if (u == u[0]).all() and (v == v[0]).all() and rnd in cup.ko:
        uu, vv = int(u[0]), int(v[0])
        win = st.pen.get((cup.comp, rnd, frozenset((uu, vv))))
        if win is not None:
            return np.full(S, win)
        # si la ronda siguiente ya está en los datos, el que aparece ahí pasó
        rounds = KO[cup.comp]; i = rounds.index(rnd)
        if i + 1 < len(rounds) and rounds[i + 1] in cup.ko:
            nxt = set(cup.ko[rounds[i + 1]]["hi"]) | set(cup.ko[rounds[i + 1]]["ai"])
            if uu in nxt or vv in nxt:
                return np.full(S, uu if uu in nxt else vv)
        legs = _legs(cup, rnd, uu, vv)
        if single:
            return two_legs(F, u, v, rng, r, [legs[0], None], neutral=True)
        return two_legs(F, u, v, rng, r, legs, extra_time=False)
    if single:
        return two_legs(F, u, v, rng, r, None, neutral=True)
    return two_legs(F, u, v, rng, r, None, extra_time=False)


def _host_order(a, b, campaign):
    """(ida de local, vuelta de local): la vuelta para el de mejor campaña."""
    S = a.shape[0]; rows = np.arange(S)
    better_b = campaign[rows, b] >= campaign[rows, a]
    return np.where(better_b, a, b), np.where(better_b, b, a)


def _bracket(st, cup, rounds, alive, F, rng, r, campaign, stage, base):
    """Rondas después de octavos: cruces de los datos si están; si no, el cuadro se sortea."""
    S = F["mu"].shape[0]; rows = np.arange(S)
    for k, rnd in enumerate(rounds):
        single = rnd == "final" and cup.syear >= 2019
        known = _known_pairs(cup, rnd)
        need = len(alive) // 2
        if known is not None and len(known) == need:
            winners = [_tie(st, cup, rnd, F, u, v, rng, r, single) for u, v in known]
        else:
            A = np.stack(alive, 1)
            A = np.take_along_axis(A, np.argsort(rng.random(A.shape), 1), 1)
            winners = []
            for j in range(need):
                a, b = A[:, 2 * j], A[:, 2 * j + 1]
                if not single:
                    a, b = _host_order(a, b, campaign)
                winners.append(_tie(st, cup, rnd, F, a, b, rng, r, single))
        for wv in winners:
            stage[rows, wv] = base + k + 1
        alive = winners
    return stage


def simulate(st: ConmebolState, draw, S=20000, seed=0, chunk=5000, r=20.0):
    """Devuelve, por copa, la etapa alcanzada (S, n) y los puestos de grupo (S, 8, 4)."""
    rng = np.random.default_rng(seed)
    n = len(st.teams)
    out = {c: dict(stage=[], groups=[], pts=[], o=[], q=[]) for c in st.cups}
    for a in range(0, S, chunk):
        b = min(S, a + chunk); s = b - a; rows = np.arange(s)
        F = draw(s, rng)
        third = None
        if LIB in st.cups:
            cup = st.cups[LIB]
            gp, camp, lab = group_phase(cup, F, rng, r)
            stage = np.full((s, n), -1, np.int8)
            stage[rows[:, None], gp.reshape(s, -1)] = 0
            stage[rows[:, None], gp[:, :, :2].reshape(s, -1)] = 1
            third = gp[:, :, 2]
            known = _known_pairs(cup, "round-of-16")
            if known is not None and len(known) == 8:
                # los cruces reales mandan sobre la tabla simulada (desempates por sorteo o partidos faltantes)
                stage[stage >= 1] = 0
                for u, v in known:
                    stage[:, u] = 1; stage[:, v] = 1
                winners = [_tie(st, cup, "round-of-16", F, u, v, rng, r) for u, v in known]
            else:
                perm = np.argsort(rng.random((s, 8)), 1)
                winners = [_tie(st, cup, "round-of-16", F, gp[rows, perm[:, g], 1], gp[:, g, 0], rng, r) for g in range(8)]
            for wv in winners:
                stage[rows, wv] = 2
            stage = _bracket(st, cup, KO[LIB][1:], winners, F, rng, r, camp, stage, 2)
            out[LIB]["stage"].append(stage); out[LIB]["groups"].append(gp)
            for k in ("pts", "o", "q"):
                out[LIB][k].append(lab[k])
        if SUD in st.cups:
            cup = st.cups[SUD]
            gp, camp, lab = group_phase(cup, F, rng, r)
            stage = np.full((s, n), -1, np.int8)
            stage[rows[:, None], gp.reshape(s, -1)] = 0
            stage[rows[:, None], gp[:, :, 1].reshape(s, -1)] = 1
            stage[rows[:, None], gp[:, :, 0].reshape(s, -1)] = 2
            # playoff: segundos de la Sudamericana contra terceros de la Libertadores
            known = _known_pairs(cup, "knockout-round-playoffs")
            if known is not None and len(known) == 8:
                stage[stage >= 1] = 0
                for u, v in known:
                    stage[:, u] = 1; stage[:, v] = 1
                po = [_tie(st, cup, "knockout-round-playoffs", F, u, v, rng, r) for u, v in known]
            elif third is not None:
                perm = np.argsort(rng.random((s, 8)), 1)
                po = [_tie(st, cup, "knockout-round-playoffs", F, gp[:, g, 1], third[rows, perm[:, g]], rng, r) for g in range(8)]
                for g in range(8):
                    stage[rows, third[:, g]] = np.maximum(stage[rows, third[:, g]], 1)
            else:
                po = [gp[:, g, 1] for g in range(8)]
            for wv in po:
                stage[rows, wv] = 2
            known = _known_pairs(cup, "round-of-16")
            if known is not None and len(known) == 8:
                stage[stage >= 2] = 1
                for u, v in known:
                    stage[:, u] = 2; stage[:, v] = 2
                winners = [_tie(st, cup, "round-of-16", F, u, v, rng, r) for u, v in known]
            else:
                P = np.stack(po, 1); P = np.take_along_axis(P, np.argsort(rng.random(P.shape), 1), 1)
                winners = [_tie(st, cup, "round-of-16", F, P[:, g], gp[:, g, 0], rng, r) for g in range(8)]
            for wv in winners:
                stage[rows, wv] = 3
            stage = _bracket(st, cup, KO[SUD][2:], winners, F, rng, r, camp, stage, 3)
            out[SUD]["stage"].append(stage); out[SUD]["groups"].append(gp)
            for k in ("pts", "o", "q"):
                out[SUD][k].append(lab[k])
    return {c: {k: np.concatenate(v, 0) for k, v in o.items()} for c, o in out.items()}
