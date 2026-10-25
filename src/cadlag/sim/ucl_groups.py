r"""Champions League con fase de grupos (formato hasta 2023-24), para validar con más ediciones.

Ocho grupos de cuatro a dos ruedas; pasan los dos primeros. Desempate: puntos, minitabla (puntos,
diferencia, goles y, hasta 2020-21, goles de visita, recalculada entre los que siguen empatados),
diferencia y goles totales; luego sorteo. Octavos: primeros contra segundos de otro grupo (no se
modela la restricción de asociación), con la vuelta en cancha del primero. Cuartos y semis: sorteo
libre. En 2019-20 cuartos, semis y final fueron a partido único en cancha neutral (Lisboa).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..leagues import CRIT_CODE
from . import tiebreak
from .ucl import _nb, _rate, _tie

ROUNDS = ["round-of-16", "quarterfinals", "semifinals", "final"]
ALIASES = {"quarter-finals": "quarterfinals", "semi-finals": "semifinals"}
STAGES = ["Grupos", "Octavos", "Cuartos", "Semifinal", "Final", "Campeón"]


@dataclass
class GroupState:
    comp: str
    syear: int
    teams: list
    groups: list                # 8 listas de índices
    done: pd.DataFrame
    todo: pd.DataFrame
    ko: dict = field(default_factory=dict)
    single_leg_from: str | None = None   # ronda desde la que todo es a partido único y neutral


def group_state(matches, comp, syear, asof=None) -> GroupState:
    d = matches[(matches["comp"] == comp) & (matches["syear"] == syear)].copy()
    d["slug"] = d["slug"].astype(str).replace(ALIASES)
    gs = d[d["slug"] == "group-stage"].copy()
    if asof is not None:
        gs.loc[gs["date"] > asof, "finished"] = False
    teams = sorted(set(gs["home"]) | set(gs["away"]))
    tix = {t: i for i, t in enumerate(teams)}
    gs["hi"] = gs["home"].map(tix); gs["ai"] = gs["away"].map(tix)
    # grupos: componentes conexas del grafo de rivales
    adj = {i: set() for i in range(len(teams))}
    for h, a in zip(gs["hi"], gs["ai"]):
        adj[h].add(a); adj[a].add(h)
    seen, groups = set(), []
    for i in range(len(teams)):
        if i in seen:
            continue
        comp_, st = [], [i]
        while st:
            j = st.pop()
            if j in seen:
                continue
            seen.add(j); comp_.append(j); st += list(adj[j])
        groups.append(sorted(comp_))
    ko, prev_end = {}, gs["date"].max()
    for r in ROUNDS:
        k = d[d["slug"] == r].copy()
        end = k["date"].max() if len(k) else None
        if asof is not None:
            if prev_end is None or asof < prev_end:
                k = k.iloc[0:0]
            k = k.copy(); k.loc[k["date"] > asof, "finished"] = False
        prev_end = end
        k = k[k["home"].isin(tix) & k["away"].isin(tix)]
        if len(k):
            k["hi"] = k["home"].map(tix); k["ai"] = k["away"].map(tix)
            ko[r] = k.sort_values("date")
    return GroupState(comp, syear, teams, groups, gs[gs["finished"]], gs[~gs["finished"]].reset_index(drop=True), ko,
                      "quarterfinals" if (comp, syear) == ("uefa.champions", 2019) else None)


def group_phase(gs: GroupState, F, rng, r):
    """(S, 8, 2): primero y segundo de cada grupo."""
    n = len(gs.teams); S = F["mu"].shape[0]
    pts, gd, gf, w, agf = (np.zeros((S, n)) for _ in range(5))
    hp, hg, hf, ha = (np.zeros((S, n, n)) for _ in range(4))

    def add(h, a, x, y):
        ph = 3.0 * (x > y) + (x == y); pa = 3.0 * (y > x) + (x == y)
        pts[:, h] += ph; pts[:, a] += pa; gd[:, h] += x - y; gd[:, a] += y - x
        gf[:, h] += x; gf[:, a] += y; agf[:, a] += y; w[:, h] += x > y; w[:, a] += y > x
        hp[:, h, a] += ph; hp[:, a, h] += pa; hg[:, h, a] += x - y; hg[:, a, h] += y - x
        hf[:, h, a] += x; hf[:, a, h] += y; ha[:, a, h] += y

    for h, a, x, y in gs.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
        add(int(h), int(a), float(x), float(y))
    for h, a in gs.todo[["hi", "ai"]].itertuples(index=False):
        lh, la = _rate(F, int(h), int(a), False)
        add(int(h), int(a), _nb(rng, lh, r).astype(float), _nb(rng, la, r).astype(float))
    crit_names = ["PTS", "H2H_PTS", "H2H_GD", "H2H_GF"] + (["H2H_AGF"] if gs.syear <= 2020 else []) + ["GD", "GF", "LOTS"]
    crit = np.array([CRIT_CODE[c] for c in crit_names], np.int64)
    out = np.zeros((S, len(gs.groups), 2), int)
    for g, idx in enumerate(gs.groups):
        ix = np.array(idx)
        pos = tiebreak.rank_all(pts[:, ix], gd[:, ix], gf[:, ix], w[:, ix], agf[:, ix],
                                hp[:, ix][:, :, ix], hg[:, ix][:, :, ix], hf[:, ix][:, :, ix], ha[:, ix][:, :, ix],
                                rng.random((S, len(ix))), crit, True)
        for k in range(2):
            out[:, g, k] = ix[np.argmax(pos == k, axis=1)]
    return out


def _derange(S, rng, k=8):
    """Permutación (S, k): de qué grupo es el segundo que enfrenta al primero de cada grupo (nunca el
    mismo grupo; muestreo por rechazo)."""
    perm = np.argsort(rng.random((S, k)), 1)
    for _ in range(200):
        bad = (perm == np.arange(k)[None, :]).any(1)
        if not bad.any():
            break
        perm[bad] = np.argsort(rng.random((bad.sum(), k)), 1)
    return perm


def _known_round(gs, rnd):
    k = gs.ko.get(rnd)
    if k is None:
        return None
    pairs = []
    for p in {frozenset((int(a), int(b))) for a, b in zip(k["hi"], k["ai"])}:
        u, v = tuple(p)
        g = k[((k["hi"] == u) & (k["ai"] == v)) | ((k["hi"] == v) & (k["ai"] == u))].sort_values("date")
        first = int(g["hi"].iloc[0]); pairs.append((first, v if first == u else u))
    need = {"round-of-16": 8, "quarterfinals": 4, "semifinals": 2, "final": 1}[rnd]
    return pairs if len(pairs) == need else None


def knockout(gs: GroupState, top2, F, rng, r):
    """Etapa alcanzada (S, n): 0 grupos, 1 octavos, 2 cuartos, 3 semis, 4 final, 5 campeón."""
    S = top2.shape[0]; n = len(gs.teams); rows = np.arange(S)
    stage = np.zeros((S, n), np.int8)
    stage[rows[:, None], top2.reshape(S, -1)] = 1
    alive = None
    for k, rnd in enumerate(ROUNDS):
        single = gs.single_leg_from is not None and k >= ROUNDS.index(gs.single_leg_from)
        neutral = single or rnd == "final"
        known = _known_round(gs, rnd)
        winners = []
        if known is not None:
            for u, v in known:
                wv = _tie(gs, F, rnd, np.full(S, u), np.full(S, v), rng, r, neutral=neutral)
                winners.append(wv)
        elif rnd == "round-of-16":
            perm = _derange(S, rng, len(gs.groups))
            for g in range(8):
                win_g = top2[:, g, 0]; ru = top2[rows, perm[:, g], 1]
                winners.append(_tie(gs, F, rnd, ru, win_g, rng, r, neutral=neutral))
        else:
            A = np.stack(alive, 1)                       # (S, m)
            m_ = A.shape[1]
            sh = np.take_along_axis(A, np.argsort(rng.random((S, m_)), 1), 1)
            for j in range(m_ // 2):
                winners.append(_tie(gs, F, rnd, sh[:, 2 * j], sh[:, 2 * j + 1], rng, r, neutral=neutral))
        for wv in winners:
            stage[rows, wv] = k + 2
        alive = winners
    return stage


def simulate_groups(gs: GroupState, draw, S=4000, seed=0, chunk=4000, r=20.0):
    rng = np.random.default_rng(seed)
    out = []
    for a in range(0, S, chunk):
        F = draw(min(S, a + chunk) - a, rng)
        out.append(knockout(gs, group_phase(gs, F, rng, r), F, rng, r))
    return np.concatenate(out, 0)
