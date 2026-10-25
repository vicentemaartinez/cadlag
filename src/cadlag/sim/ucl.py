r"""Simulación de la Champions League (formato desde 2024-25).

Fase liga: 36 equipos, 8 partidos cada uno, una sola tabla. Desempate (reglamento 2026-27, art. 18):
puntos, diferencia de gol, goles a favor, goles de visita, victorias, victorias de visita, puntos
de los rivales, diferencia de gol de los rivales, goles de los rivales; después fair play y
coeficiente UEFA, que no se modelan (sorteo).

Eliminatorias (art. 19):

  * playoff: los grupos I–IV enfrentan 9/10 con 23/24, 11/12 con 21/22, 13/14 con 19/20 y 15/16
    con 17/18; cuál con cuál se sortea. El mejor clasificado juega la vuelta de local.
  * octavos: A = 1/2 contra los ganadores de IV, B = 3/4 contra III, C = 5/6 contra II y
    D = 7/8 contra I (sorteado dentro de cada par). 1 y 2 quedan en mitades opuestas; los demás
    pares se reparten al azar entre las mitades.
  * cuartos fijos en cada mitad: A contra D y B contra C; semifinal: ganador de A–D contra el de
    B–C. La vuelta la juega de local el puesto del cuadro mejor sembrado (A, luego B), aunque lo
    ocupe quien eliminó al sembrado.
  * final a partido único en cancha neutral (si llega el dueño del estadio, juega de local:
    ``FINAL_HOST``).

Llaves a dos partidos sin gol de visita: si el global queda igualado hay alargue (tasas × 1/3, en
la cancha de la vuelta) y luego penales, que se tratan como una moneda.

Lo ya jugado se respeta: los partidos terminados de la fase liga y, en las eliminatorias, los
cruces y resultados que ya están en los datos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROUNDS = ["knockout-round-playoffs", "round-of-16", "quarterfinals", "semifinals", "final"]
STAGES = ["Fase liga", "Playoff", "Octavos", "Cuartos", "Semifinal", "Final", "Campeón"]

# Partidos de la fase liga que ESPN no trae (calendario oficial de la UEFA). Se agregan solo si el
# cruce no está ya en los datos.
MISSING_FIXTURES = {
    ("uefa.champions", 2026): [
        ("2026-11-04 20:00", "Slavia Prague", "Arsenal"),
        ("2026-11-24 20:00", "Manchester City", "Napoli"),
        ("2026-12-09 20:00", "Borussia Dortmund", "Internazionale"),
        ("2026-12-09 20:00", "PSV Eindhoven", "Atlético Madrid"),
        ("2027-01-19 20:00", "Internazionale", "Liverpool"),
    ],
}


# Final en el estadio de un club que puede llegar a jugarla: ese club juega de local (la UEFA la
# declara neutral, pero en la práctica no lo es). 2027: Metropolitano, Madrid.
FINAL_HOST = {("uefa.champions", 2026): "Atlético Madrid"}

# Mitades del cuadro, cuando ya se sorteó y los cuartos todavía no están en los datos: para X = 1, 2,
# 3 (pares 3/4, 5/6 y 7/8 de la fase liga), 0 si el 3.º, 5.º o 7.º quedó en la mitad del 1.º y 1 si
# quedó en la del 2.º. Sin dato, se sortea.
BRACKET_HALVES: dict = {}


@dataclass
class CupState:
    comp: str
    syear: int
    teams: list                 # ids ESPN (36)
    done: pd.DataFrame          # fase liga terminada: hi, ai, hg, ag
    todo: pd.DataFrame          # fase liga pendiente: hi, ai, date, id
    ko: dict = field(default_factory=dict)   # ronda -> partidos de esa ronda (hi, ai, hg, ag, finished, date)
    names: dict = field(default_factory=dict)


def cup_state(matches: pd.DataFrame, comp: str, syear: int, names: dict, asof=None) -> CupState:
    d = matches[(matches["comp"] == comp) & (matches["syear"] == syear)].copy()
    lp = d[d["slug"].astype(str) == "league-phase"].copy()
    # nombres → id solo entre los equipos de esta fase liga (hay homónimos: Liverpool de Montevideo)
    present = set(lp["home"]) | set(lp["away"])
    ids = {v: k for k, v in names.items() if k in present}
    extra = []
    for date, h, a in MISSING_FIXTURES.get((comp, syear), []):
        hid, aid = ids.get(h), ids.get(a)
        if hid is None or aid is None:
            continue
        if ((lp["home"] == hid) & (lp["away"] == aid)).any():
            continue
        extra.append(dict(comp=comp, id=f"x-{hid}-{aid}", date=pd.Timestamp(date, tz="UTC"), home=hid, away=aid,
                          hg=np.nan, ag=np.nan, finished=False, slug="league-phase", stage="regular", syear=syear,
                          neutral=False))
    if extra:
        lp = pd.concat([lp, pd.DataFrame(extra)], ignore_index=True)
    if asof is not None:
        lp.loc[lp["date"] > asof, "finished"] = False
    teams = sorted(set(lp["home"]) | set(lp["away"]))
    tix = {t: i for i, t in enumerate(teams)}
    lp["hi"] = lp["home"].map(tix); lp["ai"] = lp["away"].map(tix)
    done = lp[lp["finished"]].sort_values("date")
    todo = lp[~lp["finished"]].sort_values("date").reset_index(drop=True)
    ko = {}
    prev_end = lp["date"].max()
    for r in ROUNDS:
        k = d[d["slug"].astype(str) == r].copy()
        end = k["date"].max() if len(k) else None
        if asof is not None:
            # en un backtest, los cruces de una ronda se conocen recién cuando terminó la anterior
            if prev_end is None or asof < prev_end:
                k = k.iloc[0:0]
            k = k.copy(); k.loc[k["date"] > asof, "finished"] = False
        prev_end = end
        k = k[k["home"].isin(tix) & k["away"].isin(tix)]
        if len(k):
            k["hi"] = k["home"].map(tix); k["ai"] = k["away"].map(tix)
            ko[r] = k.sort_values("date")
    return CupState(comp, syear, teams, done, todo, ko, names)


# ----------------------------------------------------------------------------------------
# Goles
# ----------------------------------------------------------------------------------------
def _nb(rng, mean, r):
    return rng.negative_binomial(r, r / (r + mean))


def _rate(F, h, a, neutral, idx=None):
    """Tasas (local h, visita a) con F = dict(att, dfn (S, n), mu, eta (S,)); h, a: (S,) o escalares."""
    S = F["mu"].shape[0]
    rows = np.arange(S)
    h = np.broadcast_to(h, (S,)); a = np.broadcast_to(a, (S,))
    nf = 0.0 if neutral else 1.0
    lh = np.exp(F["mu"] + F["eta"] * nf + F["att"][rows, h] - F["dfn"][rows, a])
    la = np.exp(F["mu"] + F["att"][rows, a] - F["dfn"][rows, h])
    return lh, la


def two_legs(F, u, v, rng, r, legs=None, neutral=False):
    """Llave entre u (ida de local) y v (vuelta de local); devuelve el ganador (S,).
    ``legs``: resultados conocidos [(gu, gv) o None, (gu, gv) o None] (goles de u y de v)."""
    S = F["mu"].shape[0]
    u = np.broadcast_to(u, (S,)).astype(int); v = np.broadcast_to(v, (S,)).astype(int)
    legs = legs or [None, None]
    tot_u = np.zeros(S); tot_v = np.zeros(S)
    if neutral:     # final: un partido
        if legs[0] is not None:
            gu, gv = np.full(S, legs[0][0]), np.full(S, legs[0][1])
        else:
            lu, lv = _rate(F, u, v, True)
            gu, gv = _nb(rng, lu, r), _nb(rng, lv, r)
        tot_u += gu; tot_v += gv
        lu, lv = _rate(F, u, v, True)
    else:
        if legs[0] is not None:
            tot_u += legs[0][0]; tot_v += legs[0][1]
        else:
            lu, lv = _rate(F, u, v, False)
            tot_u += _nb(rng, lu, r); tot_v += _nb(rng, lv, r)
        lv2, lu2 = _rate(F, v, u, False)
        if legs[1] is not None:
            tot_u += legs[1][0]; tot_v += legs[1][1]
        else:
            tot_u += _nb(rng, lu2, r); tot_v += _nb(rng, lv2, r)
        lu, lv = lu2, lv2
    tie = tot_u == tot_v
    if tie.any():
        eu, ev = _nb(rng, lu / 3, r), _nb(rng, lv / 3, r)
        tot_u = tot_u + tie * eu; tot_v = tot_v + tie * ev
    tie = tot_u == tot_v
    coin = rng.random(S) < 0.5
    return np.where(tot_u > tot_v, u, np.where(tot_v > tot_u, v, np.where(coin, u, v)))


def final_with_host(F, u, v, host, rng, r):
    """Final a partido único: neutral, salvo que juegue el dueño del estadio (de local)."""
    S = F["mu"].shape[0]
    hv = v == host
    h = np.where(hv, v, u); a = np.where(hv, u, v); home = (u == host) | hv
    lhn, lan = _rate(F, h, a, True); lhh, lah = _rate(F, h, a, False)
    lh, la = np.where(home, lhh, lhn), np.where(home, lah, lan)
    gh, ga = _nb(rng, lh, r), _nb(rng, la, r)
    tie = gh == ga
    gh = gh + tie * _nb(rng, lh / 3, r); ga = ga + tie * _nb(rng, la / 3, r)
    coin = rng.random(S) < 0.5
    return np.where(gh > ga, h, np.where(ga > gh, a, np.where(coin, h, a)))


# ----------------------------------------------------------------------------------------
# Fase liga
# ----------------------------------------------------------------------------------------
def league_phase(cs: CupState, F, rng, r):
    n = len(cs.teams); S = F["mu"].shape[0]
    pts, gd, gf, agf, w, aw = (np.zeros((S, n)) for _ in range(6))
    O = np.zeros((n, n))

    def add(h, a, x, y):
        ph = 3.0 * (x > y) + (x == y); pa = 3.0 * (y > x) + (x == y)
        pts[:, h] += ph; pts[:, a] += pa
        gd[:, h] += x - y; gd[:, a] += y - x
        gf[:, h] += x; gf[:, a] += y; agf[:, a] += y
        w[:, h] += x > y; w[:, a] += y > x; aw[:, a] += y > x
        O[h, a] += 1; O[a, h] += 1

    for h, a, x, y in cs.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
        add(int(h), int(a), float(x), float(y))
    K = len(cs.todo)
    gx = np.zeros((S, K), np.int64); gy = np.zeros((S, K), np.int64)
    qk = np.zeros((S, K, 3))
    for k, (h, a) in enumerate(cs.todo[["hi", "ai"]].itertuples(index=False)):
        lh, la = _rate(F, int(h), int(a), False)
        gx[:, k], gy[:, k] = _nb(rng, lh, r), _nb(rng, la, r)
        qk[:, k] = outcome_probs(lh, la, r)
        add(int(h), int(a), gx[:, k].astype(float), gy[:, k].astype(float))
    opp_pts, opp_gd, opp_gf = pts @ O.T, gd @ O.T, gf @ O.T
    rnd = rng.random((S, n))
    keys = [rnd, -opp_gf, -opp_gd, -opp_pts, -aw, -w, -agf, -gf, -gd, -pts]   # del menos al más importante
    order = np.lexsort(keys, axis=1)                         # (S, n): equipo en cada puesto
    pos = np.empty_like(order); np.put_along_axis(pos, order, np.arange(n)[None, :].repeat(S, 0), 1)
    return dict(pts=pts, pos=pos, order=order, gx=gx, gy=gy, q=qk)


def outcome_probs(lh, la, r, maxg=10):
    """P(L, E, V) con goles NegBin independientes, por simulación."""
    g = np.arange(maxg + 1)
    from scipy.stats import nbinom
    ph = nbinom.pmf(g[None, :], r, r / (r + lh[:, None])); pa = nbinom.pmf(g[None, :], r, r / (r + la[:, None]))
    J = ph[:, :, None] * pa[:, None, :]
    tri = np.tril(np.ones((maxg + 1, maxg + 1)), -1)
    pH = (J * tri).sum((1, 2)); pA = (J * tri.T).sum((1, 2)); pD = np.einsum("sii->s", J)
    z = pH + pD + pA
    return np.stack([pH, pD, pA], 1) / z[:, None]


# ----------------------------------------------------------------------------------------
# Eliminatorias
# ----------------------------------------------------------------------------------------
def _known_legs(cs: CupState, rnd: str, u: int, v: int):
    """Resultados conocidos de la llave u–v en la ronda (u local en la ida). None si no está."""
    k = cs.ko.get(rnd)
    if k is None:
        return None
    g = k[((k["hi"] == u) & (k["ai"] == v)) | ((k["hi"] == v) & (k["ai"] == u))].sort_values("date")
    if not len(g):
        return None
    legs = []
    for row in g.itertuples():
        if not row.finished:
            legs.append(None)
        elif row.hi == u:
            legs.append((row.hg, row.ag))
        else:
            legs.append((row.ag, row.hg))
    return legs + [None] * (2 - len(legs))


def _pairs(cs: CupState, rnd: str):
    k = cs.ko.get(rnd)
    if k is None:
        return []
    return list({frozenset((int(a), int(b))) for a, b in zip(k["hi"], k["ai"])})


def _winner_known(cs: CupState, rnd: str, u: int, v: int):
    """Si la llave ya terminó y el ganador sale de la ronda siguiente (penales), lo devuelve."""
    i = ROUNDS.index(rnd)
    if i + 1 < len(ROUNDS):
        for p in _pairs(cs, ROUNDS[i + 1]):
            if u in p:
                return u
            if v in p:
                return v
    return None


def _tie(cs, F, rnd, u, v, rng, r, neutral=False):
    """Llave vectorizada; si u y v son iguales en todas las simulaciones y la llave está en los datos,
    usa los resultados conocidos."""
    S = F["mu"].shape[0]
    u = np.broadcast_to(u, (S,)); v = np.broadcast_to(v, (S,))
    if (u == u[0]).all() and (v == v[0]).all():
        uu, vv = int(u[0]), int(v[0])
        wk = _winner_known(cs, rnd, uu, vv)
        if wk is not None:
            return np.full(S, wk)
        legs = _known_legs(cs, rnd, uu, vv)
        if legs is not None:
            k = cs.ko[rnd]
            g = k[((k["hi"] == uu) & (k["ai"] == vv)) | ((k["hi"] == vv) & (k["ai"] == uu))].sort_values("date")
            first_home = int(g["hi"].iloc[0])
            if first_home != uu and not neutral:      # la ida se jugó en cancha de v: se invierte
                legs2 = _known_legs(cs, rnd, vv, uu)
                return two_legs(F, vv, uu, rng, r, legs2)
            return two_legs(F, uu, vv, rng, r, legs, neutral)
    return two_legs(F, u, v, rng, r, None, neutral)


def knockout(cs: CupState, lp: dict, F, rng, r, halves: dict | None = None):
    """Devuelve la etapa alcanzada por cada equipo, (S, n): 0 fase liga, 1 playoff, 2 octavos,
    3 cuartos, 4 semis, 5 final, 6 campeón."""
    S, n = lp["pos"].shape
    rows = np.arange(S)
    T = lp["order"]                       # T[s, k]: equipo en el puesto k (0-based)
    stage = np.zeros((S, n), np.int8)
    stage[rows[:, None], T[:, 8:24]] = 1
    stage[rows[:, None], T[:, :8]] = 2
    known_po = _pairs(cs, ROUNDS[0])
    # ---- playoff: grupo g, sembrados 8+2g, 9+2g contra 22-2g, 23-2g
    po_win = np.zeros((S, 4, 2), int)
    for g in range(4):
        f = rng.integers(0, 2, S)
        if known_po:
            # cruce real (la fase liga ya terminó, así que T es el mismo en todas las simulaciones)
            s0 = int(T[0, 8 + 2 * g])
            opp = next((int(x) for p in known_po if s0 in p for x in p if x != s0), -1)
            if opp >= 0:
                f = np.where(T[:, 22 - 2 * g] == opp, 0, 1)
        for j in range(2):
            seeded = T[:, 8 + 2 * g + j]
            un = T[rows, np.where(j == 0, 22 - 2 * g + f, 23 - 2 * g - f)]
            win = _tie(cs, F, ROUNDS[0], un, seeded, rng, r)          # el sembrado juega la vuelta de local
            po_win[:, g, j] = win
            stage[rows, win] = 2
    # ---- octavos: letra X (A..D) = sembrados 2X, 2X+1 contra ganadores del grupo 3-X
    known16 = _pairs(cs, ROUNDS[1])
    r16 = np.zeros((S, 4, 2), int)          # ganador de la llave del sembrado 2X+j
    for X in range(4):
        e = rng.integers(0, 2, S)
        if known16:
            s0 = int(T[0, 2 * X])
            opp = next((int(x) for p in known16 if s0 in p for x in p if x != s0), -1)
            if opp >= 0:
                e = np.where(po_win[:, 3 - X, 0] == opp, 0, 1)
        for j in range(2):
            seeded = T[:, 2 * X + j]
            o = po_win[rows, 3 - X, np.where(j == 0, e, 1 - e)]
            win = _tie(cs, F, ROUNDS[1], o, seeded, rng, r)
            r16[:, X, j] = win
            stage[rows, win] = 3
    # ---- mitades: el sembrado 2X va a la mitad hX (A: el 1 en la mitad 0)
    h = np.zeros((S, 4), int)
    for X in (1, 2, 3):
        h[:, X] = rng.integers(0, 2, S)
        if halves and X in halves:
            h[:, X] = halves[X]
    knownqf = _pairs(cs, ROUNDS[2])
    if knownqf:
        # cuartos reales: A–D y B–C dicen en qué mitad quedó cada par
        for X, Y in ((0, 3), (1, 2)):
            for p in knownqf:
                for jx in range(2):
                    a = int(r16[0, X, jx])
                    if a in p:
                        b = next(int(x) for x in p if x != a)
                        jy = 0 if int(r16[0, Y, 0]) == b else 1
                        # el de A (o B) en la mitad hX^jx; el de D (o C) debe estar en la misma
                        h[:, Y] = h[:, X] ^ jx ^ jy
        # semis reales: dicen qué cuarto B–C va con cuál A–D (fija h_B, y con él h_C)
        knownsf = _pairs(cs, ROUNDS[3])
        if knownsf:
            who = {int(r16[0, X, j]): (X, j) for X in range(4) for j in range(2)}
            relC = h[0, 2] ^ h[0, 1]
            for p in knownsf:
                (a, b) = tuple(p)
                if a not in who or b not in who:
                    continue
                (Xa, ja), (Xb, jb) = who[a], who[b]
                if Xa in (1, 2):
                    (Xa, ja), (Xb, jb) = (Xb, jb), (Xa, ja)
                if Xa not in (0, 3) or Xb not in (1, 2):
                    continue
                half_a = ja if Xa == 0 else int(h[0, 3]) ^ ja
                h1 = half_a ^ jb ^ (relC if Xb == 2 else 0)
                h[:, 1] = h1; h[:, 2] = h1 ^ relC
                break

    def slot(X, half):
        j = h[:, X] ^ half          # cuál de los dos sembrados (2X o 2X+1) quedó en esa mitad
        return r16[rows, X, j]
    sf = np.zeros((S, 2), int)
    for half in range(2):
        a, d_, b, c = slot(0, half), slot(3, half), slot(1, half), slot(2, half)
        q1 = _tie(cs, F, ROUNDS[2], d_, a, rng, r); stage[rows, q1] = 4
        q2 = _tie(cs, F, ROUNDS[2], c, b, rng, r); stage[rows, q2] = 4
        s = _tie(cs, F, ROUNDS[3], q2, q1, rng, r); stage[rows, s] = 5
        sf[:, half] = s
    host_name = FINAL_HOST.get((cs.comp, cs.syear))
    host = next((cs.teams.index(k) for k, v in cs.names.items() if v == host_name and k in cs.teams), None) if host_name else None
    if host is None or _pairs(cs, ROUNDS[4]):
        champ = _tie(cs, F, ROUNDS[4], sf[:, 0], sf[:, 1], rng, r, neutral=True)
    else:
        champ = final_with_host(F, sf[:, 0], sf[:, 1], host, rng, r)
    stage[rows, champ] = 6
    return stage


def simulate_cup(cs: CupState, draw, S: int = 20000, seed: int = 0, chunk: int = 5000, r: float = 20.0):
    """``draw(S, rng)`` devuelve F = dict(att, dfn (S, n), mu, eta (S,)) para los equipos de ``cs``."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("pos", "pts", "stage", "gx", "gy", "q")}
    for a in range(0, S, chunk):
        b = min(S, a + chunk)
        F = draw(b - a, rng)
        lp = league_phase(cs, F, rng, r)
        st = knockout(cs, lp, F, rng, r, BRACKET_HALVES.get((cs.comp, cs.syear)))
        for k in ("pos", "pts", "gx", "gy", "q"):
            out[k].append(lp[k])
        out["stage"].append(st)
    return {k: np.concatenate(v, 0) for k, v in out.items()}
