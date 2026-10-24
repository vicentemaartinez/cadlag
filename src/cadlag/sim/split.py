r"""Formato argentino (desde 2025): dos torneos por año con zonas, playoffs y descensos anuales.

Una temporada (año calendario) tiene Apertura y Clausura. En cada torneo, 30 equipos en dos zonas
de 15 juegan 14 partidos de zona y 2 interzonales; los 8 primeros de cada zona pasan a una llave
de 16 a partido único (1.ºA-8.ºB, 1.ºB-8.ºA, 2.ºA-7.ºB, 2.ºB-7.ºA, 3.ºA-6.ºB, 3.ºB-6.ºA, 4.ºA-5.ºB,
4.ºB-5.ºA; cuartos 1-2, 3-4, 5-6, 7-8; semis entre ganadores consecutivos), con localía del mejor
ubicado y final neutral. La **tabla anual** suma la fase regular de los dos torneos y, con los
**promedios** (puntos de las tres últimas temporadas sobre partidos jugados), define los dos
descensos.

Lo que este módulo agrega al simulador general (``sim.season``):

* ``season_state_split``: el estado de la temporada con el torneo en curso, la zona de cada
  equipo (inferida del fixture: dos equipos de la misma zona comparten casi todos sus rivales), el
  fixture del Clausura completado con el del Apertura invertido (ESPN lo publica por tramos), los
  resultados de playoffs ya jugados y la base de los promedios.
* ``split_outcomes``: para cada temporada simulada, la tabla de zonas del torneo en curso, el
  campeón (llave con los resultados ya jugados fijados) y los dos descendidos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOURN = {"apertura": 0, "clausura": 1}
ZONE_A_ANCHOR = {"arg.1": "5"}   # Boca Juniors (id ESPN) está en la zona A desde 2025
# octavos: (posición en A, posición en B) con el de zona A local si es mejor ubicado, etc.
R16 = [((0, 0), (1, 7)), ((1, 0), (0, 7)), ((0, 1), (1, 6)), ((1, 1), (0, 6)),
       ((0, 2), (1, 5)), ((1, 2), (0, 5)), ((0, 3), (1, 4)), ((1, 3), (0, 4))]   # ((zona, pos), (zona, pos))


def _tourn(slug: str) -> int | None:
    s = str(slug)
    for k, v in TOURN.items():
        if s.startswith(k) or s.startswith(f"torneo-{k}"):
            return v
    return None


def infer_zones(pairs: list[tuple[int, int]], n: int) -> np.ndarray:
    """Zona (0/1) de cada equipo a partir de los partidos de un torneo: los de la misma zona se
    enfrentan casi todos entre sí (14 de 16 partidos), así que comparten casi todos sus rivales."""
    adj = np.zeros((n, n), bool)
    for a, b in pairs:
        adj[a, b] = adj[b, a] = True
    common = adj.astype(int) @ adj.astype(int)
    same = common >= n // 2 - 3          # mismos rivales de zona (≈12) frente a interzonales (≈2)
    z = -np.ones(n, int); z[0] = 0
    for _ in range(n):
        for i in range(n):
            if z[i] >= 0:
                z[same[i] & (z < 0)] = z[i]
    rest = z < 0
    z[rest] = 1
    if (z == 0).sum() != (z == 1).sum():  # respaldo: la mayoría de rivales define la zona
        zA = z == 0
        for i in range(n):
            z[i] = 0 if adj[i, zA].sum() >= adj[i, ~zA].sum() else 1
    return z


def season_state_split(matches, comp, syear, team_index, asof, league):
    from ..leagues import PROMEDIO_BASE, point_adjustments
    from .season import SeasonState
    asof_given = asof
    all_ = matches[(matches["comp"] == comp) & (matches["syear"] == syear)].copy()
    d = all_[all_["stage"] == "regular"].copy()
    teams = sorted(set(d["home"]) | set(d["away"]), key=int)
    loc = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    d["hi"] = d["home"].map(loc); d["ai"] = d["away"].map(loc)
    d["tourn"] = d["slug"].map(_tourn)
    d = d[d["tourn"].notna()].copy()
    d["tourn"] = d["tourn"].astype(int)
    if asof is None:
        played = d["finished"]
        asof = all_.loc[all_["finished"], "date"].max() if all_["finished"].any() else d["date"].min()
    else:
        played = d["finished"] & (d["date"] <= asof)
    # zonas (del Apertura, que se publica completo; si no, del Clausura)
    base = d[d["tourn"] == 0] if (d["tourn"] == 0).any() else d[d["tourn"] == 1]
    zone = infer_zones(list(zip(base["hi"], base["ai"])), n)
    anchor = ZONE_A_ANCHOR.get(comp)          # solo para el nombre: la zona de este equipo es la "A"
    if anchor in loc and zone[loc[anchor]] == 1:
        zone = 1 - zone
    # Clausura: el fixture del Apertura con las localías invertidas
    ap = set(zip(d.loc[d.tourn == 0, "hi"], d.loc[d.tourn == 0, "ai"]))
    cl = set(zip(d.loc[d.tourn == 1, "hi"], d.loc[d.tourn == 1, "ai"]))
    missing = sorted({(a, h) for h, a in ap} - cl)
    last = d.loc[d.tourn == 1, "date"].max() if (d.tourn == 1).any() else pd.Timestamp(f"{syear}-12-01", tz="UTC")
    done = d[played][["hi", "ai", "hg", "ag", "date", "tourn"]].reset_index(drop=True)
    todo = d[~played][["hi", "ai", "date", "neutral", "tourn"]].copy()
    todo["synthetic"] = False
    if missing:
        todo = pd.concat([todo, pd.DataFrame([(h, a, last, False, 1, True) for h, a in missing],
                                             columns=["hi", "ai", "date", "neutral", "tourn", "synthetic"])])
    todo["days_ahead"] = np.maximum((todo["date"] - asof).dt.total_seconds() / 86400.0, 0.0)
    todo = todo.sort_values("days_ahead").reset_index(drop=True)
    todo["tourn"] = todo["tourn"].astype(int)
    # torneo en curso: el Apertura hasta que se juega su final
    po = all_[all_["stage"] == "playoff"].copy()
    po["tourn"] = po["slug"].map(_tourn)
    po_fin = po[po["finished"] & (po["date"] <= asof)]
    ap_final = po_fin[(po_fin["tourn"] == 0) & po_fin["slug"].astype(str).str.endswith("final")]
    cur = 1 if len(ap_final) else 0
    # resultados de playoffs ya jugados en el torneo en curso: {(i, j): ganador}
    known = {}
    cur_po = po[po["tourn"] == cur].sort_values("date")
    for r in cur_po[cur_po["finished"] & (cur_po["date"] <= asof)].itertuples():
        if r.home not in loc or r.away not in loc:
            continue
        i, j = loc[r.home], loc[r.away]
        if r.hg != r.ag:
            w = i if r.hg > r.ag else j
        else:   # por penales: ganó el que aparece en la ronda siguiente
            later = cur_po[cur_po["date"] > r.date]
            nxt = set(later["home"]) | set(later["away"])
            w = i if r.home in nxt else (j if r.away in nxt else -1)
        known[(min(i, j), max(i, j))] = w
    # base de los promedios
    pb = PROMEDIO_BASE.get((comp, int(syear)))
    if pb:
        from ..data import store
        _, tm = store.load()
        nm = dict(zip(tm["team_id"], tm["name"]))
        prev = np.array([pb.get(nm.get(t, ""), (0, 0)) for t in teams], float)
    elif int(syear) - 2 >= league.split_since:
        # desde 2027 las dos temporadas anteriores están completas en los datos (fase regular)
        prev = np.zeros((n, 2))
        old = matches[(matches["comp"] == comp) & (matches["stage"] == "regular") & matches["finished"]
                      & matches["syear"].isin([int(syear) - 1, int(syear) - 2])]
        for h, a, x, y in old[["home", "away", "hg", "ag"]].itertuples(index=False):
            if h in loc:
                prev[loc[h], 0] += 3 * (x > y) + (x == y); prev[loc[h], 1] += 1
            if a in loc:
                prev[loc[a], 0] += 3 * (y > x) + (x == y); prev[loc[a], 1] += 1
    else:
        prev = None
    games = np.bincount(np.r_[d["hi"], d["ai"], todo.loc[todo.synthetic, "hi"], todo.loc[todo.synthetic, "ai"]],
                        minlength=n).astype(float)
    adjd = point_adjustments(comp, syear, None if asof_given is None else asof_given)
    adj = np.array([adjd.get(t, 0) for t in teams], float)
    mi = np.array([team_index[t] for t in teams])
    st = SeasonState(teams, mi, done, todo, adj, comp, int(syear))
    st.split = dict(zone=zone, cur=cur, known=known, prev=prev, games=games)
    return st


def tournament_table_now(st):
    """Puntos y partidos jugados de cada equipo en el torneo en curso."""
    n = len(st.teams); cur = st.split["cur"]
    p = np.zeros(n); pj = np.zeros(n)
    for h, a, x, y, t in st.done[["hi", "ai", "hg", "ag", "tourn"]].itertuples(index=False):
        if t == cur:
            p[h] += 3 * (x > y) + (x == y); p[a] += 3 * (y > x) + (x == y); pj[h] += 1; pj[a] += 1
    return p, pj


def _rank_rows(keys: list[np.ndarray]) -> np.ndarray:
    """Orden descendente lexicográfico por fila: devuelve el índice de columna en cada puesto."""
    k = np.zeros_like(keys[0], float)
    for x in keys:   # combina en una sola clave (escalas separadas)
        k = k * 1000.0 + (x - x.min())
    return np.argsort(-k, axis=1, kind="stable")


def split_outcomes(st, gx, gy, pos, pts, X, n, params, rng):
    """Tabla de zonas del torneo en curso, campeón y descendidos de cada temporada simulada."""
    from .season import knockout
    sp = st.split
    S = pos.shape[0]
    zone, cur = sp["zone"], sp["cur"]
    zp = np.zeros((S, n)); zgd = np.zeros((S, n)); zgf = np.zeros((S, n))
    for h, a, x, y, t in st.done[["hi", "ai", "hg", "ag", "tourn"]].itertuples(index=False):
        if t != cur:
            continue
        zp[:, h] += 3 * (x > y) + (x == y); zp[:, a] += 3 * (y > x) + (x == y)
        zgd[:, h] += x - y; zgd[:, a] += y - x; zgf[:, h] += x; zgf[:, a] += y
    tt = st.todo["tourn"].to_numpy()
    hi, ai = st.todo["hi"].to_numpy(), st.todo["ai"].to_numpy()
    for k in np.where(tt == cur)[0]:
        x, y = gx[:, k], gy[:, k]; h, a = hi[k], ai[k]
        zp[:, h] += 3 * (x > y) + (x == y); zp[:, a] += 3 * (y > x) + (x == y)
        zgd[:, h] += x - y; zgd[:, a] += y - x; zgf[:, h] += x; zgf[:, a] += y
    rnd = rng.random((S, n))
    zpos = np.zeros((S, n), int)
    order = {}
    for z in (0, 1):
        cols = np.where(zone == z)[0]
        o = _rank_rows([zp[:, cols], zgd[:, cols] + 200, zgf[:, cols], rnd[:, cols]])
        order[z] = cols[o]                                   # (S, 15) equipo en cada puesto
        zpos[np.arange(S)[:, None], order[z]] = np.arange(len(cols))[None, :]
    # llave: el mejor ubicado (puesto en su zona, luego puntos) es local
    known = sp["known"]

    def tie(a, b, neutral=False):
        better = (zpos[np.arange(S), a] < zpos[np.arange(S), b]) | (
            (zpos[np.arange(S), a] == zpos[np.arange(S), b]) & (zp[np.arange(S), a] >= zp[np.arange(S), b]))
        A, B = np.where(better, a, b), np.where(better, b, a)
        w = knockout(X, A, B, n, params, rng, legs=1, neutral=neutral)
        for (i, j), win in known.items():
            if win < 0:
                continue
            m = ((A == i) & (B == j)) | ((A == j) & (B == i))
            w[m] = win
        return w

    r16 = [tie(order[za][:, pa], order[zb][:, pb]) for (za, pa), (zb, pb) in R16]
    qf = [tie(r16[2 * i], r16[2 * i + 1]) for i in range(4)]
    sf = [tie(qf[0], qf[1]), tie(qf[2], qf[3])]
    champ = tie(sf[0], sf[1], neutral=True)
    title = np.zeros((S, n), bool); title[np.arange(S), champ] = True
    # descensos (si hay base de promedios): último de promedios y último de la tabla anual (sin él)
    rel = np.zeros((S, n), bool)
    if sp["prev"] is not None:
        prev = sp["prev"]
        prom = (prev[:, 0][None, :] + pts) / np.maximum(prev[:, 1][None, :] + sp["games"][None, :], 1)
        prom = prom + 1e-9 * rng.random((S, n))   # empates exactos: sorteo (en rigor, desempate)
        r1 = prom.argmin(1)
        rel[np.arange(S), r1] = True
        ordr = np.argsort(pos, axis=1)[:, ::-1]   # del último hacia arriba
        first = np.where(ordr[:, 0] == r1, ordr[:, 1], ordr[:, 0])
        second = np.where(ordr[:, 0] == r1, ordr[:, 2], np.where(ordr[:, 1] == r1, ordr[:, 2], ordr[:, 1]))
        # empate en puntos por el último lugar de la tabla anual: partido de desempate neutral
        tie_m = pts[np.arange(S), first] == pts[np.arange(S), second]
        r2 = first.copy()
        if tie_m.any():
            w = knockout(X, second, first, n, params, rng, legs=1, neutral=True)
            r2 = np.where(tie_m, np.where(w == first, second, first), first)
        rel[np.arange(S), r2] = True
    return dict(zpos=zpos, zpts=zp, title=title, rel=rel, zone=zone)
