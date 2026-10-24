r"""Análisis sobre las simulaciones: importancia de partidos, escenarios y eliminación matemática.

Importancia como información mutua
----------------------------------
Sea O_k \in {L, E, V} el resultado del partido pendiente k y F una variable de la tabla final
(el campeón, el conjunto de descendidos, la posición de un equipo). La importancia de k para F es

    I(O_k; F) = sum_{o,f} p(o, f) log2 [ p(o, f) / (p(o) p(f)) ]      (bits)

Mide cuánto se reduce, en promedio, la incertidumbre sobre F al conocer el resultado de k.
Es simétrica, no negativa, nula si y solo si O_k y F son independientes, y está acotada por
min(H(O_k), H(F)) <= log2 3. Con S simulaciones, el estimador plug-in tiene un sesgo positivo
~ (|O|-1)(|F|-1) / (2 S ln 2) (Miller, 1955), que se resta (corrección de Miller–Madow).

Para eventos binarios de un equipo (título, descenso) también se reporta el "apalancamiento"
(leverage) max_o P(E | O_k=o) - min_o P(E | O_k=o): cuánto mueve el partido la probabilidad.

Eliminación y clasificación matemática
--------------------------------------
Con 3-1-0, decidir si un equipo todavía puede salir campeón es NP-completo (Kern & Paulusma,
2001), así que no hay un flujo máximo como en el béisbol. Se resuelve como programa entero
(HiGHS): por cada partido pendiente k tres binarias x_kL + x_kE + x_kV = 1, puntos finales
P_t = P_t^0 + sum_k (3 x_kL + x_kE)[t local] + (3 x_kV + x_kE)[t visita], y

    mejor posición de i  = 1 + min  #{ j : P_j > P_i }      (empates a favor)
    peor posición de i   = 1 + max  #{ j : P_j >= P_i }     (empates en contra)

con indicadores linealizados por big-M. Si la peor posición es 1, el título está asegurado;
si la mejor posición está fuera de una zona, el equipo está matemáticamente eliminado de ella.
Los empates se tratan de forma conservadora porque la diferencia de gol final no está acotada
por los puntos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .season import SeasonState, SimResult


# ----------------------------------------------------------------------------------------
# Resultados simulados
# ----------------------------------------------------------------------------------------
def outcomes(res: SimResult) -> np.ndarray:
    """(S, M) con 0 = gana local, 1 = empate, 2 = gana visita."""
    gx, gy = res.goals
    return np.where(gx > gy, 0, np.where(gx == gy, 1, 2))


def _mi_bits(a: np.ndarray, b: np.ndarray, na: int, nb: int, miller_madow=True) -> float:
    S = len(a)
    joint = np.zeros((na, nb))
    np.add.at(joint, (a, b), 1.0)
    joint /= S
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float((joint[nz] * np.log2(joint[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
    if miller_madow:
        ka, kb = (pa > 0).sum(), (pb > 0).sum()
        mi -= (ka - 1) * (kb - 1) / (2 * S * np.log(2))
    return max(mi, 0.0)


def match_importance(res: SimResult, n_relegated: int, top: int | None = None) -> pd.DataFrame:
    """Importancia de cada partido pendiente para el título y para el descenso."""
    O = outcomes(res)
    S, M = O.shape
    n = len(res.teams)
    champ = res.pos.argmin(1)  # equipo en la posición 0
    rel = res.pos >= n - n_relegated  # (S, n)
    if res.extra is not None:      # formato argentino: campeón del torneo y descendidos reales
        champ = res.extra["title"].argmax(1)
        rel = res.extra["rel"]
        n_relegated = 2
    rows = []
    synth = res.todo["synthetic"].to_numpy() if "synthetic" in res.todo else np.zeros(M, bool)
    for k in range(M):
        if synth[k]:   # sin fecha publicada: no se presenta como partido decisivo
            continue
        o = O[:, k]
        i_title = _mi_bits(o, champ, 3, n)
        # descenso: suma sobre equipos de I(O_k; R_i)  (cota superior de I(O_k; R) por subaditividad
        # no aplica en general; se usa como resumen interpretable)
        i_rel = sum(_mi_bits(o, rel[:, t].astype(int), 3, 2) for t in range(n)) if n_relegated else 0.0
        # ascenso (ligas con playoffs): mismo resumen que el descenso, sobre "asciende" de cada equipo
        i_pro = (sum(_mi_bits(o, res.promoted[:, t].astype(int), 3, 2) for t in range(n))
                 if res.promoted is not None else 0.0)
        p = np.bincount(o, minlength=3) / S
        rows.append(dict(k=k, home=res.todo["hi"].iloc[k], away=res.todo["ai"].iloc[k],
                         date=res.todo["date"].iloc[k], p_home=p[0], p_draw=p[1], p_away=p[2],
                         mi_title=i_title, mi_relegation=i_rel, mi_promotion=i_pro))
    df = pd.DataFrame(rows)
    top_col = "mi_promotion" if res.promoted is not None else "mi_title"
    df["mi_total"] = df[top_col] + df["mi_relegation"]
    df = df.sort_values("mi_total", ascending=False)
    return df.head(top) if top else df


def leverage(res: SimResult, team: int, event: np.ndarray, k: int) -> dict:
    """P(evento | resultado del partido k) para un evento booleano (S,) de un equipo."""
    o = outcomes(res)[:, k]
    out = {}
    for code, name in enumerate(("home", "draw", "away")):
        m = o == code
        out[name] = float(event[m].mean()) if m.any() else np.nan
    vals = [v for v in out.values() if not np.isnan(v)]
    out["leverage"] = max(vals) - min(vals) if vals else 0.0
    return out


# ----------------------------------------------------------------------------------------
# Escenarios condicionales
# ----------------------------------------------------------------------------------------
def condition(res: SimResult, fixed: dict[int, int]) -> np.ndarray:
    """Máscara (S,) de las simulaciones compatibles con resultados fijados {k: 0|1|2}."""
    O = outcomes(res)
    m = np.ones(O.shape[0], bool)
    for k, o in fixed.items():
        m &= O[:, k] == o
    return m


def position_matrix(pos: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    p = pos if mask is None else pos[mask]
    n = pos.shape[1]
    return np.stack([(p == k).mean(0) for k in range(n)], 1)


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Regresión isotónica creciente (pool adjacent violators) con pesos."""
    blocks = []   # [suma w·y, suma w, largo]
    for yi, wi in zip(y, w):
        blocks.append([yi * wi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            a, b = blocks.pop(), blocks.pop()
            blocks.append([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
    out = []
    for sw, ww, k in blocks:
        out += [sw / ww] * k
    return np.array(out)


def _shrunk_rate(flag: np.ndarray, pos: np.ndarray, team: int, npos: int, m: float = 50.0) -> np.ndarray:
    """P(flag | equipo, puesto k) para k = 0..npos-1, estimada con las temporadas simuladas: la del
    equipo, encogida hacia la de cualquier equipo en ese puesto (m simulaciones de peso)."""
    out = np.zeros(npos)
    for k in range(npos):
        at = pos == k
        pooled = flag[at].mean() if at.any() else 0.0
        mine = at[:, team]
        out[k] = (flag[mine, team].sum() + m * pooled) / (mine.sum() + m)
    return out


def _rank(keys_desc: np.ndarray, tiebreak_asc: np.ndarray) -> np.ndarray:
    """Puesto (0-based) por fila: más puntos primero; empate, el orden original de la simulación."""
    order = np.lexsort((tiebreak_asc, -keys_desc), axis=1)
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(order.shape[1])[None, :], axis=1)
    return pos


def need_curves(res: SimResult, st: SeasonState, team: int, n_rel: int, league=None, m=None, P=None,
                params=None, N_model=None, thetas=(-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
                seed: int = 0) -> dict:
    """'¿Qué necesita?': P(objetivo | puntos que sume desde hoy), para todos los puntos posibles.

    Contar éxitos entre las temporadas simuladas en que el equipo sumó exactamente g es exacto pero
    deja sin datos los extremos (un equipo muy superior casi nunca suma pocos puntos; uno modesto casi
    nunca gana todo). Aquí, en cada temporada simulada se vuelven a sortear solo los resultados de los
    partidos que le quedan al equipo, con P_θ(o) ∝ P(o | fuerzas de esa temporada)·e^{θ·puntos(o)}
    para una grilla de θ que lleva su total desde casi cero hasta casi todos los puntos; sus rivales
    ganan o pierden esos mismos partidos y el resto de la temporada queda como se simuló (dadas las
    fuerzas, los partidos son independientes). Como la inclinación depende solo de los puntos,
    condicionada a un total g deja intacta la distribución de qué partidos gana: dentro de cada g
    todas las muestras valen lo mismo y se pueden juntar entre θ. Lo único que no se recalcula es la
    fuerza del propio equipo dado g (no influye en sus rivales salvo por la correlación a posteriori).
    La tabla se reordena por puntos, con el orden original de la simulación como desempate.

    En la Championship, el ascenso por playoffs y, en Argentina, ganar el torneo se toman de
    P(lo logra | equipo, puesto), estimada con todas las temporadas simuladas."""
    params = params or {}
    rng = np.random.default_rng(seed + 7919 * (team + 1))
    from .season import outcome_probs_given
    n = len(st.teams)
    S = res.pts.shape[0]
    hi, ai = st.todo["hi"].to_numpy(), st.todo["ai"].to_numpy()
    ks = np.where((hi == team) | (ai == team))[0]
    opp = np.where(hi[ks] == team, ai[ks], hi[ks])
    home = hi[ks] == team
    x, y = res.goals[0][:, ks], res.goals[1][:, ks]
    o0 = np.where(x > y, 0, np.where(x == y, 1, 2))                       # resultado simulado (local)
    PT = np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])                     # puntos del equipo: local / visita
    mine0 = np.where(home[None, :], PT[0][o0], PT[1][o0])                 # puntos del equipo en cada partido
    theirs0 = 3.0 * (o0 != 1) + 1.0 * (o0 == 1) - mine0                    # puntos del rival
    base = res.pts.astype(float).copy()
    base[:, team] -= mine0.sum(1)
    np.subtract.at(base.T, opp, theirs0.T)                                # sin esos partidos
    pall = getattr(res, "_pall", None)
    if pall is None:   # P(L, E, V | fuerzas) de todos los partidos pendientes, una vez por corrida
        pall = np.concatenate([outcome_probs_given(res.rates[0][:, c:c + 40], res.rates[1][:, c:c + 40], params,
                                                   params.get("kind", "dc")).astype(np.float32)
                               for c in range(0, res.rates[0].shape[1], 40)], axis=1)
        res._pall = pall
    p = pall[:, ks].astype(float)
    ptm = np.where(home[None, :, None], PT[0][None, None, :], PT[1][None, None, :])   # (1, K, 3)
    X = res.extra
    cur = st.split["cur"] if st.split is not None else None
    in_cur = (st.todo["tourn"].to_numpy()[ks] == cur) if X is not None else np.ones(len(ks), bool)
    # tasas por puesto para los objetivos que dependen de una llave
    w_asc = _shrunk_rate(res.promoted, res.pos, team, n) if res.promoted is not None else None
    if X is not None:
        zone = X["zone"]; zcols = [np.where(zone == z)[0] for z in (0, 1)]
        q_title = _shrunk_rate(X["title"], X["zpos"], team, 16)
        zbase = X["zpts"].astype(float).copy()
        mc = mine0[:, in_cur]; tc = theirs0[:, in_cur]
        zbase[:, team] -= mc.sum(1)
        np.subtract.at(zbase.T, opp[in_cur], tc.T)
        sp = st.split
    rows = {}
    for th in thetas:
        q = p * np.exp(th * ptm); q /= q.sum(-1, keepdims=True)
        u = rng.random(q.shape[:2])[..., None]
        o = (u > np.cumsum(q, -1)).sum(-1)
        mine = np.where(home[None, :], PT[0][o], PT[1][o]); theirs = 3.0 * (o != 1) + 1.0 * (o == 1) - mine
        pts = base.copy(); pts[:, team] += mine.sum(1)
        np.add.at(pts.T, opp, theirs.T)
        g = mine.sum(1).astype(int)
        pos = _rank(pts, res.pos)
        hits = {}
        if X is None:
            hits["title"] = (pos[:, team] == 0, g)
            if n_rel:
                hits["survival"] = (pos[:, team] < n - n_rel, g)
            if w_asc is not None:
                hits["promotion"] = (w_asc[pos[:, team]], g)
        else:
            zp = zbase.copy(); zp[:, team] += mine[:, in_cur].sum(1)
            np.add.at(zp.T, opp[in_cur], theirs[:, in_cur].T)
            zg = mine[:, in_cur].sum(1).astype(int)
            zpos = np.zeros((S, n), int)
            for zc in zcols:
                zr = _rank(zp[:, zc], X["zpos"][:, zc])
                zpos[:, zc] = zr
            hits["title"] = (q_title[zpos[:, team]], zg)
            hits["playoffs"] = (zpos[:, team] < 8, zg)
            if sp.get("prev") is not None:
                prom = (sp["prev"][:, 0][None, :] + pts) / np.maximum(sp["prev"][:, 1][None, :] + sp["games"][None, :], 1)
                r1 = prom.argmin(1)
                last = np.argsort(pos, axis=1)[:, ::-1]
                r2 = np.where(last[:, 0] == r1, last[:, 1], last[:, 0])
                hits["survival"] = ((r1 != team) & (r2 != team), g)
        for k_, (h, gv) in hits.items():
            rows.setdefault(k_, []).append((np.asarray(h, float), gv))
    out = {}
    now = st.points_now()
    for key, lst in rows.items():
        h = np.concatenate([a for a, _ in lst]); gv = np.concatenate([b for _, b in lst])
        if X is not None and key in ("title", "playoffs"):
            from .split import tournament_table_now
            b0 = tournament_table_now(st)[0][team]; G_ = 3 * int(in_cur.sum()) + 1
        else:
            b0 = now[team]; G_ = 3 * len(ks) + 1
        cnt = np.bincount(gv, minlength=G_).astype(float)
        hs = np.bincount(gv, weights=h, minlength=G_)
        gg = np.arange(G_)
        pr = np.where(cnt > 0, hs / np.maximum(cnt, 1), np.nan)
        ok = ~np.isnan(pr)
        if ok.sum() == 0:
            pr = np.zeros(G_)
        elif (~ok).any():   # totales que no se pueden sumar (p. ej. 3·partidos − 1): se interpola
            pr = np.interp(gg, gg[ok], pr[ok])
        pr = np.clip(_pava(pr, np.maximum(cnt, 1e-3)), 0, 1)
        direct = np.bincount(np.clip(np.rint(res.pts[:, team] - now[team]).astype(int), 0, G_ - 1), minlength=G_) \
            if key not in ("title", "playoffs") or X is None else cnt * 0
        df = pd.DataFrame(dict(points_gained=gg, final_points=(b0 + gg).astype(int), n=cnt.astype(int),
                               p_reach=direct / S, p_objective=pr))
        for qv in (0.5, 0.9):
            okq = df[df["p_objective"] >= qv]
            df.attrs[f"need_{int(qv * 100)}"] = int(okq["points_gained"].min()) if len(okq) else None
        out[key] = df
    return out


def points_curve(res: SimResult, st: SeasonState, team: int, positions: list[int] | None = None,
                 hit: np.ndarray | None = None) -> pd.DataFrame:
    """Versión directa (solo simulaciones): se mantiene para análisis; el sitio usa ``need_curves``."""
    now = st.points_now()
    gained = res.pts[:, team] - now[team]
    if hit is None:
        hit = np.isin(res.pos[:, team], positions)
    rows = []
    for g in np.unique(gained):
        m = gained == g
        rows.append(dict(points_gained=int(g), final_points=int(now[team] + g), n=int(m.sum()),
                         p_reach=float(m.mean()), p_objective=float(hit[m].mean())))
    df = pd.DataFrame(rows)
    for q in (0.5, 0.9):
        ok = df[(df["p_objective"] >= q) & (df["n"] >= 30)]
        df.attrs[f"need_{int(q * 100)}"] = int(ok["points_gained"].min()) if len(ok) else None
    return df


# ----------------------------------------------------------------------------------------
# Eliminación / clasificación matemática (programa entero)
# ----------------------------------------------------------------------------------------
def _points_now(st: SeasonState) -> np.ndarray:
    return st.points_now()


def position_bounds(st: SeasonState, time_limit: float = 20.0) -> pd.DataFrame:
    """Mejor y peor posición matemáticamente posibles para cada equipo."""
    n = len(st.teams)
    P0 = _points_now(st)
    todo = st.todo
    M = len(todo)
    hi, ai = todo["hi"].to_numpy(), todo["ai"].to_numpy()
    rem = np.bincount(np.r_[hi, ai], minlength=n)
    BIG = 3 * rem.max() + P0.max() - P0.min() + 3 if M else 1
    nx = 3 * M
    out = []
    for i in range(n):
        res = {}
        for sense in ("best", "worst"):
            # variables: x (3M) + y_j (n)
            nv = nx + n
            c = np.zeros(nv)
            c[nx:] = 1.0 if sense == "best" else -1.0
            c[nx + i] = 0.0
            A = lil_matrix((M + n, nv))
            lb, ub = np.zeros(M + n), np.zeros(M + n)
            for k in range(M):
                A[k, 3 * k:3 * k + 3] = 1
                lb[k] = ub[k] = 1
            # fila para cada j != i: P_j - P_i (lineal en x) contra y_j
            for j in range(n):
                r = M + j
                if j == i:
                    lb[r], ub[r] = -np.inf, np.inf
                    continue
                coef = np.zeros(nx)
                for k in range(M):
                    for t, sgn in ((j, 1.0), (i, -1.0)):
                        if hi[k] == t:
                            coef[3 * k] += 3 * sgn; coef[3 * k + 1] += sgn
                        if ai[k] == t:
                            coef[3 * k + 2] += 3 * sgn; coef[3 * k + 1] += sgn
                d0 = P0[j] - P0[i]
                nzk = np.nonzero(coef)[0]
                for kk in nzk:
                    A[r, kk] = coef[kk]
                if sense == "best":
                    # y_j = 0  =>  P_j <= P_i :   coef.x + d0 - BIG y_j <= 0
                    A[r, nx + j] = -BIG
                    lb[r], ub[r] = -np.inf, -d0
                else:
                    # y_j = 1  =>  P_j >= P_i :   coef.x + d0 + BIG(1 - y_j) >= 0
                    A[r, nx + j] = -BIG
                    lb[r], ub[r] = -d0 - BIG, np.inf
            integrality = np.ones(nv)
            bnds = Bounds(np.zeros(nv), np.ones(nv))
            if sense == "best":
                bnds.ub[nx + i] = 0
            else:
                bnds.ub[nx + i] = 0
            r_ = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integrality,
                      bounds=bnds, options=dict(time_limit=time_limit))
            if r_.x is None:
                res[sense] = np.nan
            else:
                cnt = int(round(np.sum(r_.x[nx:])))
                res[sense] = 1 + cnt
        out.append(dict(team=st.teams[i], points=P0[i], remaining=int(rem[i]),
                        best_position=res["best"], worst_position=res["worst"]))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------
# Competitividad y suspenso
# ----------------------------------------------------------------------------------------
def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def surprise_suspense(beliefs: np.ndarray) -> dict:
    """Ely, Frankel & Kamenica (2015) sobre una secuencia de creencias (T, K) sobre el campeón.

    sorpresa_t = || mu_t - mu_{t-1} ||_2^2   (cuánto cambió la creencia)
    La sorpresa total sum_t sorpresa_t mide la "emoción realizada" de la temporada.
    """
    d = np.diff(beliefs, axis=0)
    s = (d ** 2).sum(1)
    return dict(surprise=s, total_surprise=float(s.sum()),
                entropy=np.array([entropy_bits(b) for b in beliefs]))


def balance_metrics(points: np.ndarray, matches_per_team: int, draw_rate: float) -> dict:
    """Balance competitivo estático de una tabla de puntos final."""
    p = np.asarray(points, float)
    share = p / p.sum()
    hhi = float((share ** 2).sum())
    n = len(p)
    g = np.sort(p)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(g) / (n * g.sum()))
    # Noll–Scully con empates: sd de puntos por partido frente a la de una liga de equipos iguales
    ppg = p / matches_per_team
    pw = (1 - draw_rate) / 2
    var_ideal = (9 * pw + 1 * draw_rate) - (3 * pw + draw_rate) ** 2
    ns = float(ppg.std(ddof=0) / np.sqrt(var_ideal / matches_per_team))
    return dict(hhi=hhi, hhi_norm=hhi * n, gini=gini, noll_scully=ns)
