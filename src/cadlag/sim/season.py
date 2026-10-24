r"""Simulación Monte Carlo de lo que queda de una temporada.

Dos modos, que es justamente lo que se compara en el backtest:

``posterior`` (predictiva posterior)
    Por simulación se sortea UNA vez el estado actual x ~ N(m, P) y luego se hace evolucionar
    el paseo aleatorio latente hasta la fecha de cada partido pendiente. Los goles se sortean
    de Dixon–Coles con las fuerzas *de esa simulación*. Así la incertidumbre sobre lo bueno
    que es un equipo se propaga a todos sus partidos a la vez: por la ley de varianza total,
        Var(Pts_i) = E[Var(Pts_i | x)] + Var(E[Pts_i | x]),
    y el segundo término, que crece como el cuadrado del número de partidos restantes, es el
    que desaparece en el modo plug-in.

``plugin``
    Las fuerzas quedan fijas en la media m y no evolucionan: el error clásico de simular con
    el estimador puntual.

Después se arma la tabla con las reglas reales (desempates recursivos y partidos de
definición) y se cuentan las posiciones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..leagues import CRIT_CODE, League
from . import tiebreak


@dataclass
class SeasonState:
    """Lo que el simulador necesita saber de la temporada en curso."""
    teams: list            # ids ESPN de los equipos de la temporada (orden local 0..n-1)
    model_index: np.ndarray  # índice de cada equipo en el vector de estado del modelo
    done: pd.DataFrame     # partidos jugados (hi, ai, hg, ag) en índices locales
    todo: pd.DataFrame     # partidos por jugar (hi, ai, days_ahead, neutral)
    adj: np.ndarray | None = None  # ajustes de puntos (descuentos) por equipo local, ya conocidos
    comp: str | None = None
    syear: int | None = None
    split: dict | None = None      # formato argentino (ver sim.split)

    def points_now(self) -> np.ndarray:
        """Puntos actuales: resultados jugados más los ajustes."""
        p = np.zeros(len(self.teams)) if self.adj is None else self.adj.astype(float).copy()
        for h, a, x, y in self.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
            p[h] += 3 * (x > y) + (x == y)
            p[a] += 3 * (y > x) + (x == y)
        return p


def season_state(matches: pd.DataFrame, comp: str, syear: int, team_index: dict,
                 asof: pd.Timestamp | None = None, complete_round_robin: bool = True) -> SeasonState:
    """Estado de la temporada ``syear`` a la fecha ``asof`` (por defecto, ahora)."""
    from ..leagues import LEAGUES
    lg = LEAGUES.get(comp)
    if lg is not None and lg.split_since and int(syear) >= lg.split_since:
        from .split import season_state_split
        return season_state_split(matches, comp, syear, team_index, asof, lg)
    if lg is not None and lg.split_since:
        complete_round_robin = False   # formatos argentinos anteriores: solo el fixture publicado
    asof_given = asof
    d = matches[(matches["comp"] == comp) & (matches["syear"] == syear) & (matches["stage"] == "regular")].copy()
    teams = sorted(set(d["home"]) | set(d["away"]), key=int)
    loc = {t: i for i, t in enumerate(teams)}
    d["hi"] = d["home"].map(loc)
    d["ai"] = d["away"].map(loc)
    if asof is None:
        played = d["finished"]
        asof = d.loc[played, "date"].max() if played.any() else d["date"].min()
    else:
        played = d["finished"] & (d["date"] <= asof)
    done = d[played][["hi", "ai", "hg", "ag", "date"]].reset_index(drop=True)
    todo = d[~played][["hi", "ai", "date", "neutral"]].copy()
    todo["synthetic"] = False
    if complete_round_robin:
        # partidos que ESPN todavía no publica (la Championship sale por tramos): sin fecha, se
        # ubican al final del calendario y se marcan para no presentarlos como partidos reales
        have = set(zip(d["hi"], d["ai"]))
        last = d["date"].max()
        extra = [(h, a, last, False, True) for h in range(len(teams)) for a in range(len(teams))
                 if h != a and (h, a) not in have]
        if extra:
            todo = pd.concat([todo, pd.DataFrame(extra, columns=["hi", "ai", "date", "neutral", "synthetic"])])
    todo["days_ahead"] = np.maximum((todo["date"] - asof).dt.total_seconds() / 86400.0, 0.0)
    todo = todo.sort_values("days_ahead").reset_index(drop=True)
    mi = np.array([team_index[t] for t in teams])
    from ..leagues import point_adjustments
    known = point_adjustments(comp, syear, None if asof_given is None else asof_given)
    adj = np.array([known.get(t, 0) for t in teams], float)
    return SeasonState(teams, mi, done, todo, adj, comp, int(syear))


# ----------------------------------------------------------------------------------------
# Muestreo de Dixon–Coles por rechazo (exacto)
# ----------------------------------------------------------------------------------------
def _tau(x, y, lam, nu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * nu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + nu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.maximum(t, 1e-12)


def sample_dc(lam: np.ndarray, nu: np.ndarray, rho: float, rng: np.random.Generator):
    """Sortea (X, Y) ~ DC(lam, nu, rho) por aceptación–rechazo sobre Poisson independientes.

    Propuesta: Poisson(lam) x Poisson(nu); se acepta con probabilidad tau(x, y)/tau_max,
    tau_max = max(1, 1 - lam nu rho, 1 + lam rho, 1 + nu rho, 1 - rho). Es exacto porque DC
    es la densidad Poisson-producto multiplicada por tau (que conserva la masa).
    """
    x = rng.poisson(lam)
    y = rng.poisson(nu)
    tmax = np.maximum.reduce([np.ones_like(lam), 1 - lam * nu * rho, 1 + lam * rho, 1 + nu * rho,
                              np.full_like(lam, 1 - rho)])

    def tau(x, y):
        t = np.ones_like(lam)
        t = np.where((x == 0) & (y == 0), 1 - lam * nu * rho, t)
        t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
        t = np.where((x == 1) & (y == 0), 1 + nu * rho, t)
        t = np.where((x == 1) & (y == 1), 1 - rho, t)
        return t

    rej = rng.random(lam.shape) * tmax > tau(x, y)
    while rej.any():
        x[rej] = rng.poisson(lam[rej])
        y[rej] = rng.poisson(nu[rej])
        rej_new = rng.random(lam.shape) * tmax > tau(x, y)
        rej = rej & rej_new
    return x, y


# ----------------------------------------------------------------------------------------
# Simulador
# ----------------------------------------------------------------------------------------
@dataclass
class SimResult:
    teams: list
    pos: np.ndarray          # (S, n) posición 0-indexada
    pts: np.ndarray          # (S, n)
    goals: tuple             # (X, Y) de los partidos pendientes, (S, M)
    todo: pd.DataFrame
    logw: np.ndarray | None = None  # log razón de verosimilitudes si se inclinaron los resultados
    gd: np.ndarray | None = None    # (S, n) diferencia de gol final
    probs: np.ndarray | None = None  # (S, K, 3) P(L, E, V | fuerzas sorteadas) de los primeros K partidos
    promoted: np.ndarray | None = None  # (S, n) bool: asciende (directo o por playoffs), si la liga asciende
    po_winner: np.ndarray | None = None  # (S,) índice local del ganador de los playoffs de ascenso
    extra: dict | None = None        # formato argentino: zpos, zpts, title, rel, zone (ver sim.split)
    rates: tuple | None = None       # (λ, ν) sin inclinar de cada partido pendiente, (S, M) cada una
    x0: np.ndarray | None = None     # estado local sorteado de cada temporada (con keep_rates)

    def position_matrix(self) -> np.ndarray:
        n = len(self.teams)
        M = np.zeros((n, n))
        for k in range(n):
            M[:, k] = (self.pos == k).mean(0)
        return M


def simulate(league: League, st: SeasonState, m: np.ndarray, P: np.ndarray, params: dict,
             N_model: int, n_sims: int = 10000, mode: str = "posterior", seed: int = 0,
             x_draw: np.ndarray | None = None, chunk: int = 1000,
             tilt: tuple | None = None, probs_k: int = 0, boost: tuple | None = None,
             keep_rates: bool = False) -> SimResult:
    """``tilt=(t, θ)`` sortea los partidos del equipo local t con su tasa de goles × e^{-θ} y la del
    rival × e^{+θ} (muestreo por importancia, ver ``sim.rare``) y devuelve en ``logw`` el log de la
    razón de verosimilitudes de Dixon–Coles, exacta partido a partido.

    ``params["kind"] == "shots2"`` usa el modelo de dos latentes (``model.filter_shots2``): estado
    (a, b, f, g, c, k, φ), tasas e^{c + k + a_i − b_j} σ(φ + f_i − g_j) y goles NegBin(·, r).

    ``boost=(t, θ)`` multiplica por e^{θ} la tasa de goles del equipo t y por e^{-θ} la de sus rivales
    en sus partidos pendientes (para muestrear temporadas en que suma muchos o pocos puntos); con
    ``keep_rates`` se guardan las tasas sin inclinar de cada partido en ``rates``, de modo que la
    razón de verosimilitudes de cualquier inclinación se calcula después (ver
    ``analysis.need_curves``).

    ``probs_k > 0`` devuelve además, por simulación, P(L, E, V) de los primeros ``probs_k``
    partidos dadas las fuerzas sorteadas: con eso el laboratorio de la web condiciona en
    resultados reponderando todas las temporadas (ver ``export``)."""
    rng = np.random.default_rng(seed)
    n = len(st.teams)
    mi = st.model_index
    kind = params.get("kind", "dc")
    if kind == "shots2":
        idx = np.r_[mi, N_model + mi, 2 * N_model + mi, 3 * N_model + mi, 4 * N_model, 4 * N_model + 1, 4 * N_model + 2]
    else:
        idx = np.r_[mi, N_model + mi, 2 * N_model, 2 * N_model + 1]  # (alpha, delta, mu, eta) locales
    mu_x, P_x = m[idx], P[np.ix_(idx, idx)]
    S = n_sims
    todo = st.todo
    M = len(todo)
    hi, ai = todo["hi"].to_numpy(), todo["ai"].to_numpy()
    neu = todo["neutral"].to_numpy(float)

    # ---- estado latente en la fecha de cada partido pendiente -----------------------
    # Se procesa por bloques de simulaciones para no materializar un tensor (S, M, D).
    if x_draw is not None:
        X0 = x_draw
    elif mode == "posterior":
        L = np.linalg.cholesky(P_x + 1e-10 * np.eye(len(idx)))
        X0 = mu_x + rng.standard_normal((S, len(idx))) @ L.T
    else:
        X0 = np.broadcast_to(mu_x, (S, len(idx))).copy()
    X_end = X0.copy()
    logw = np.zeros(S)
    gx = np.zeros((S, M), np.int64)
    gy = np.zeros((S, M), np.int64)
    K = min(probs_k, M)
    probs = np.zeros((S, K, 3)) if K else None
    rates = (np.zeros((S, M)), np.zeros((S, M))) if keep_rates else None
    if M:
        days = todo["days_ahead"].to_numpy()
        udays, inv = np.unique(days, return_inverse=True)
        dd = np.diff(udays, prepend=0.0)
        if kind == "shots2":
            q = np.r_[np.full(2 * n, params["qs"] ** 2), np.full(2 * n, params["qf"] ** 2), np.full(3, params["qlev"] ** 2)]
        else:
            q = np.r_[np.full(2 * n, params["s_team"] ** 2), params["s_mu"] ** 2, params["s_eta"] ** 2]
        for a in range(0, S, chunk):
            b = min(S, a + chunk)
            if mode == "posterior":
                inc = rng.standard_normal((b - a, len(udays), len(idx))) * np.sqrt(dd[None, :, None] * q[None, None, :])
                walk = np.cumsum(inc, axis=1)                     # (s, U, D)
                X_end[a:b] = X0[a:b] + walk[:, -1, :]
                W = walk[:, inv, :]                                # (s, M, D) solo para este bloque
                del walk, inc
            else:
                W = None
            lam, nu = match_rates(X0[a:b], W, hi, ai, neu, n, kind)
            del W
            if K:
                probs[a:b] = outcome_probs_given(lam[:, :K], nu[:, :K], params, kind)
            if keep_rates:
                rates[0][a:b], rates[1][a:b] = lam, nu
            if kind == "shots2":
                r_ = params["r"]
                gx[a:b] = rng.negative_binomial(r_, r_ / (r_ + lam))
                gy[a:b] = rng.negative_binomial(r_, r_ / (r_ + nu))
                if boost is not None:
                    _boost_outcomes(gx, gy, a, b, lam, nu, hi, ai, boost, params, kind, rng)
            elif tilt is None:
                gx[a:b], gy[a:b] = sample_dc(lam, nu, params["rho"], rng)
                if boost is not None:
                    _boost_outcomes(gx, gy, a, b, lam, nu, hi, ai, boost, params, kind, rng)
            else:
                t, th = tilt
                sh = np.where(hi == t, -th, np.where(ai == t, th, 0.0))[None, :]   # local
                lt, nt = lam * np.exp(sh), nu * np.exp(-sh)
                x, y = sample_dc(lt, nt, params["rho"], rng)
                gx[a:b], gy[a:b] = x, y
                lw = (x * np.log(lam / lt) - (lam - lt) + y * np.log(nu / nt) - (nu - nt)
                      + np.log(_tau(x, y, lam, nu, params["rho"])) - np.log(_tau(x, y, lt, nt, params["rho"])))
                logw[a:b] = lw.sum(1)

    # ---- tabla ------------------------------------------------------------------------
    def zeros(*shape):
        return np.zeros(shape)

    pts, gd, gf, w, agf = zeros(S, n), zeros(S, n), zeros(S, n), zeros(S, n), zeros(S, n)
    if st.adj is not None:
        pts += st.adj[None, :]
    hp, hg, hf, ha = zeros(S, n, n), zeros(S, n, n), zeros(S, n, n), zeros(S, n, n)

    def add(h, a, x, y, sl=slice(None)):
        # x, y: (S,) o escalares; h, a escalares
        ph = 3.0 * (x > y) + 1.0 * (x == y)
        pa = 3.0 * (y > x) + 1.0 * (x == y)
        pts[:, h] += ph; pts[:, a] += pa
        gd[:, h] += x - y; gd[:, a] += y - x
        gf[:, h] += x; gf[:, a] += y
        w[:, h] += x > y; w[:, a] += y > x
        agf[:, a] += y
        hp[:, h, a] += ph; hp[:, a, h] += pa
        hg[:, h, a] += x - y; hg[:, a, h] += y - x
        hf[:, h, a] += x; hf[:, a, h] += y
        ha[:, a, h] += y

    for h, a, x, y in st.done[["hi", "ai", "hg", "ag"]].itertuples(index=False):
        add(int(h), int(a), float(x), float(y))
    for k in range(M):
        add(int(hi[k]), int(ai[k]), gx[:, k].astype(float), gy[:, k].astype(float))

    crit = np.array([CRIT_CODE[c] for c in league.tiebreak], np.int64)
    rnd = rng.random((S, n))
    pos = tiebreak.rank_all(pts, gd, gf, w, agf, hp, hg, hf, ha, rnd, crit, league.h2h_reapply)

    # ---- partidos de definición (título y, en Serie A, descenso) ----------------------
    if league.title_playoff or league.relegation_playoff_pos:
        pos = _playoffs(league, pos, pts, X_end, n, params, rng)

    # ---- playoffs de ascenso (Championship) -------------------------------------------
    promoted = po_winner = None
    fmt = league.playoff_format(st.syear) if (league.promotion_playoff and st.syear is not None) else None
    if fmt:
        po_winner = promotion_playoff(fmt, pos, X_end, n, params, rng)
        promoted = pos < 2
        promoted[np.arange(S), po_winner] = True

    extra = None
    if st.split is not None:
        from .split import split_outcomes
        extra = split_outcomes(st, gx, gy, pos, pts, X_end, n, params, rng)

    return SimResult(st.teams, pos, pts, (gx, gy), todo, logw if tilt is not None else None, gd, probs,
                     promoted, po_winner, extra, rates, X0 if keep_rates else None)


# ----------------------------------------------------------------------------------------
# Llaves de ascenso, vectorizadas sobre las simulaciones
# ----------------------------------------------------------------------------------------
def _pair_rates(X, h, a, neutral, n, kind):
    """Tasas (local h, visita a) con equipos distintos en cada simulación: h, a de forma (S,)."""
    g = lambda c: np.take_along_axis(X, c[:, None], 1)[:, 0]  # noqa: E731
    nf = 0.0 if neutral else 1.0
    if kind == "shots2":
        sh = X[:, 4 * n] + X[:, 4 * n + 1] * nf + g(h) - g(n + a)
        sa = X[:, 4 * n] + g(a) - g(n + h)
        fh = X[:, 4 * n + 2] + g(2 * n + h) - g(3 * n + a)
        fa = X[:, 4 * n + 2] + g(2 * n + a) - g(3 * n + h)
        return np.exp(sh) / (1 + np.exp(-fh)), np.exp(sa) / (1 + np.exp(-fa))
    lam = np.exp(X[:, 2 * n] + X[:, 2 * n + 1] * nf + g(h) - g(n + a))
    nu = np.exp(X[:, 2 * n] + g(a) - g(n + h))
    return lam, nu


def _goals(lam, nu, params, rng):
    if params.get("kind", "dc") == "shots2":
        r = params["r"]
        return rng.negative_binomial(r, r / (r + lam)), rng.negative_binomial(r, r / (r + nu))
    return sample_dc(lam, nu, params["rho"], rng)


def knockout(X, a, b, n, params, rng, legs=1, neutral=False):
    """Llave entre a (mejor ubicado) y b, vectorizada: devuelve el ganador (S,). A partido único se
    juega en casa de a (o en cancha neutral); a ida y vuelta, a cierra de local. Si queda empatada,
    alargue (un tercio de partido en la cancha del último) y penales (moneda)."""
    kind = params.get("kind", "dc")
    ga = np.zeros(len(a)); gb = np.zeros(len(a))
    for leg in range(legs):
        home, away = (a, b) if (legs == 1 or leg == 1) else (b, a)
        lam, nu = _pair_rates(X, home, away, neutral, n, kind)
        x, y = _goals(lam, nu, params, rng)
        if legs == 1 or leg == 1:
            ga += x; gb += y
        else:
            gb += x; ga += y
    tie = ga == gb
    if tie.any():
        lam, nu = _pair_rates(X, a, b, neutral, n, kind)
        x, y = _goals(lam / 3, nu / 3, params, rng)
        ga = ga + np.where(tie, x, 0); gb = gb + np.where(tie, y, 0)
        tie = ga == gb
    coin = rng.random(len(a)) < 0.5
    return np.where(ga > gb, a, np.where(gb > ga, b, np.where(coin, a, b)))


def promotion_playoff(fmt, pos, X, n, params, rng):
    """Ganador de los playoffs de ascenso (índice local, por simulación).

    ``efl4`` (hasta 2025-26): 3.º-6.º y 4.º-5.º a ida y vuelta, final única neutral.
    ``efl6`` (desde 2026-27): 5.º-8.º y 6.º-7.º a partido único en casa del mejor ubicado; el 3.º
    enfrenta al peor ubicado de los dos ganadores y el 4.º al otro, a ida y vuelta; final neutral."""
    order = np.argsort(pos, axis=1)          # order[s, k] = equipo en la posición k (0-indexada)
    at = lambda k: order[:, k]               # noqa: E731
    if fmt == "efl4":
        s1 = knockout(X, at(2), at(5), n, params, rng, legs=2)
        s2 = knockout(X, at(3), at(4), n, params, rng, legs=2)
    elif fmt == "efl6":
        e1 = knockout(X, at(4), at(7), n, params, rng, legs=1)
        e2 = knockout(X, at(5), at(6), n, params, rng, legs=1)
        S_ = np.arange(len(e1))
        p1, p2 = pos[S_, e1], pos[S_, e2]
        lo = np.where(p1 > p2, e1, e2); hi = np.where(p1 > p2, e2, e1)
        s1 = knockout(X, at(2), lo, n, params, rng, legs=2)
        s2 = knockout(X, at(3), hi, n, params, rng, legs=2)
    else:
        raise ValueError(fmt)
    S_ = np.arange(len(s1))
    first = np.where(pos[S_, s1] < pos[S_, s2], s1, s2); second = np.where(first == s1, s2, s1)
    return knockout(X, first, second, n, params, rng, legs=1, neutral=True)


def _boost_outcomes(gx, gy, a, b, lam, nu, hi, ai, boost, params, kind, rng):
    """Vuelve a sortear los partidos del equipo t con el resultado inclinado: P_θ(o) ∝ P(o) e^{θ·puntos
    de t con o}, y los goles del modelo condicionados a ese resultado (por rechazo). Dentro de un mismo
    total de puntos, la razón de verosimilitudes solo depende de las fuerzas sorteadas, no de los
    goles: e^{θ g} / Π_k Z_k(θ) (ver ``analysis.need_curves``)."""
    t, th = boost
    ks = np.where((hi == t) | (ai == t))[0]
    if not len(ks):
        return
    L, N_ = lam[:, ks], nu[:, ks]
    p = outcome_probs_given(L, N_, params, kind)                       # (s, K, 3) local/empate/visita
    home = (hi[ks] == t)[None, :, None]
    pts = np.where(home, np.array([3.0, 1.0, 0.0]), np.array([0.0, 1.0, 3.0]))
    q = p * np.exp(th * pts); q /= q.sum(-1, keepdims=True)
    u = rng.random(q.shape[:2])[..., None]
    o = (u > np.cumsum(q, -1)).sum(-1)                                  # resultado inclinado
    x = np.zeros(L.shape, np.int64); y = np.zeros(L.shape, np.int64)
    todo = np.ones(L.shape, bool)
    for _ in range(500):
        if not todo.any():
            break
        if kind == "shots2":
            r_ = params["r"]
            xx = rng.negative_binomial(r_, r_ / (r_ + L)); yy = rng.negative_binomial(r_, r_ / (r_ + N_))
        else:
            xx, yy = sample_dc(L, N_, params["rho"], rng)
        oo = np.where(xx > yy, 0, np.where(xx == yy, 1, 2))
        hit = todo & (oo == o)
        x[hit], y[hit] = xx[hit], yy[hit]
        todo &= ~hit
    if todo.any():   # resultado muy improbable: marcador mínimo con ese resultado
        x[todo] = np.where(o[todo] == 0, 1, 0); y[todo] = np.where(o[todo] == 2, 1, 0)
    gx[a:b, ks], gy[a:b, ks] = x, y


def match_rates(X0, W, hi, ai, neu, n, kind="dc"):
    """Tasas de goles (local, visita) de cada partido pendiente, (S, M), con el estado local
    X0 (S, D) más el paseo aleatorio W (S, M, D) hasta la fecha del partido (None: sin paseo)."""
    M = len(hi); ar = np.arange(M)

    def col(c):          # columna c (vector por partido o escalar) en la fecha de cada partido
        if np.ndim(c) == 0:
            base = X0[:, [c]]
            return base + W[:, :, c] if W is not None else np.broadcast_to(base, (X0.shape[0], M))
        return X0[:, c] + W[:, ar, c] if W is not None else X0[:, c]

    if kind == "shots2":
        sh = col(4 * n) + col(4 * n + 1) * (1 - neu)[None, :] + col(hi) - col(n + ai)
        sa = col(4 * n) + col(ai) - col(n + hi)
        fh = col(4 * n + 2) + col(2 * n + hi) - col(3 * n + ai)
        fa = col(4 * n + 2) + col(2 * n + ai) - col(3 * n + hi)
        return np.exp(sh) / (1 + np.exp(-fh)), np.exp(sa) / (1 + np.exp(-fa))
    lam = np.exp(col(2 * n) + col(2 * n + 1) * (1 - neu)[None, :] + col(hi) - col(n + ai))
    nu = np.exp(col(2 * n) + col(ai) - col(n + hi))
    return lam, nu


def outcome_probs_given(lam, nu, params, kind="dc", maxg=12):
    """P(local gana, empate, visita gana) dadas las tasas (Dixon–Coles o NegBin independientes)."""
    g = np.arange(maxg + 1)
    if kind == "shots2":
        from scipy.stats import nbinom
        r = params["r"]
        ph = nbinom.pmf(g, r, (r / (r + lam))[..., None])
        pa = nbinom.pmf(g, r, (r / (r + nu))[..., None])
        J = ph[..., :, None] * pa[..., None, :]
    else:
        from scipy.stats import poisson
        rho = params["rho"]
        ph = poisson.pmf(g, lam[..., None]); pa = poisson.pmf(g, nu[..., None])
        J = ph[..., :, None] * pa[..., None, :]
        J[..., 0, 0] *= 1 - lam * nu * rho; J[..., 0, 1] *= 1 + lam * rho
        J[..., 1, 0] *= 1 + nu * rho; J[..., 1, 1] *= 1 - rho
    out = np.stack([np.tril(J, -1).sum((-1, -2)), np.trace(J, axis1=-2, axis2=-1), np.triu(J, 1).sum((-1, -2))], -1)
    return out / out.sum(-1, keepdims=True)


def _decider(X, s, a, b, n, params, rng, legs=1, home_first=True):
    """Partido(s) de definición entre a (mejor ubicado) y b en la simulación s. Devuelve el ganador."""
    kind = params.get("kind", "dc")
    ga = gb = 0
    for leg in range(legs):
        home = a if (legs == 1 or leg == 1) else b  # en ida y vuelta, el mejor ubicado cierra de local
        away = b if home == a else a
        lam, nu = match_rates(X[s:s + 1], None, np.array([home]), np.array([away]), np.zeros(1), n, kind)
        if kind == "shots2":
            r_ = params["r"]
            x, y = rng.negative_binomial(r_, r_ / (r_ + lam[0])), rng.negative_binomial(r_, r_ / (r_ + nu[0]))
        else:
            x, y = sample_dc(lam[0], nu[0], params["rho"], rng)
        if home == a:
            ga += x[0]; gb += y[0]
        else:
            gb += x[0]; ga += y[0]
    if ga != gb:
        return a if ga > gb else b
    return a if rng.random() < 0.5 else b  # penales


def _playoffs(league, pos, pts, X, n, params, rng):
    pos = pos.copy()
    S = pos.shape[0]
    for s in range(S):
        order = np.argsort(pos[s])
        if league.title_playoff:
            p0, p1 = pts[s, order[0]], pts[s, order[1]]
            p2 = pts[s, order[2]] if n > 2 else -1
            if p0 == p1 and p1 != p2:
                win = _decider(X, s, order[0], order[1], n, params, rng, legs=1)
                if win == order[1]:
                    pos[s, order[0]], pos[s, order[1]] = 1, 0
        r = league.relegation_playoff_pos
        if r:
            a, b = order[r - 1], order[r]       # último a salvo y primero en descender
            pa_, pb_ = pts[s, a], pts[s, b]
            above = pts[s, order[r - 2]] if r >= 2 else np.inf
            below = pts[s, order[r + 1]] if r + 1 < n else -np.inf
            if pa_ == pb_ and above != pa_ and below != pa_:
                win = _decider(X, s, a, b, n, params, rng, legs=2)
                if win == b:
                    pos[s, a], pos[s, b] = r, r - 1
    return pos
