r"""Eventos raros: muestreo por importancia (pendiente 4).

Monte Carlo directo no puede estimar probabilidades como "el Manchester City desciende": con
20.000 temporadas el estimador vale 0. Su error relativo es \sqrt{(1-p)/(np)}: para p = 10^{-6} y
un 10 % de error harían falta 10^8 temporadas.

**Método (``importance_results``).** Un evento extremo de un equipo ocurre por dos caminos que se
combinan: que el equipo *sea* peor de lo que se cree (fuerza) y que le vaya mal (resultados). La
propuesta inclina ambos:

  * el bloque w = (α_t, δ_t) se sortea de N(μ_w − c u, P_ww), con u la dirección que baja la fuerza
    neta en una desviación a priori, y el resto del estado de su condicional exacta dado w;
  * en cada partido del equipo, su tasa de goles se multiplica por e^{-θ} y la del rival por e^{θ}.

La razón de verosimilitudes es exacta y se factoriza:

    W = \frac{N(w; μ_w, P_ww)}{N(w; μ_w − c u, P_ww)} \prod_{\text{partidos de } t}
        \frac{\mathrm{DC}(x, y; λ, ν)}{\mathrm{DC}(x, y; λ e^{-θ}, ν e^{θ})},
    \hat p = \frac1n \sum_k W_k 1\{A_k\},   E_q[\hat p] = p.

(c, θ) se eligen en corridas piloto minimizando la varianza relativa estimada
E_q[W² 1_A] / E_q[W 1_A]² − 1.

**Lo que no funcionó (``importance``, entropía cruzada sobre la fuerza sola).** Inclinar solo el
estado latente, aun con entropía cruzada, mezclas y covarianza adaptada, da estimadores de cola
pesada (tamaño efectivo 1–7 y semillas que discrepan en órdenes de magnitud) para eventos de 10^{-7}:
el camino que domina mezcla fuerza y mala suerte, y la mala suerte sin inclinar casi nunca se
muestrea. Se deja como referencia.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .season import simulate


@dataclass
class RareResult:
    p: float                 # estimación
    rel_err: float           # error relativo (desviación estándar / p)
    n: int                   # temporadas de la corrida final
    ess: float               # tamaño efectivo de muestra entre las que cumplen el evento
    levels: list             # umbrales por iteración de entropía cruzada
    shift_sd: float          # cuántas desviaciones a priori se movió el bloque (Mahalanobis)


def _local(st, m, P, N_model):
    n = len(st.teams); mi = st.model_index
    idx = np.r_[mi, N_model + mi, 2 * N_model, 2 * N_model + 1]
    return n, m[idx], P[np.ix_(idx, idx)] + 1e-10 * np.eye(len(idx))


def margin_below(t: int, k: int):
    """Puntaje para "t termina en la posición k (0-based) o peor": puntos del k-ésimo mejor de los
    demás menos los de t. Vale ≥ 0 (aprox.) cuando ocurre el evento."""
    def f(r):
        others = np.delete(r.pts, t, axis=1)
        return -np.sort(-others, axis=1)[:, k - 1] - r.pts[:, t]
    return f


def margin_above(t: int, k: int = 0):
    """Para "t termina en la posición k o mejor": puntos de t menos los del (k+1)-ésimo mejor de los demás."""
    def f(r):
        others = np.delete(r.pts, t, axis=1)
        return r.pts[:, t] + np.sort(-others, axis=1)[:, k]
    return f


def _logpdf(x, mean, C):
    L = np.linalg.cholesky(C)
    u = np.linalg.solve(L, (x - mean).T).T
    return -0.5 * (u ** 2).sum(1) - np.log(np.diag(L)).sum() - 0.5 * len(mean) * np.log(2 * np.pi)


def importance(league, st, m, P, params, N_model, score, event, team: int, n_sims=4000, n_final=20000,
               rho=0.1, max_iter=40, smooth=0.9, refine=2, infl=2.0, seed=0) -> RareResult:
    """P(event), inclinando el bloque (ataque, defensa) del equipo local ``team``."""
    rng = np.random.default_rng(seed)
    n, mu, Px = _local(st, m, P, N_model)
    D = len(mu)
    b = np.array([team, n + team]); o = np.setdiff1d(np.arange(D), b)
    mu_w, Pww = mu[b], Px[np.ix_(b, b)]
    G = Px[np.ix_(o, b)] @ np.linalg.inv(Pww)                     # regresión de o sobre w
    Lo = np.linalg.cholesky(Px[np.ix_(o, o)] - G @ Px[np.ix_(b, o)] + 1e-10 * np.eye(len(o)))
    mw, Cw = mu_w.copy(), Pww.copy()
    levels = []

    path = []  # todas las propuestas del camino de entropía cruzada

    def sim_w(w, s):
        S = len(w)
        X = np.empty((S, D))
        X[:, b] = w
        X[:, o] = mu[o] + (w - mu_w) @ G.T + rng.standard_normal((S, len(o))) @ Lo.T
        return simulate(league, st, m, P, params, N_model, n_sims=S, seed=s, x_draw=X)

    def run(S, mw, Cw, s):
        w = mw + rng.standard_normal((S, 2)) @ np.linalg.cholesky(Cw).T
        path.append((mw.copy(), Cw.copy()))
        return w, sim_w(w, s), _logpdf(w, mu_w, Pww) - _logpdf(w, mw, Cw)

    for it in range(max_iter):
        w, r, logw = run(n_sims, mw, Cw, seed + 1 + it)
        sc = score(r)
        gamma = min(0.0, float(np.quantile(sc, 1 - rho)))
        levels.append(gamma)
        we = w[sc >= gamma]
        mw = smooth * we.mean(0) + (1 - smooth) * mw
        Cw = smooth * (np.cov(we, rowvar=False) + 1e-6 * np.eye(2)) + (1 - smooth) * Cw
        if gamma >= 0:
            break
    # refinamiento con pesos: ya en el umbral, la media y la covarianza ponderadas por W estiman
    # las de w dado A bajo el modelo (las de la élite sin pesos se pasan de largo)
    for it in range(refine):
        w, r, logw = run(n_sims, mw, Cw, seed + 500 + it)
        hit = event(r)
        if hit.sum() < 20:
            break
        q = np.exp(logw[hit] - logw[hit].max()); q /= q.sum()
        mw = q @ w[hit]
        dw = w[hit] - mw
        Cw = infl * (dw.T * q) @ dw + 1e-6 * np.eye(2)   # más ancha: colas más seguras
    # corrida final: mezcla determinística de todas las propuestas del camino (heurística de
    # balance de Veach–Guibas). El camino cubre la zona intermedia entre el modelo y el evento,
    # que con una sola gaussiana al final queda sin muestrear y produce pesos gigantes
    path.append((mw.copy(), Cw.copy()))
    K = len(path)
    comp = np.repeat(np.arange(K), int(np.ceil(n_final / K)))[:n_final]
    w = np.empty((n_final, 2))
    for k in range(K):
        sel = comp == k
        w[sel] = path[k][0] + rng.standard_normal((sel.sum(), 2)) @ np.linalg.cholesky(path[k][1]).T
    r = sim_w(w, seed + 1000)
    lq = np.logaddexp.reduce(np.stack([_logpdf(w, mk, Ck) for mk, Ck in path]), axis=0) - np.log(K)
    logw = _logpdf(w, mu_w, Pww) - lq
    hit = event(r)
    wv = np.where(hit, np.exp(logw), 0.0)
    p = wv.mean()
    se = wv.std(ddof=1) / np.sqrt(n_final)
    wh = wv[hit]
    ess = (wh.sum() ** 2 / (wh ** 2).sum()) if hit.any() else 0.0
    dm = mw - mu_w
    return RareResult(p=float(p), rel_err=float(se / p) if p > 0 else float("inf"), n=n_final, ess=float(ess),
                      levels=levels, shift_sd=float(np.sqrt(dm @ np.linalg.solve(Pww, dm))))


def importance_results(league, st, m, P, params, N_model, event, team: int, sign: float = 1.0,
                       thetas=(0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0), shifts=(0.0, 1.0, 2.0, 3.0, 4.0),
                       pilot=2000, n_final=40000, seed=0) -> RareResult:
    """P(event) inclinando a la vez la **fuerza** y los **resultados** del equipo.

    * Fuerza: el bloque w = (α_t, δ_t) se sortea de N(μ_w − s·c·u, P_ww), con u la dirección que baja
      la fuerza neta α+δ en c desviaciones a priori (el resto del estado, de su condicional exacta).
    * Resultados: en cada partido del equipo su tasa de goles se multiplica por e^{-sθ} y la del
      rival por e^{+sθ}.
    s = +1 empeora al equipo y s = −1 lo mejora. La razón de verosimilitudes es el producto de la
    gaussiana del bloque y de los cocientes de Dixon–Coles partido a partido, ambas exactas. (c, θ)
    se eligen en corridas piloto minimizando la varianza relativa estimada E_q[W²1_A]/E_q[W1_A]² − 1.
    """
    rng = np.random.default_rng(seed)
    n, mu, Px = _local(st, m, P, N_model)
    D = len(mu)
    b = np.array([team, n + team]); o = np.setdiff1d(np.arange(D), b)
    mu_w, Pww = mu[b], Px[np.ix_(b, b)]
    G = Px[np.ix_(o, b)] @ np.linalg.inv(Pww)
    Lo = np.linalg.cholesky(Px[np.ix_(o, o)] - G @ Px[np.ix_(b, o)] + 1e-10 * np.eye(len(o)))
    one = np.ones(2); u = Pww @ one / np.sqrt(one @ Pww @ one)   # desplazamiento de 1 desv. en α+δ
    Lw = np.linalg.cholesky(Pww)

    def run(S, c, th, sd):
        mw = mu_w - sign * c * u
        w = mw + rng.standard_normal((S, 2)) @ Lw.T
        X = np.empty((S, D)); X[:, b] = w
        X[:, o] = mu[o] + (w - mu_w) @ G.T + rng.standard_normal((S, len(o))) @ Lo.T
        r = simulate(league, st, m, P, params, N_model, n_sims=S, seed=sd, x_draw=X, tilt=(team, sign * th))
        return r, r.logw + _logpdf(w, mu_w, Pww) - _logpdf(w, mw, Pww)

    best, k = None, 0
    for c in shifts:
        for th in thetas:
            k += 1
            r, lw = run(pilot, c, th, seed + k)
            hit = event(r)
            if hit.sum() < 5:
                continue
            w = np.exp(lw[hit]); p = w.sum() / pilot
            rv = (w ** 2).sum() / pilot / p ** 2 - 1
            if best is None or rv < best[2]:
                best = (c, th, rv)
    c, th = (best[0], best[1]) if best else (shifts[-1], thetas[-1])
    r, lw = run(n_final, c, th, seed + 1000)
    hit = event(r)
    wv = np.where(hit, np.exp(lw), 0.0)
    p = wv.mean(); se = wv.std(ddof=1) / np.sqrt(n_final)
    wh = wv[hit]; ess = (wh.sum() ** 2 / (wh ** 2).sum()) if hit.any() else 0.0
    return RareResult(p=float(p), rel_err=float(se / p) if p > 0 else float("inf"), n=n_final, ess=float(ess),
                      levels=[float(c), float(th)], shift_sd=float(c))


def naive(league, st, m, P, params, N_model, event, n_sims=200000, seed=0, chunk=20000):
    hits = 0
    for k in range(0, n_sims, chunk):
        r = simulate(league, st, m, P, params, N_model, n_sims=min(chunk, n_sims - k), seed=seed + k)
        hits += int(event(r).sum())
    p = hits / n_sims
    return p, (np.sqrt(p * (1 - p) / n_sims) / p if p > 0 else float("inf"))
