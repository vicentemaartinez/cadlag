"""Comparador recursivo de desempates (compilado con Numba).

Cada liga define una lista ordenada de criterios (``cadlag.leagues.CRITERIA``). El
algoritmo mantiene una pila de segmentos empatados:

    1. ordenar el segmento por el criterio k (los criterios H2H se calculan dentro del
       grupo empatado actual: la "minitabla");
    2. partir en sub-segmentos con clave igual;
    3. cada sub-segmento con más de un equipo sigue con el criterio k+1, salvo en ligas con
       ``h2h_reapply`` (LaLiga): si el criterio era H2H y el sub-grupo es más chico que el
       grupo original, la minitabla se recalcula desde el primer criterio H2H solo con él.
"""
from __future__ import annotations

import numpy as np
from numba import njit

PTS, GD, GF, W, AGF, H2H_PTS, H2H_GD, H2H_GF, H2H_AGF, LOTS = range(10)


@njit(cache=True)
def _key(c, i, grp, pts, gd, gf, w, agf, hp, hg, hf, ha, rnd):
    if c == PTS:
        return pts[i]
    if c == GD:
        return gd[i]
    if c == GF:
        return gf[i]
    if c == W:
        return w[i]
    if c == AGF:
        return agf[i]
    if c == LOTS:
        return rnd[i]
    s = 0.0
    for j in grp:
        if j == i:
            continue
        if c == H2H_PTS:
            s += hp[i, j]
        elif c == H2H_GD:
            s += hg[i, j]
        elif c == H2H_GF:
            s += hf[i, j]
        elif c == H2H_AGF:
            s += ha[i, j]
    return s


@njit(cache=True)
def rank_one(pts, gd, gf, w, agf, hp, hg, hf, ha, rnd, crit, reapply):
    """Devuelve ``order``: order[r] = equipo en la posición r (0 = primero)."""
    n = pts.shape[0]
    order = np.arange(n)
    nc = crit.shape[0]
    first_h2h = -1
    for k in range(nc):
        if crit[k] >= H2H_PTS and crit[k] <= H2H_AGF:
            first_h2h = k
            break
    st_a = np.empty(4 * n * (nc + 1), np.int64)
    st_b = np.empty_like(st_a)
    st_k = np.empty_like(st_a)
    top = 0
    st_a[0], st_b[0], st_k[0] = 0, n, 0
    top = 1
    keys = np.empty(n)
    while top > 0:
        top -= 1
        a, b, k = st_a[top], st_b[top], st_k[top]
        if b - a <= 1 or k >= nc:
            continue
        grp = order[a:b].copy()
        c = crit[k]
        for t in range(b - a):
            keys[t] = _key(c, grp[t], grp, pts, gd, gf, w, agf, hp, hg, hf, ha, rnd)
        idx = np.argsort(-keys[: b - a])
        sk = keys[: b - a][idx]
        order[a:b] = grp[idx]
        # partir en corridas de clave igual
        s = 0
        for t in range(1, b - a + 1):
            if t == b - a or sk[t] != sk[s]:
                if t - s > 1:
                    is_h2h = c >= H2H_PTS and c <= H2H_AGF
                    if reapply and is_h2h and (t - s) < (b - a):
                        nk = first_h2h
                    else:
                        nk = k + 1
                    st_a[top], st_b[top], st_k[top] = a + s, a + t, nk
                    top += 1
                s = t
    return order


@njit(cache=True)
def rank_all(pts, gd, gf, w, agf, hp, hg, hf, ha, rnd, crit, reapply):
    """Versión vectorizada sobre simulaciones: entradas con eje 0 = simulación.

    Devuelve ``pos`` (S, n): posición 0-indexada de cada equipo en cada simulación.
    """
    S, n = pts.shape
    pos = np.empty((S, n), np.int64)
    for s in range(S):
        order = rank_one(pts[s], gd[s], gf[s], w[s], agf[s], hp[s], hg[s], hf[s], ha[s], rnd[s], crit, reapply)
        for r in range(n):
            pos[s, order[r]] = r
    return pos
