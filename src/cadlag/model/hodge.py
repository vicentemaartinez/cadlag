r"""Descomposición de Hodge (HodgeRank) de los flujos de goles.

Sobre el grafo G = (V, E) de equipos (aristas = pares que se enfrentaron), un *flujo* es una
función antisimétrica Y en las aristas: Y_ij = diferencia de gol media de i contra j, ya
descontada la localía. Con pesos w_ij (número de partidos), el espacio de flujos se descompone
ortogonalmente (Jiang, Lim, Yao & Ye, 2011):

    Y = grad s  ⊕  curl* Φ  ⊕  h
        (transitivo)   (cíclico local, triángulos)   (armónico, ciclos globales)

  * grad s : (grad s)_ij = s_i - s_j. s resuelve el sistema laplaciano L_w s = div Y; es el
             "rating" de mínimos cuadrados (Massey).
  * curl   : la parte que se explica por ciclos en triángulos i -> j -> k -> i (piedra-papel-tijera).
  * h      : lo que no es ni gradiente ni ciclo local; vive en H^1 del complejo de cliques y es
             distinto de cero solo si el grafo tiene "agujeros" (por ejemplo, ligas unidas por
             pocos partidos internacionales).

En una liga con todos contra todos el grafo es completo y h = 0: toda la intransitividad es
local. La proporción ||curl||^2 / ||Y||^2 mide cuánta estructura cíclica tienen los resultados.

Nulo calibrado. Incluso un modelo puramente "de fuerzas" como Poisson log-lineal genera algo
de flujo cíclico: (i) por el ruido de muestreo de los goles y (ii) porque E[GD_ij] =
e^{mu+a_i-d_j} - e^{mu+a_j-d_i} no es exactamente un gradiente (la exponencial no es lineal).
Por eso la proporción observada se compara con su distribución bajo el modelo ajustado.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import lsqr


def build_flows(home: np.ndarray, away: np.ndarray, gd: np.ndarray, home_adv: float = 0.0):
    """Agrega partidos en aristas no dirigidas i<j con flujo Y_ij = media(GD de i contra j)."""
    acc = {}
    for h, a, g in zip(home, away, gd):
        g = g - home_adv
        i, j, s = (h, a, 1.0) if h < a else (a, h, -1.0)
        y, w = acc.get((i, j), (0.0, 0))
        acc[(i, j)] = (y + s * g, w + 1)
    edges = np.array(list(acc.keys()), int)
    Y = np.array([v[0] / v[1] for v in acc.values()])
    W = np.array([v[1] for v in acc.values()], float)
    return edges, Y, W


def decompose(n: int, edges: np.ndarray, Y: np.ndarray, W: np.ndarray, max_triangles: int = 200000):
    m = len(edges)
    # operador gradiente: (grad s)_e = s_i - s_j
    rows = np.r_[np.arange(m), np.arange(m)]
    cols = np.r_[edges[:, 0], edges[:, 1]]
    vals = np.r_[np.ones(m), -np.ones(m)]
    G = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(m, n)))
    sw = np.sqrt(W)
    Gw = diags(sw) @ G
    s = lsqr(Gw, sw * Y, atol=1e-12, btol=1e-12)[0]
    grad = G @ s
    R = Y - grad
    # triángulos del complejo de cliques
    eix = {(int(i), int(j)): k for k, (i, j) in enumerate(edges)}
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    tri = []
    for i in range(n):
        for j, k in combinations(sorted(x for x in adj[i] if x > i), 2):
            if k in adj[j]:
                tri.append((i, j, k))
                if len(tri) >= max_triangles:
                    break
    curl = np.zeros(m)
    if tri:
        # curl: flujo en triángulo (i,j,k) = Y_ij + Y_jk + Y_ki ; su adjunto lleva triángulos a aristas
        T = len(tri)
        r_, c_, v_ = [], [], []
        for t, (i, j, k) in enumerate(tri):
            for (a, b) in ((i, j), (j, k), (k, i)):
                if (a, b) in eix:
                    r_.append(t); c_.append(eix[(a, b)]); v_.append(1.0)
                else:
                    r_.append(t); c_.append(eix[(b, a)]); v_.append(-1.0)
        C = csr_matrix(coo_matrix((v_, (r_, c_)), shape=(T, m)))
        # proyección W-ortogonal de R sobre im(W^{-1} C^T):  min || sqrt(W) (R - W^{-1} C^T phi) ||
        A = diags(1 / sw) @ C.T
        phi = lsqr(A, sw * R, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
        curl = (diags(1 / W) @ (C.T @ phi))
    harm = R - curl
    norm = lambda x: float((W * x ** 2).sum())
    tot = norm(Y)
    return dict(s=s, grad=grad, curl=curl, harmonic=harm,
                share_grad=norm(grad) / tot, share_curl=norm(curl) / tot, share_harmonic=norm(harm) / tot,
                n_triangles=len(tri))
