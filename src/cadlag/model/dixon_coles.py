"""Verosimilitud de Dixon–Coles (1997).

    P(X=x, Y=y) = tau_rho(x, y; lam, mu) * Pois(x; lam) * Pois(y; mu)

    tau(0,0) = 1 - lam*mu*rho      tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho          tau(1,1) = 1 - rho      tau = 1 en otro caso.

La corrección es una perturbación local que conserva la masa total (la suma de los cuatro
ajustes es exactamente cero), así que no hace falta renormalizar:

    lam*mu*rho*p00 ... se cancela: -lam mu rho e^{-lam-mu} + lam rho * mu e^{-lam-mu}
                                   + mu rho * lam e^{-lam-mu} - rho lam mu e^{-lam-mu} = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

MAXG = 10  # goles máximos por equipo en las grillas de probabilidad (P(X>10) < 1e-6)


def tau_np(x, y, lam, mu, rho):
    t = np.ones(np.broadcast(x, y, lam, mu).shape)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return np.maximum(t, 1e-10)


def score_grid(lam, mu, rho, maxg: int = MAXG) -> np.ndarray:
    """Matriz de probabilidades de marcador, forma (..., maxg+1, maxg+1)."""
    lam = np.asarray(lam, float)[..., None, None]
    mu = np.asarray(mu, float)[..., None, None]
    g = np.arange(maxg + 1)
    x = g[:, None]
    y = g[None, :]
    logp = (x * np.log(lam) - lam - gammaln(x + 1)) + (y * np.log(mu) - mu - gammaln(y + 1))
    p = np.exp(logp) * tau_np(x, y, lam, mu, rho)
    return p


def outcome_probs(grid: np.ndarray) -> np.ndarray:
    """(local, empate, visita) a partir de una grilla de marcadores."""
    home = np.tril(grid, -1).sum(axis=(-2, -1))
    draw = np.trace(grid, axis1=-2, axis2=-1)
    away = np.triu(grid, 1).sum(axis=(-2, -1))
    tot = home + draw + away
    return np.stack([home, draw, away], -1) / tot[..., None]


# Nodos de Gauss–Hermite (probabilistas) para integrar sobre la incertidumbre de las fuerzas.
def gh_nodes(n: int = 7):
    u, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    U1, U2 = np.meshgrid(u, u, indexing="ij")
    W = np.outer(w, w)
    return np.stack([U1.ravel(), U2.ravel()], -1), W.ravel()


def predictive_grid(zbar: np.ndarray, S: np.ndarray, rho: float, n_gh: int = 7,
                    maxg: int = MAXG, chunk: int = 2000) -> np.ndarray:
    """Predictiva posterior del marcador: E_z[ DC(x, y | e^{z}) ],  z ~ N(zbar, S).

    zbar: (M, 2) log-tasas medias (local, visita); S: (M, 2, 2) covarianzas.
    Con S = 0 se recupera la predicción *plug-in*.
    """
    nodes, w = gh_nodes(n_gh)
    out = np.empty((len(zbar), maxg + 1, maxg + 1))
    for a in range(0, len(zbar), chunk):
        zb = zbar[a:a + chunk]
        L = np.linalg.cholesky(S[a:a + chunk] + 1e-12 * np.eye(2))
        z = zb[:, None, :] + np.einsum("mij,kj->mki", L, nodes)  # (m, K, 2)
        g = score_grid(np.exp(z[..., 0]), np.exp(z[..., 1]), rho, maxg)  # (m, K, G, G)
        out[a:a + chunk] = np.einsum("mkxy,k->mxy", g, w)
    return out
