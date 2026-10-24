r"""Probabilidades implícitas en las cuotas: métodos para quitar el margen ("de-vig").

Con cuotas decimales o_k, las probabilidades brutas pi_k = 1/o_k suman 1 + margen > 1.

  multiplicativo   p_k = pi_k / sum pi
  potencia         p_k = pi_k^{1/k*},   k* tal que sum p = 1
  Shin (1991/93)   p_k = ( sqrt(z^2 + 4 (1-z) pi_k^2 / Pi) - z ) / (2 (1-z)),   Pi = sum pi

En el modelo de Shin una fracción z del dinero proviene de apostadores con información
privilegiada; el corredor fija precios que lo protegen, lo que produce el sesgo
favorito–longshot. z se obtiene resolviendo sum_k p_k(z) = 1 (monótona en z).
Whelan (2025) argumenta que z mide sobre todo el margen y no "insiders" reales.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def multiplicative(odds: np.ndarray) -> np.ndarray:
    pi = 1.0 / odds
    return pi / pi.sum(-1, keepdims=True)


def power(odds: np.ndarray) -> np.ndarray:
    pi = 1.0 / np.atleast_2d(odds)
    out = np.empty_like(pi)
    for i, row in enumerate(pi):
        k = brentq(lambda k: (row ** (1 / k)).sum() - 1, 0.5, 1.0 + 1e-12) if row.sum() > 1 else 1.0
        out[i] = row ** (1 / k)
    return out


def shin_probs(pi: np.ndarray, z: float) -> np.ndarray:
    P = pi.sum()
    return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / P) - z) / (2 * (1 - z))


def shin(odds: np.ndarray, return_z: bool = False):
    pi = 1.0 / np.atleast_2d(odds)
    out = np.empty_like(pi)
    zs = np.zeros(len(pi))
    for i, row in enumerate(pi):
        if row.sum() <= 1:
            out[i], zs[i] = row / row.sum(), 0.0
            continue
        z = brentq(lambda z: shin_probs(row, z).sum() - 1, 0.0, 0.5)
        out[i], zs[i] = shin_probs(row, z), z
    return (out, zs) if return_z else out
