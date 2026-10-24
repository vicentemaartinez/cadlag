r"""Reglas de puntuación propias para pronósticos categóricos y ordinales.

Para un pronóstico p sobre K categorías ordenadas y el resultado observado o:

  log score   LS(p, o)  = -log p_o                                  (estrictamente propia, local)
  Brier       BS(p, o)  = sum_k (p_k - 1{k=o})^2                    (propia, cuadrática)
  RPS         RPS(p, o) = 1/(K-1) sum_{k<K} (F_k - 1{o<=k})^2        (propia, sensible a la distancia)

El RPS es la versión discreta del CRPS: (K-1) RPS = sum_k (F_k - H_k)^2, una distancia L2
entre funciones de distribución acumuladas. *No* debe confundirse con la distancia de
Wasserstein-1, sum_k |F_k - H_k|, que es impropia: se minimiza concentrando toda la masa
en la mediana del pronóstico (ver docs/teoria.md).
"""
from __future__ import annotations

import numpy as np


def log_score(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(o)), o], 1e-300, None))


def brier(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    e = np.zeros_like(p)
    e[np.arange(len(o)), o] = 1.0
    return ((p - e) ** 2).sum(-1)


def rps(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    K = p.shape[-1]
    F = np.cumsum(p, -1)[..., :-1]
    H = (np.arange(K - 1)[None, :] >= o[:, None]).astype(float)
    return ((F - H) ** 2).sum(-1) / (K - 1)


def summarize(p: np.ndarray, o: np.ndarray) -> dict:
    return dict(n=len(o), log=float(log_score(p, o).mean()), brier=float(brier(p, o).mean()),
                rps=float(rps(p, o).mean()))


def reliability(prob: np.ndarray, hit: np.ndarray, n_bins: int = 10) -> dict:
    """Diagrama de confiabilidad por cuantiles (bins de igual tamaño).

    Devuelve la probabilidad media pronosticada y la frecuencia observada en cada bin,
    con un intervalo binomial (Wilson) de 90 %.
    """
    order = np.argsort(prob)
    bins = np.array_split(order, n_bins)
    out = []
    for b in bins:
        if len(b) == 0:
            continue
        n = len(b)
        f = hit[b].mean()
        z = 1.645
        den = 1 + z ** 2 / n
        c = (f + z ** 2 / (2 * n)) / den
        h = z * np.sqrt(f * (1 - f) / n + z ** 2 / (4 * n ** 2)) / den
        out.append(dict(p=float(prob[b].mean()), freq=float(f), lo=float(c - h), hi=float(c + h), n=n))
    return dict(bins=out)


def murphy_decomposition(prob: np.ndarray, hit: np.ndarray, n_bins: int = 10) -> dict:
    """Brier = confiabilidad - resolución + incertidumbre (Murphy 1973), con bins por cuantil."""
    order = np.argsort(prob)
    obar = hit.mean()
    rel = res = 0.0
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        w = len(b) / len(prob)
        pk, ok = prob[b].mean(), hit[b].mean()
        rel += w * (pk - ok) ** 2
        res += w * (ok - obar) ** 2
    return dict(reliability=rel, resolution=res, uncertainty=obar * (1 - obar),
                brier=float(((prob - hit) ** 2).mean()))
