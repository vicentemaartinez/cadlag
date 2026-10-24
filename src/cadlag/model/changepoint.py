r"""Extensión E7: detección bayesiana en línea de cambios de régimen (Adams & MacKay, 2007).

Sobre la serie de residuos de un equipo, e_k = (GF_k - GA_k) - E[GF_k - GA_k | pasado], que
el filtro debería dejar como ruido blanco, se supone que hay regímenes en los que la media de e
es constante. Con la longitud de corrida r_t (partidos desde el último cambio) y una tasa de
riesgo H(r) = 1/L, la recursión es

    P(r_t = 0, e_{1:t})     = sum_r P(r_{t-1} = r, e_{1:t-1}) * pi^{(r)}(e_t) * H
    P(r_t = r+1, e_{1:t})   = P(r_{t-1} = r, e_{1:t-1}) * pi^{(r)}(e_t) * (1 - H)

donde pi^{(r)} es la predictiva del régimen actual. Con prior Normal–Gamma sobre (media,
precisión), la predictiva es t de Student y las estadísticas suficientes se actualizan en O(1).

Interpretación: el paseo aleatorio del filtro supone cambios pequeños y continuos; un régimen
nuevo detectado con alta probabilidad señala un cambio abrupto (técnico, lesiones, mercado)
que el filtro tarda en absorber. Es la motivación para innovaciones de cola pesada en el estado.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def bocpd(x: np.ndarray, L: float = 60.0, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0, max_run: int = 300):
    T = len(x)
    H = 1.0 / L
    R = np.zeros((T + 1, min(T, max_run) + 1))
    R[0, 0] = 1.0
    mu = np.array([mu0]); kappa = np.array([kappa0]); alpha = np.array([alpha0]); beta = np.array([beta0])
    cp = np.zeros(T)
    mean_run = np.zeros(T)
    for t in range(T):
        # predictiva t de Student de cada longitud de corrida
        df = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        z = (x[t] - mu) ** 2 / scale2
        logpred = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df * scale2)
                   - (df + 1) / 2 * np.log1p(z / df))
        pred = np.exp(logpred)
        prev = R[t, :len(pred)]
        growth = prev * pred * (1 - H)
        change = (prev * pred * H).sum()
        new = np.r_[change, growth][: R.shape[1]]
        new /= new.sum()
        R[t + 1, :len(new)] = new
        cp[t] = new[0]
        mean_run[t] = (np.arange(len(new)) * new).sum()
        # actualizar estadísticas (la corrida 0 arranca del prior)
        mu_n = (kappa * mu + x[t]) / (kappa + 1)
        kappa_n = kappa + 1
        alpha_n = alpha + 0.5
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.r_[mu0, mu_n][: R.shape[1]]
        kappa = np.r_[kappa0, kappa_n][: R.shape[1]]
        alpha = np.r_[alpha0, alpha_n][: R.shape[1]]
        beta = np.r_[beta0, beta_n][: R.shape[1]]
    return dict(run_posterior=R[1:], cp_prob=cp, mean_run=mean_run)


def recent_change_prob(R: np.ndarray, window: int = 5) -> float:
    """P(el último cambio de régimen ocurrió en los últimos ``window`` partidos)."""
    return float(R[-1, :window].sum())
