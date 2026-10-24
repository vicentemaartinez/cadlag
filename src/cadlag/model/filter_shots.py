r"""Extensión E4: tiros al arco como observación adicional del mismo estado latente.

El xG no está disponible gratis para todas las ligas (FBref lo perdió en 2026 y Chile nunca lo
tuvo), pero ESPN publica tiros al arco (SOT) de casi todos los partidos, Chile incluido. Se
modelan como una segunda medición ruidosa de la tasa ofensiva:

    goles:  (X, Y) ~ DC(e^{z_h}, e^{z_a}, rho)
    tiros:  SOT_h ~ NegBin(media = exp(k0 + k_home + k1 * (alpha_i - delta_j)), forma r)
            SOT_a ~ NegBin(media = exp(k0          + k1 * (alpha_j - delta_i)), forma r)

Los tiros comparten con los goles solo la parte de *fuerzas* del predictor (alpha - delta); el
nivel de la liga mu y la localía eta quedan determinados por los goles, y los tiros tienen su
propio nivel k0 y su propia localía k_home. (Una primera versión usaba el predictor completo
z = mu + eta + alpha - delta también para los tiros: como los tiros traen mucha más información
que los goles, el ajuste distorsionaba mu y eta para explicar los tiros y empeoraba la predicción
de goles. Separar los niveles es lo que hace al modelo coherente.)

Con k1 = 1 la conversión (goles por tiro al arco) no depende de las fuerzas; k1 < 1 significa
que los equipos fuertes convierten mejor de lo que tiran. La binomial negativa absorbe la
heterogeneidad de estilo.

Como la verosimilitud sigue dependiendo de x solo vía z \in R^2, la actualización de Laplace
en dos dimensiones del filtro base se aplica sin cambios: simplemente hay más información
(curvatura) por partido. La pregunta empírica es si esa información extra mejora la
predicción **de goles** fuera de muestra, que es lo que se reporta (``logp_goals``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import gammaln

from .filter import (DEFAULT_HYPER, DynamicDC, _gauge_projector, _loglik_dc, _obs_matrix,
                     _season_transition, init_state, to_arrays, unpack)

SHOT_HYPER = np.r_[DEFAULT_HYPER, np.log(4.4), 1.0, np.log(20.0), 0.15]  # k0, k1, log r, k_home


def _loglik_nb(z, s, k0, k1, r):
    m = jnp.exp(k0 + k1 * z)
    return (gammaln(s + r) - gammaln(r) - gammaln(s + 1) + r * jnp.log(r / (r + m)) + s * jnp.log(m / (r + m)))


def _loglik_full(z, x, y, sh, sa, has, rho, k0, k1, r, base_h, base_a, kh):
    """z = log-tasas de goles; base_* = parte no-fuerza (mu + eta·local) que se resta para los tiros."""
    l = _loglik_dc(z, x, y, rho)
    ls = _loglik_nb(z[0] - base_h, sh, k0 + kh, k1, r) + _loglik_nb(z[1] - base_a, sa, k0, k1, r)
    return l + has * ls


_g_full = jax.grad(_loglik_full)
_h_full = jax.hessian(_loglik_full)
_g_dc = jax.grad(_loglik_dc)
_h_dc = jax.hessian(_loglik_dc)


def _laplace(zbar, S, Sinv, grad, hess, ll, n_newton=6):
    def newton(z, _):
        g = -Sinv @ (z - zbar) + grad(z)
        H = -Sinv + hess(z)
        return z - jnp.linalg.solve(H, g), None
    z, _ = jax.lax.scan(newton, zbar, None, length=n_newton)
    Sig = jnp.linalg.inv(Sinv - hess(z))
    d = z - zbar
    logp = -0.5 * d @ Sinv @ d + ll(z) + 0.5 * jnp.linalg.slogdet(Sig)[1] - 0.5 * jnp.linalg.slogdet(S)[1]
    return z, Sig, logp


def make_filter_shots(N: int):
    D = 2 * N + 2

    def run(h, arr, active, promoted):
        hp = unpack(h[:12])
        k0, k1, r, kh = h[12], h[13], jnp.exp(h[14]), h[15]
        q = jnp.concatenate([jnp.full(2 * N, hp["s_team"] ** 2), jnp.array([hp["s_mu"] ** 2, hp["s_eta"] ** 2])])
        m0, P0 = init_state(hp, N, active[0].astype(float))

        def step(carry, inp):
            m, P = carry
            act = active[inp["si"]].astype(float)
            pro = promoted[inp["si"]].astype(float)
            mt, Pt = _season_transition(m, P, hp, act, pro, N)
            m = jnp.where(inp["new"], mt, m)
            P = jnp.where(inp["new"], Pt, P)
            P = P + jnp.diag(q * jnp.concatenate([act, act, jnp.ones(2)]) * inp["dt"])
            A = _obs_matrix(inp["hi"], inp["ai"], inp["neutral"], N)
            zbar = A @ m
            S = A @ P @ A.T + 1e-9 * jnp.eye(2)
            Sinv = jnp.linalg.inv(S)
            x, y = inp["x"], inp["y"]
            base_h = m[2 * N] + m[2 * N + 1] * (1 - inp["neutral"])
            base_a = m[2 * N]
            args = (inp["sh"], inp["sa"], inp["hs"], hp["rho"], k0, k1, r, base_h, base_a, kh)
            # predictiva de goles sola (para evaluar) ...
            _, _, logp_g = _laplace(zbar, S, Sinv, lambda z: _g_dc(z, x, y, hp["rho"]),
                                    lambda z: _h_dc(z, x, y, hp["rho"]), lambda z: _loglik_dc(z, x, y, hp["rho"]))
            # ... y actualización con goles + tiros
            z, Sig, logp = _laplace(zbar, S, Sinv, lambda z: _g_full(z, x, y, *args),
                                    lambda z: _h_full(z, x, y, *args), lambda z: _loglik_full(z, x, y, *args))
            K = P @ A.T @ Sinv
            m = m + K @ (z - zbar)
            P = P + K @ (Sig - S) @ K.T
            P = 0.5 * (P + P.T)
            return (m, P), dict(zbar=zbar, S=S, logp=logp, logp_goals=logp_g)

        (mT, PT), outs = jax.lax.scan(step, (m0, P0), arr)
        return mT, PT, outs

    return run


class DynamicDCShots(DynamicDC):
    """Mismo modelo, con tiros al arco. ``ld.matches`` debe traer columnas sot_h, sot_a."""

    def __init__(self, ld, h=None):
        self.ld = ld
        self.h = np.array(SHOT_HYPER if h is None else h, float)
        arr = to_arrays(ld)
        from .filter_shots2 import clean_sot  # descarta SOT corruptos de ESPN (p. ej. Premier 2016-17)
        sh, sa, hs = clean_sot(ld.matches)
        arr.update(sh=sh, sa=sa, hs=hs)
        self.arr = {k: jnp.asarray(v) for k, v in arr.items()}
        self.active = jnp.asarray(ld.active)
        self.promoted = jnp.asarray(ld.promoted)
        self._run = make_filter_shots(ld.N)
        self._jrun = jax.jit(lambda h: self._run(h, self.arr, self.active, self.promoted))

    def objective(self, burn_in_seasons: int = 1, upto: int | None = None, l2: float = 1e-2,
                  target: str = "goals"):
        """``target='goals'``: hiperparámetros elegidos por la log-verosimilitud predictiva de los
        **goles** (los tiros solo entran como información en la actualización). ``'joint'``:
        verosimilitud conjunta goles + tiros. La versión conjunta, en la Premier, deja que los
        tiros dominen el ajuste y empeora mucho la predicción de goles (ver docs)."""
        si = np.asarray(self.arr["si"])
        w = (si >= burn_in_seasons).astype(float)
        if upto is not None:
            w[upto:] = 0.0
        w = jnp.asarray(w)
        h0 = jnp.asarray(SHOT_HYPER)

        def f(h):
            _, _, outs = self._run(h, self.arr, self.active, self.promoted)
            lp = outs["logp_goals"] if target == "goals" else outs["logp"]
            return -(lp * w).sum() + l2 * jnp.sum((h - h0) ** 2) * 100.0

        return jax.jit(jax.value_and_grad(f))

    def run(self, states=False):
        mT, PT, outs = self._jrun(jnp.asarray(self.h))
        return np.asarray(mT), np.asarray(PT), {k: np.asarray(v) for k, v in outs.items()}

    def params(self) -> dict:
        p = {k: float(v) for k, v in unpack(jnp.asarray(self.h[:12])).items()}
        p.update(k0=float(self.h[12]), k1=float(self.h[13]), r=float(np.exp(self.h[14])), k_home=float(self.h[15]))
        return p


def with_shots(ld, stats: pd.DataFrame):
    """Agrega sot_h / sot_a a ``ld.matches``."""
    d = ld.matches.merge(stats[["id", "sot_h", "sot_a"]], on="id", how="left")
    ld.matches = d
    return ld
