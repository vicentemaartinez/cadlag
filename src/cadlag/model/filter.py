r"""Filtro de Laplace para el modelo dinámico de goles (núcleo M0/M3).

Modelo generativo (para una liga con N equipos)
-------------------------------------------------
Estado latente x_t = (alpha_1..N, delta_1..N, mu, eta) \in R^{2N+2}:

  * alpha_i, delta_i : ataque y defensa del equipo i (mayor delta = mejor defensa)
  * mu               : nivel de goles de la liga (log-tasa base)
  * eta              : ventaja de localía

Dinámica en tiempo continuo (t en días):
  dentro de una temporada   x_{t+dt} = x_t + w,   w ~ N(0, dt * Q),  Q = diag(s_team^2, ..., s_mu^2, s_eta^2)
  entre temporadas          (alpha, delta) <- phi (alpha, delta) + N(0, s_season^2 I)
  equipo ascendido          (alpha_i, delta_i) <- N((a_prom, d_prom), v_prom I), sin correlación previa

Observación de un partido (local i, visita j):
  log lam = mu + eta * (1 - neutral) + alpha_i - delta_j
  log nu  = mu + alpha_j - delta_i
  (X, Y) ~ Dixon–Coles(lam, nu, rho)

Invarianza de gauge
-------------------
La verosimilitud no cambia al mover el estado a lo largo de
  g_a = (1_A, 0, -1, 0)  y  g_d = (0, 1_A, +1, 0),
con A el conjunto de equipos activos. Se fija el gauge con la proyección oblicua
  Pi = I - G (C^T G)^{-1} C^T,
que impone sum_A alpha = sum_A delta = 0 sin cambiar ninguna predicción.

Actualización (ADF / Laplace en dos dimensiones)
------------------------------------------------
La verosimilitud depende de x solo a través de z = A x \in R^2. Con prior z ~ N(zbar, S),
se busca el modo z* de  g(z) = -1/2 (z-zbar)^T S^{-1} (z-zbar) + l(z)  por Newton, con
Sigma* = (S^{-1} - grad^2 l(z*))^{-1}, y se "levanta" al estado completo como un Kalman:
  K = P A^T S^{-1},   m' = m + K (z* - zbar),   P' = P + K (Sigma* - S) K^T.
Es exacto si la verosimilitud fuera gaussiana; con Poisson es la aproximación de Laplace
(equivalente a un filtro de Kalman extendido iterado). Con varianzas fijas y un solo paso de
gradiente se recupera Elo para goles (ver docs).

La log-verosimilitud predictiva un paso adelante (Laplace),
  log p(x_k, y_k | pasado) ~ g(z*) + 1/2 log|Sigma*| - 1/2 log|S|,
es diferenciable respecto de los hiperparámetros: se ajustan maximizando la suma (regla de
puntuación logarítmica prequential, Dawid 1984) con gradientes exactos por autodiferenciación
a través de todo el filtro.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import gammaln

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------------------
# Hiperparámetros
# ----------------------------------------------------------------------------------------
HYPER_NAMES = ["log_s_team", "log_s_mu", "log_s_eta", "logit_phi", "log_s_season", "atanh_rho",
               "log_v_prom", "a_prom", "d_prom", "log_v0", "mu0", "eta0"]

DEFAULT_HYPER = np.array([
    np.log(0.0025),   # s_team: desv. estándar por sqrt(día) (≈ 0.048 por año)
    np.log(0.0008),   # s_mu
    np.log(0.0008),   # s_eta
    2.5,              # logit(phi) -> phi ≈ 0.92
    np.log(0.08),     # s_season
    np.arctanh(-0.08 / 0.3),  # rho = 0.3 * tanh(.)
    np.log(0.05),     # v_prom
    -0.15,            # a_prom (ascendidos: peor ataque)
    -0.15,            # d_prom (y peor defensa)
    np.log(0.10),     # v0
    np.log(1.30),     # mu0
    0.25,             # eta0
])


def unpack(h):
    return dict(
        s_team=jnp.exp(h[0]), s_mu=jnp.exp(h[1]), s_eta=jnp.exp(h[2]),
        phi=jax.nn.sigmoid(h[3]), s_season=jnp.exp(h[4]), rho=0.3 * jnp.tanh(h[5]),
        v_prom=jnp.exp(h[6]), a_prom=h[7], d_prom=h[8], v0=jnp.exp(h[9]), mu0=h[10], eta0=h[11],
    )


# ----------------------------------------------------------------------------------------
# Preparación de datos de una liga
# ----------------------------------------------------------------------------------------
@dataclass
class LeagueData:
    comp: str
    team_ids: list            # id ESPN por índice
    seasons: list             # syear por índice de temporada
    active: np.ndarray        # (S, N) bool: equipos de cada temporada
    promoted: np.ndarray      # (S, N) bool: activos en s y no en s-1
    matches: pd.DataFrame     # partidos terminados ordenados por fecha (con columnas de índice)
    arrays: dict = field(default_factory=dict)
    relegated: np.ndarray | None = None  # (S, N) bool: entrantes que vienen de la división superior

    @property
    def N(self):
        return len(self.team_ids)


def prepare(matches: pd.DataFrame, comp: str, include_playoffs: bool = True,
            min_syear: int | None = None) -> LeagueData:
    d = matches[(matches["comp"] == comp)].copy()
    if not include_playoffs:
        d = d[d["stage"] == "regular"]
    if min_syear is not None:
        d = d[d["syear"] >= min_syear]
    d = d.sort_values(["date", "id"]).reset_index(drop=True)
    seasons = sorted(d["syear"].dropna().unique().tolist())
    teams = sorted(set(d["home"]) | set(d["away"]), key=lambda s: int(s))
    tix = {t: i for i, t in enumerate(teams)}
    six = {s: i for i, s in enumerate(seasons)}
    S, N = len(seasons), len(teams)
    active = np.zeros((S, N), bool)
    for s, g in d.groupby("syear"):
        active[six[s], [tix[t] for t in set(g["home"]) | set(g["away"])]] = True
    promoted = np.zeros_like(active)
    promoted[1:] = active[1:] & ~active[:-1]
    d["hi"] = d["home"].map(tix)
    d["ai"] = d["away"].map(tix)
    d["si"] = d["syear"].map(six)
    fin = d[d["finished"]].reset_index(drop=True)
    relegated = np.zeros_like(active)
    from ..leagues import LEAGUES
    parent = LEAGUES[comp].parent if comp in LEAGUES else None
    if parent:
        up = matches[matches["comp"] == parent]
        for s_, i in six.items():
            prev = up[up["syear"] == s_ - 1]
            came = set(prev["home"]) | set(prev["away"])
            relegated[i] = promoted[i] & np.isin(np.array(teams), list(came))
    return LeagueData(comp, teams, seasons, active, promoted, fin, relegated=relegated)


def to_arrays(ld: LeagueData) -> dict:
    d = ld.matches
    t = (d["date"] - d["date"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    dt = np.diff(t, prepend=t[0])
    si = d["si"].to_numpy()
    new = np.r_[False, si[1:] != si[:-1]]
    return dict(
        hi=d["hi"].to_numpy(np.int32), ai=d["ai"].to_numpy(np.int32),
        x=d["hg"].to_numpy(float), y=d["ag"].to_numpy(float),
        dt=dt.astype(float), new=new, si=si.astype(np.int32),
        neutral=d["neutral"].to_numpy(float),
    )


# ----------------------------------------------------------------------------------------
# Núcleo del filtro (JAX)
# ----------------------------------------------------------------------------------------
def _loglik_dc(z, x, y, rho):
    lam, nu = jnp.exp(z[0]), jnp.exp(z[1])
    ll = x * z[0] - lam - gammaln(x + 1) + y * z[1] - nu - gammaln(y + 1)
    tau = jnp.where((x == 0) & (y == 0), 1 - lam * nu * rho,
          jnp.where((x == 0) & (y == 1), 1 + lam * rho,
          jnp.where((x == 1) & (y == 0), 1 + nu * rho,
          jnp.where((x == 1) & (y == 1), 1 - rho, 1.0))))
    return ll + jnp.log(jnp.maximum(tau, 1e-10))


_grad_l = jax.grad(_loglik_dc)
_hess_l = jax.hessian(_loglik_dc)


def _obs_matrix(hi, ai, neutral, N):
    D = 2 * N + 2
    a = jnp.zeros((2, D))
    a = a.at[0, hi].add(1.0).at[0, N + ai].add(-1.0).at[0, 2 * N].add(1.0).at[0, 2 * N + 1].add(1.0 - neutral)
    a = a.at[1, ai].add(1.0).at[1, N + hi].add(-1.0).at[1, 2 * N].add(1.0)
    return a


def _gauge_projector(mask, N):
    """Proyección oblicua que impone sum_A alpha = sum_A delta = 0 (ver docstring)."""
    D = 2 * N + 2
    n = jnp.maximum(mask.sum(), 1.0)
    ga = jnp.concatenate([mask, jnp.zeros(N), jnp.array([-1.0, 0.0])])
    gd = jnp.concatenate([jnp.zeros(N), mask, jnp.array([1.0, 0.0])])
    ca = jnp.concatenate([mask, jnp.zeros(N + 2)])
    cd = jnp.concatenate([jnp.zeros(N), mask, jnp.zeros(2)])
    G = jnp.stack([ga, gd], 1)
    C = jnp.stack([ca, cd], 1)
    return jnp.eye(D) - G @ C.T / n


def init_state(hp, N, active0):
    D = 2 * N + 2
    m = jnp.zeros(D).at[2 * N].set(hp["mu0"]).at[2 * N + 1].set(hp["eta0"])
    v = jnp.concatenate([jnp.full(2 * N, hp["v0"]), jnp.array([0.05 ** 2, 0.05 ** 2])])
    P = jnp.diag(v)
    Pi = _gauge_projector(active0, N)
    return Pi @ m, Pi @ P @ Pi.T


def _season_transition(m, P, hp, active, promoted, N):
    D = 2 * N + 2
    team = jnp.concatenate([jnp.ones(2 * N), jnp.zeros(2)])
    scale = jnp.where(team > 0, hp["phi"], 1.0)
    m = m * scale
    P = P * scale[:, None] * scale[None, :] + jnp.diag(team * hp["s_season"] ** 2)
    r = jnp.concatenate([promoted, promoted, jnp.zeros(2)])
    prior_m = jnp.concatenate([jnp.full(N, hp["a_prom"]), jnp.full(N, hp["d_prom"]), jnp.zeros(2)])
    m = jnp.where(r > 0, prior_m, m)
    keep = 1.0 - r
    P = P * keep[:, None] * keep[None, :] + jnp.diag(r * hp["v_prom"])
    Pi = _gauge_projector(active, N)
    return Pi @ m, Pi @ P @ Pi.T


def _laplace_update(m, P, A, x, y, rho, n_newton=6):
    zbar = A @ m
    S = A @ P @ A.T + 1e-9 * jnp.eye(2)
    Sinv = jnp.linalg.inv(S)

    def newton(z, _):
        g = -Sinv @ (z - zbar) + _grad_l(z, x, y, rho)
        H = -Sinv + _hess_l(z, x, y, rho)
        return z - jnp.linalg.solve(H, g), None

    z, _ = jax.lax.scan(newton, zbar, None, length=n_newton)
    Hpost = Sinv - _hess_l(z, x, y, rho)
    Sig = jnp.linalg.inv(Hpost)
    K = P @ A.T @ Sinv
    m_new = m + K @ (z - zbar)
    P_new = P + K @ (Sig - S) @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    d = z - zbar
    logp = (-0.5 * d @ Sinv @ d + _loglik_dc(z, x, y, rho)
            + 0.5 * jnp.linalg.slogdet(Sig)[1] - 0.5 * jnp.linalg.slogdet(S)[1])
    return m_new, P_new, zbar, S, logp


def make_filter(N: int):
    D = 2 * N + 2

    def run(h, arr, active, promoted, return_states=False, n_ckpt=0):
        """Corre el filtro. Con ``n_ckpt > 0``, ``arr`` debe traer ``ckpt`` (índice de buffer o -1)
        y se devuelven además las medias y covarianzas completas guardadas en esos partidos."""
        hp = unpack(h)
        q_team = jnp.concatenate([jnp.full(2 * N, hp["s_team"] ** 2), jnp.array([hp["s_mu"] ** 2, hp["s_eta"] ** 2])])
        m0, P0 = init_state(hp, N, active[0].astype(float))

        def step(carry, inp):
            if n_ckpt:
                m, P, Mb, Pb = carry
            else:
                m, P = carry
            act = active[inp["si"]].astype(float)
            pro = promoted[inp["si"]].astype(float)
            mt, Pt = _season_transition(m, P, hp, act, pro, N)
            m = jnp.where(inp["new"], mt, m)
            P = jnp.where(inp["new"], Pt, P)
            mask = jnp.concatenate([act, act, jnp.ones(2)])
            P = P + jnp.diag(q_team * mask * inp["dt"])
            A = _obs_matrix(inp["hi"], inp["ai"], inp["neutral"], N)
            m, P, zbar, S, logp = _laplace_update(m, P, A, inp["x"], inp["y"], hp["rho"])
            out = dict(zbar=zbar, S=S, logp=logp)
            if return_states:
                out["m"] = m
                out["Pdiag"] = jnp.diag(P)
            if n_ckpt:
                c = inp["ckpt"]
                cc = jnp.maximum(c, 0)
                Mb = Mb.at[cc].set(jnp.where(c >= 0, m, Mb[cc]))
                Pb = Pb.at[cc].set(jnp.where(c >= 0, P, Pb[cc]))
                return (m, P, Mb, Pb), out
            return (m, P), out

        if n_ckpt:
            init = (m0, P0, jnp.zeros((n_ckpt, D)), jnp.zeros((n_ckpt, D, D)))
            (mT, PT, Mb, Pb), outs = jax.lax.scan(step, init, arr)
            outs["ckpt_m"], outs["ckpt_P"] = Mb, Pb
            return mT, PT, outs
        (mT, PT), outs = jax.lax.scan(step, (m0, P0), arr)
        return mT, PT, outs

    return run


# ----------------------------------------------------------------------------------------
# Interfaz de alto nivel
# ----------------------------------------------------------------------------------------
class DynamicDC:
    """Modelo dinámico de Dixon–Coles para una liga, ajustado por máxima verosimilitud prequential."""

    def __init__(self, ld: LeagueData, h: np.ndarray | None = None):
        self.ld = ld
        self.h = np.array(DEFAULT_HYPER if h is None else h, float)
        self.arr = {k: jnp.asarray(v) for k, v in to_arrays(ld).items()}
        self.active = jnp.asarray(ld.active)
        self.promoted = jnp.asarray(ld.promoted)
        self._run = make_filter(ld.N)
        self._jrun = jax.jit(lambda h: self._run(h, self.arr, self.active, self.promoted))
        self._jrun_states = jax.jit(lambda h: self._run(h, self.arr, self.active, self.promoted, True))

    # ---- ajuste --------------------------------------------------------------------
    def objective(self, burn_in_seasons: int = 1, upto: int | None = None, l2: float = 1e-2):
        si = np.asarray(self.arr["si"])
        w = (si >= burn_in_seasons).astype(float)
        if upto is not None:
            w[upto:] = 0.0
        w = jnp.asarray(w)
        h0 = jnp.asarray(DEFAULT_HYPER)

        def f(h):
            _, _, outs = self._run(h, self.arr, self.active, self.promoted)
            # suma (no promedio): el prior gaussiano débil sobre h pesa como l2 'partidos'
            return -(outs["logp"] * w).sum() + l2 * jnp.sum((h - h0) ** 2) * 100.0

        return jax.jit(jax.value_and_grad(f))

    def fit(self, burn_in_seasons: int = 1, upto: int | None = None, maxiter: int = 200, verbose=True):
        from scipy.optimize import minimize

        vg = self.objective(burn_in_seasons, upto)

        def fun(h):
            v, g = vg(jnp.asarray(h))
            return float(v), np.asarray(g, float)

        res = minimize(fun, self.h, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
        self.h = res.x
        if verbose:
            w = np.asarray(self.arr["si"]) >= burn_in_seasons
            if upto is not None:
                w[upto:] = False
            n = float(w.sum())
            print(f"[{self.ld.comp}] -logp/partido = {res.fun / n:.4f}  ({res.nit} iter, {res.message})")
        return res

    # ---- salidas ---------------------------------------------------------------------
    def run(self, states=False):
        f = self._jrun_states if states else self._jrun
        mT, PT, outs = f(jnp.asarray(self.h))
        return np.asarray(mT), np.asarray(PT), {k: np.asarray(v) for k, v in outs.items()}

    def run_checkpoints(self, idx) -> tuple[np.ndarray, np.ndarray, dict]:
        """Estados (m, P) completos inmediatamente después de los partidos ``idx`` (índices en ld.matches)."""
        idx = np.asarray(idx, int)
        ck = -np.ones(len(self.ld.matches), np.int32)
        ck[idx] = np.arange(len(idx))
        arr = dict(self.arr, ckpt=jnp.asarray(ck))
        f = jax.jit(lambda h: self._run(h, arr, self.active, self.promoted, False, len(idx)))
        _, _, outs = f(jnp.asarray(self.h))
        outs = {k: np.asarray(v) for k, v in outs.items()}
        return outs.pop("ckpt_m"), outs.pop("ckpt_P"), outs

    def params(self) -> dict:
        return {k: float(v) for k, v in unpack(jnp.asarray(self.h)).items()}
