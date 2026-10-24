r"""Estilos latentes: interacción bilineal de bajo rango sobre el modelo dinámico.

Dado el predictor del filtro (log-tasas un paso adelante zbar_h, zbar_a), se agrega

    log lam_ij = zbar_h + u_i^T v_j        log nu_ij = zbar_a + u_j^T v_i

con u_i \in R^K el "estilo ofensivo" de i y v_j la "vulnerabilidad" de la defensa de j a ese
estilo. Con K pequeño (1-3) y priors de relevancia automática (ARD)
    u_ik, v_ik ~ N(0, s_k^2),   s_k ~ HalfNormal(0.1),
las columnas que no aportan se encogen a cero.

Propiedades:
  * Gauge: (U, V) -> (U G, V G^{-T}) deja U V^T invariante para G \in GL(K). Se reporta
    Gamma = U V^T y su SVD; para dibujar se alinean las muestras por Procrustes.
  * Intransitividad inducida: en la aproximación normal de la diferencia de goles,
    E[GD_ij] - (término de fuerzas) ≈ e^{z} (u_i^T v_j - u_j^T v_i): la parte antisimétrica
    de Gamma, Gamma - Gamma^T, es un término tipo "blade–chest" (Chen & Joachims, 2016) que
    puede producir ciclos i > j > k > i aunque las fuerzas sean transitivas.
  * Prueba: se ajusta en una ventana de entrenamiento y se evalúa la log-verosimilitud
    *fuera de muestra* frente al modelo sin interacción (Diebold–Mariano pareado).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

from .dixon_coles import score_grid


def model(hi, ai, zh, za, x=None, y=None, n_teams=0, K=2, scale=None):
    if scale is None:
        s = numpyro.sample("s", dist.HalfNormal(0.1).expand([K]))
    else:
        s = jnp.full(K, scale)
    with numpyro.plate("teams", n_teams):
        U = numpyro.sample("U", dist.Normal(0.0, 1.0).expand([K]).to_event(1)) * s
        V = numpyro.sample("V", dist.Normal(0.0, 1.0).expand([K]).to_event(1)) * s
    numpyro.deterministic("Us", U)
    numpyro.deterministic("Vs", V)
    ih = (U[hi] * V[ai]).sum(-1)
    ia = (U[ai] * V[hi]).sum(-1)
    with numpyro.plate("matches", len(hi)):
        numpyro.sample("x", dist.Poisson(jnp.exp(zh + ih)), obs=x)
        numpyro.sample("y", dist.Poisson(jnp.exp(za + ia)), obs=y)


def fit_map(hi, ai, zh, za, x, y, n_teams, K=2, scale=0.1, steps=3000, seed=0):
    """MAP de (U, V) con escala de prior FIJA.

    Nota: el MAP conjunto de (U, V, s) con s jerárquico es degenerado: la densidad posterior
    diverge cuando s -> 0 con U = V = 0 (el "embudo" de Neal), así que el optimizador colapsa
    la interacción a cero. Por eso la escala se fija (y se compara una grilla) en vez de
    estimarse por MAP; la versión jerárquica se ajusta con NUTS (``fit_nuts``).
    """
    # Inicio aleatorio: (U, V) = (0, 0) es un punto silla del problema bilineal (gradiente nulo).
    guide = autoguide.AutoDelta(model, init_loc_fn=numpyro.infer.init_to_sample)
    svi = SVI(model, guide, numpyro.optim.Adam(0.02), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), steps, jnp.asarray(hi), jnp.asarray(ai), jnp.asarray(zh),
                  jnp.asarray(za), jnp.asarray(x), jnp.asarray(y), n_teams=n_teams, K=K, scale=scale,
                  progress_bar=False)
    p = guide.median(res.params)
    return np.asarray(p["U"]) * scale, np.asarray(p["V"]) * scale, np.full(K, scale)


def fit_nuts(hi, ai, zh, za, x, y, n_teams, K=2, warmup=500, samples=500, seed=0):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(hi), jnp.asarray(ai), jnp.asarray(zh), jnp.asarray(za),
             jnp.asarray(x), jnp.asarray(y), n_teams=n_teams, K=K)
    smp = mcmc.get_samples()
    return np.asarray(smp["Us"]), np.asarray(smp["Vs"]), np.asarray(smp["s"]), mcmc


def interaction_logscore(zh, za, x, y, U, V, hi, ai, rho):
    """Log score (marcador exacto, Dixon–Coles) con y sin la interacción."""
    ih = (U[hi] * V[ai]).sum(-1) if U is not None else 0.0
    ia = (U[ai] * V[hi]).sum(-1) if U is not None else 0.0
    g = score_grid(np.exp(zh + ih), np.exp(za + ia), rho)
    xi, yi = np.clip(x, 0, 10).astype(int), np.clip(y, 0, 10).astype(int)
    return -np.log(g[np.arange(len(x)), xi, yi])


def procrustes_align(Us: np.ndarray, Vs: np.ndarray, ref_U: np.ndarray):
    """Alinea muestras (S, n, K) a una referencia por rotación ortogonal (en U; V con la misma)."""
    out_U, out_V = np.empty_like(Us), np.empty_like(Vs)
    for s in range(len(Us)):
        M = Us[s].T @ ref_U
        A, _, Bt = np.linalg.svd(M)
        R = A @ Bt
        out_U[s] = Us[s] @ R
        out_V[s] = Vs[s] @ R
    return out_U, out_V


# ----------------------------------------------------------------------------------------
# Prueba espectral
# ----------------------------------------------------------------------------------------
def residual_matrix(hi, ai, x, y, lam, nu, n):
    """R[a, d] = sum (goles de a contra d - esperados) / sqrt(sum esperados): atacante x defensor.

    Bajo el modelo sin interacción, cada entrada es ~ N(0, 1) (aprox.) e independiente, así que
    R es una matriz de ruido; una interacción de bajo rango Gamma = U V^T aparece como valores
    singulares que sobresalen del "bulto" de ruido (teoría de matrices aleatorias). Además,
    el MAP de (U, V) con prior N(0, s^2) es el estimador con penalización de norma nuclear
    ||Gamma||_* / s^2, que es exactamente cero si ||grad||_2 <= 1/s^2: el umbral espectral.
    """
    num = np.zeros((n, n)); den = np.zeros((n, n))
    np.add.at(num, (hi, ai), x - lam); np.add.at(den, (hi, ai), lam)
    np.add.at(num, (ai, hi), y - nu); np.add.at(den, (ai, hi), nu)
    R = np.where(den > 0, num / np.sqrt(np.maximum(den, 1e-12)), 0.0)
    # Doble centrado: un error en las fuerzas (alpha_a, delta_d) produce residuos aditivos
    # fila + columna (rango <= 2) que no son "estilo". Se eliminan antes de mirar el espectro.
    obs = den > 0
    for _ in range(20):
        rm = np.where(obs, R, 0).sum(1) / np.maximum(obs.sum(1), 1)
        R = np.where(obs, R - rm[:, None], 0)
        cm = np.where(obs, R, 0).sum(0) / np.maximum(obs.sum(0), 1)
        R = np.where(obs, R - cm[None, :], 0)
    return R


def spectral_test(hi, ai, x, y, lam, nu, n, rho, B=200, k=3, seed=0):
    from ..sim.season import sample_dc
    rng = np.random.default_rng(seed)
    sv = np.linalg.svd(residual_matrix(hi, ai, x, y, lam, nu, n), compute_uv=False)[:k]
    null = np.zeros((B, k))
    for b in range(B):
        xb, yb = sample_dc(lam, nu, rho, rng)
        null[b] = np.linalg.svd(residual_matrix(hi, ai, xb, yb, lam, nu, n), compute_uv=False)[:k]
    p = (null >= sv[None, :]).mean(0)
    return sv, null, p
