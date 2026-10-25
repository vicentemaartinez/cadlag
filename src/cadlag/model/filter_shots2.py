r"""Pendiente 1: dos latentes para los tiros, **creación** y **definición**.

R7 mostró que meter los tiros al arco como segunda medición del *mismo* latente no ayuda: o los
tiros dominan y distorsionan, o el acople se apaga. La lectura natural es que los tiros miden
otra cosa. Aquí cada equipo tiene cuatro fuerzas:

  * a_i, b_i : creación y prevención de tiros al arco (SOT)
  * f_i, g_i : definición (convertir un SOT) y "arquero" (evitar que el SOT recibido sea gol)

Para un partido (local i, visita j):

    SOT_h ~ NegBin(media e^{c + k + a_i - b_j}, forma r)          SOT_a ~ NegBin(e^{c + a_j - b_i}, r)
    X | SOT_h ~ Binomial(SOT_h, \sigma(\phi + f_i - g_j))       Y | SOT_a ~ Binomial(SOT_a, \sigma(\phi + f_j - g_i))

**Adelgazamiento exacto.** Como la binomial negativa es una mezcla Poisson–Gamma, adelgazarla
con probabilidad p la deja binomial negativa con la misma forma: X ~ NegBin(m p, r). Así la
predictiva de goles se calcula en forma cerrada sin sumar sobre los tiros, y los partidos sin
estadísticas (ESPN no siempre las trae) informan igual a través de esa marginal.

El estado es (a, b, f, g, c, k, \phi) \in R^{4N+3}; la verosimilitud depende de él solo por
z \in R^4, así que la actualización de Laplace del núcleo se aplica en cuatro dimensiones. Hay
cuatro invarianzas de gauge (una por bloque, compensada por c o \phi), que se fijan con la
misma proyección oblicua.

Los goles que no salen de un tiro al arco (autogoles) se descartan de la binomial: se usa
min(X, SOT). La pregunta empírica es si separar creación y definición mejora la predicción
**de goles** fuera de muestra, y si la definición es una habilidad persistente o ruido (lo dice
el ajuste de su varianza).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import gammaln

from .filter import to_arrays

jax.config.update("jax_enable_x64", True)

HYPER_NAMES = ["log_qs", "log_qf", "log_qlev", "logit_phis", "logit_phif", "log_ss", "log_sf", "log_vprom",
               "a_prom", "b_prom", "log_v0s", "log_v0f", "c0", "k0", "f0", "log_r",
               "a_rel", "b_rel", "log_vrel"]   # prior de los que descienden de la división superior
DEFAULT_HYPER = np.array([np.log(0.0025), np.log(0.001), np.log(0.0006), 2.5, 1.5, np.log(0.08), np.log(0.04),
                          np.log(0.04), -0.12, -0.12, np.log(0.05), np.log(0.02), np.log(4.3), 0.12, -0.85, np.log(20.0),
                          0.10, 0.10, np.log(0.04)])


def pad(h):
    """Hiperparámetros guardados antes del prior de descendidos (16 entradas): se completan con el
    valor por defecto, que no influye en ligas sin división superior."""
    h = np.asarray(h, float)
    return np.r_[h, DEFAULT_HYPER[len(h):]] if len(h) < len(DEFAULT_HYPER) else h


def unpack(h):
    return dict(qs=jnp.exp(h[0]), qf=jnp.exp(h[1]), qlev=jnp.exp(h[2]), phis=jax.nn.sigmoid(h[3]), phif=jax.nn.sigmoid(h[4]),
                ss=jnp.exp(h[5]), sf=jnp.exp(h[6]), vprom=jnp.exp(h[7]), a_prom=h[8], b_prom=h[9],
                v0s=jnp.exp(h[10]), v0f=jnp.exp(h[11]), c0=h[12], k0=h[13], f0=h[14], r=jnp.exp(h[15]),
                a_rel=h[16], b_rel=h[17], vrel=jnp.exp(h[18]))


def _nb(x, logm, r):
    m = jnp.exp(logm)
    return gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * jnp.log(r / (r + m)) + x * (logm - jnp.log(r + m))


def _binom(k, n, logit):
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + k * jax.nn.log_sigmoid(logit) + (n - k) * jax.nn.log_sigmoid(-logit))


def loglik_goals(z, x, y, r):
    """Marginal de goles: NegBin(e^{zs} σ(zf), r) para cada lado."""
    return _nb(x, z[0] + jax.nn.log_sigmoid(z[2]), r) + _nb(y, z[1] + jax.nn.log_sigmoid(z[3]), r)


def loglik_full(z, x, y, sh, sa, has, r):
    xs, ys = jnp.minimum(x, sh), jnp.minimum(y, sa)
    full = _nb(sh, z[0], r) + _nb(sa, z[1], r) + _binom(xs, sh, z[2]) + _binom(ys, sa, z[3])
    return jnp.where(has > 0, full, loglik_goals(z, x, y, r))


_gg, _hg = jax.grad(loglik_goals), jax.hessian(loglik_goals)
_gf, _hf = jax.grad(loglik_full), jax.hessian(loglik_full)


def _A(hi, ai, neu, N):
    D = 4 * N + 3
    A = jnp.zeros((4, D))
    A = A.at[0, hi].add(1).at[0, N + ai].add(-1).at[0, 4 * N].add(1).at[0, 4 * N + 1].add(1 - neu)
    A = A.at[1, ai].add(1).at[1, N + hi].add(-1).at[1, 4 * N].add(1)
    A = A.at[2, 2 * N + hi].add(1).at[2, 3 * N + ai].add(-1).at[2, 4 * N + 2].add(1)
    A = A.at[3, 2 * N + ai].add(1).at[3, 3 * N + hi].add(-1).at[3, 4 * N + 2].add(1)
    return A


def _projector(mask, N):
    D = 4 * N + 3
    n = jnp.maximum(mask.sum(), 1.0)
    z = jnp.zeros(N)
    blocks = []
    for k, (lev, sign) in enumerate([(0, -1.0), (0, 1.0), (2, -1.0), (2, 1.0)]):
        g = jnp.concatenate([mask if j == k else z for j in range(4)] + [jnp.zeros(3).at[lev].set(sign)])
        c = jnp.concatenate([mask if j == k else z for j in range(4)] + [jnp.zeros(3)])
        blocks.append((g, c))
    G = jnp.stack([b[0] for b in blocks], 1); C = jnp.stack([b[1] for b in blocks], 1)
    return jnp.eye(D) - G @ C.T / n


def _laplace(zbar, S, grad, hess, ll, n_newton=8):
    Sinv = jnp.linalg.inv(S)

    def newton(z, _):
        g = -Sinv @ (z - zbar) + grad(z)
        H = -Sinv + hess(z)
        step = jnp.linalg.solve(H, g)
        return z - jnp.clip(step, -1.0, 1.0), None
    z, _ = jax.lax.scan(newton, zbar, None, length=n_newton)
    Sig = jnp.linalg.inv(Sinv - hess(z))
    d = z - zbar
    logp = -0.5 * d @ Sinv @ d + ll(z) + 0.5 * jnp.linalg.slogdet(Sig)[1] - 0.5 * jnp.linalg.slogdet(S)[1]
    return z, Sig, logp


def make_filter(N):
    D = 4 * N + 3

    def run(h, arr, active, promoted, relegated, n_ckpt=0):
        hp = unpack(h)
        team_q = jnp.concatenate([jnp.full(2 * N, hp["qs"] ** 2), jnp.full(2 * N, hp["qf"] ** 2), jnp.full(3, hp["qlev"] ** 2)])
        isS = jnp.concatenate([jnp.ones(2 * N), jnp.zeros(2 * N + 3)])
        isF = jnp.concatenate([jnp.zeros(2 * N), jnp.ones(2 * N), jnp.zeros(3)])
        act0 = active[0].astype(float)
        m0 = jnp.zeros(D).at[4 * N].set(hp["c0"]).at[4 * N + 1].set(hp["k0"]).at[4 * N + 2].set(hp["f0"])
        P0 = jnp.diag(isS * hp["v0s"] + isF * hp["v0f"] + (1 - isS - isF) * 0.05 ** 2)
        Pi = _projector(act0, N); m0, P0 = Pi @ m0, Pi @ P0 @ Pi.T

        def transition(m, P, act, pro, rel):
            sc = isS * hp["phis"] + isF * hp["phif"] + (1 - isS - isF)
            m = m * sc
            P = P * sc[:, None] * sc[None, :] + jnp.diag(isS * hp["ss"] ** 2 + isF * hp["sf"] ** 2)
            r = jnp.concatenate([pro, pro, pro, pro, jnp.zeros(3)])
            rr = jnp.concatenate([rel, rel, jnp.zeros(2 * N + 3)])
            prior = jnp.concatenate([jnp.where(rel > 0, hp["a_rel"], hp["a_prom"]),
                                     jnp.where(rel > 0, hp["b_rel"], hp["b_prom"]), jnp.zeros(2 * N + 3)])
            m = jnp.where(r > 0, prior, m)
            keep = 1 - r
            vS = rr * hp["vrel"] + (1 - rr) * hp["vprom"]
            P = P * keep[:, None] * keep[None, :] + jnp.diag(r * (isS * vS + isF * hp["v0f"]))
            Pi = _projector(act, N)
            return Pi @ m, Pi @ P @ Pi.T

        def step(carry, inp):
            if n_ckpt:
                m, P, Mb, Pb = carry
            else:
                m, P = carry
            act = active[inp["si"]].astype(float); pro = promoted[inp["si"]].astype(float)
            rel = relegated[inp["si"]].astype(float)
            mt, Pt = transition(m, P, act, pro, rel)
            m = jnp.where(inp["new"], mt, m); P = jnp.where(inp["new"], Pt, P)
            mask = jnp.concatenate([act, act, act, act, jnp.ones(3)])
            P = P + jnp.diag(team_q * mask * inp["dt"])
            A = _A(inp["hi"], inp["ai"], inp["neutral"], N)
            zbar = A @ m; S = A @ P @ A.T + 1e-9 * jnp.eye(4)
            x, y, r = inp["x"], inp["y"], hp["r"]
            _, _, logp_g = _laplace(zbar, S, lambda z: _gg(z, x, y, r), lambda z: _hg(z, x, y, r), lambda z: loglik_goals(z, x, y, r))
            a = (inp["sh"], inp["sa"], inp["hs"], r)
            z, Sig, _ = _laplace(zbar, S, lambda z: _gf(z, x, y, *a), lambda z: _hf(z, x, y, *a), lambda z: loglik_full(z, x, y, *a))
            K = P @ A.T @ jnp.linalg.inv(S)
            m = m + K @ (z - zbar); P = P + K @ (Sig - S) @ K.T; P = 0.5 * (P + P.T)
            # estado filtrado (a, b, f, g) de cada equipo tras el partido: lo usa la capa europea
            ih = jnp.array([0, N, 2 * N, 3 * N]) + inp["hi"]; ia = jnp.array([0, N, 2 * N, 3 * N]) + inp["ai"]
            out = dict(zbar=zbar, S=S, logp_goals=logp_g, hm=m[ih], hP=P[jnp.ix_(ih, ih)], am=m[ia], aP=P[jnp.ix_(ia, ia)],
                       phi=m[4 * N + 2])
            if n_ckpt:
                c = inp["ckpt"]; cc = jnp.maximum(c, 0)
                Mb = Mb.at[cc].set(jnp.where(c >= 0, m, Mb[cc])); Pb = Pb.at[cc].set(jnp.where(c >= 0, P, Pb[cc]))
                return (m, P, Mb, Pb), out
            return (m, P), out

        if n_ckpt:
            init = (m0, P0, jnp.zeros((n_ckpt, D)), jnp.zeros((n_ckpt, D, D)))
            (mT, PT, Mb, Pb), outs = jax.lax.scan(step, init, arr)
            outs["ckpt_m"], outs["ckpt_P"] = Mb, Pb
            return mT, PT, outs
        (mT, PT), outs = jax.lax.scan(jax.checkpoint(step), (m0, P0), arr)  # remat: menos memoria en el gradiente
        return mT, PT, outs
    return run


def clean_sot(d, max_bad=0.2):
    """Tiros al arco utilizables. ESPN a veces trae ceros o cifras imposibles (la Premier 2016-17
    entera tiene SOT ≈ 0): se descartan los partidos con SOT total 0 o con menos tiros al arco que
    goles − 1 (deja margen para un autogol), y las temporadas donde eso pasa en más del 20 %."""
    sh = d["sot_h"].to_numpy(float); sa = d["sot_a"].to_numpy(float)
    ok = ~np.isnan(sh) & ~np.isnan(sa)
    ok &= (np.nan_to_num(sh) + np.nan_to_num(sa)) > 0
    ok &= (np.nan_to_num(sh) >= d["hg"].to_numpy() - 1) & (np.nan_to_num(sa) >= d["ag"].to_numpy() - 1)
    present = ~np.isnan(sh) & ~np.isnan(sa)   # faltantes ≠ corruptos: solo los presentes cuentan
    bad = pd.Series(~ok[present]).groupby(d["syear"].to_numpy()[present]).mean()
    ok &= ~d["syear"].isin(bad[bad > max_bad].index).to_numpy()
    return np.nan_to_num(sh), np.nan_to_num(sa), ok.astype(float)


class TwoLatentShots:
    def __init__(self, ld, h=None):
        self.ld = ld
        self.h = pad(DEFAULT_HYPER if h is None else h)
        arr = to_arrays(ld)
        sh, sa, has = clean_sot(ld.matches)
        arr.update(sh=np.nan_to_num(sh), sa=np.nan_to_num(sa), hs=has)
        self.arr = {k: jnp.asarray(v) for k, v in arr.items()}
        self.active, self.promoted = jnp.asarray(ld.active), jnp.asarray(ld.promoted)
        rel = ld.relegated if getattr(ld, "relegated", None) is not None else np.zeros_like(ld.active)
        self.relegated = jnp.asarray(rel)
        self._run = make_filter(ld.N)
        self._jrun = jax.jit(lambda h: self._run(h, self.arr, self.active, self.promoted, self.relegated))

    def fit(self, upto=None, burn_in_seasons=1, maxiter=150, l2=1e-2, verbose=True):
        from scipy.optimize import minimize
        si = np.asarray(self.arr["si"]); w = (si >= burn_in_seasons).astype(float)
        if upto is not None:
            w[upto:] = 0
        w = jnp.asarray(w); h0 = jnp.asarray(DEFAULT_HYPER)

        def f(h):
            _, _, o = self._run(h, self.arr, self.active, self.promoted, self.relegated)
            return -(o["logp_goals"] * w).sum() + l2 * jnp.sum((h - h0) ** 2) * 100.0
        vg = jax.jit(jax.value_and_grad(f))

        def fun(h):
            v, g = vg(jnp.asarray(h)); return float(v), np.nan_to_num(np.asarray(g, float))
        res = minimize(fun, self.h, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
        self.h = res.x
        if verbose:
            print(f"[{self.ld.comp} 2 latentes] -logp/partido = {res.fun / float(np.asarray(w).sum()):.4f} ({res.nit} iter)", flush=True)
        return res

    def run(self):
        mT, PT, o = self._jrun(jnp.asarray(self.h))
        return np.asarray(mT), np.asarray(PT), {k: np.asarray(v) for k, v in o.items()}

    def params(self):
        p = {k: float(v) for k, v in unpack(jnp.asarray(self.h)).items()}
        p["kind"] = "shots2"
        return p

    def run_checkpoints(self, idx):
        """Estados (m, P) completos inmediatamente después de los partidos ``idx`` (sin repetidos)."""
        idx = np.asarray(idx, int)
        ck = -np.ones(len(self.ld.matches), np.int32); ck[idx] = np.arange(len(idx))
        arr = dict(self.arr, ckpt=jnp.asarray(ck))
        f = jax.jit(lambda h: self._run(h, arr, self.active, self.promoted, self.relegated, len(idx)))
        _, _, outs = f(jnp.asarray(self.h))
        outs = {k: np.asarray(v) for k, v in outs.items()}
        return outs.pop("ckpt_m"), outs.pop("ckpt_P"), outs


def season_transition_np(m, P, hp, ld, syear):
    """Transición entre temporadas en numpy (pretemporada del pronóstico y de la trayectoria)."""
    N = ld.N; si = ld.seasons.index(syear)
    act, pro = ld.active[si].astype(float), ld.promoted[si].astype(float)
    rel = ld.relegated[si].astype(float) if getattr(ld, "relegated", None) is not None else np.zeros(N)
    isS = np.r_[np.ones(2 * N), np.zeros(2 * N + 3)]; isF = np.r_[np.zeros(2 * N), np.ones(2 * N), np.zeros(3)]
    sc = isS * hp["phis"] + isF * hp["phif"] + (1 - isS - isF)
    m = m * sc
    P = P * sc[:, None] * sc[None, :] + np.diag(isS * hp["ss"] ** 2 + isF * hp["sf"] ** 2)
    r = np.r_[pro, pro, pro, pro, np.zeros(3)]
    rr = np.r_[rel, rel, np.zeros(2 * N + 3)]
    prior = np.r_[np.where(rel > 0, hp.get("a_rel", hp["a_prom"]), hp["a_prom"]),
                  np.where(rel > 0, hp.get("b_rel", hp["b_prom"]), hp["b_prom"]), np.zeros(2 * N + 3)]
    m = np.where(r > 0, prior, m); keep = 1 - r
    vS = rr * hp.get("vrel", hp["vprom"]) + (1 - rr) * hp["vprom"]
    P = P * keep[:, None] * keep[None, :] + np.diag(r * (isS * vS + isF * hp["v0f"]))
    Pi = np.asarray(_projector(jnp.asarray(act), N))
    return Pi @ m, Pi @ P @ Pi.T


def outcome_probs_mc(zbar, S, r, n_draws=128, maxg=12, seed=0):
    """1X2 por Monte Carlo sobre z ~ N(zbar, S) con goles NegBin independientes (forma cerrada)."""
    from scipy.stats import nbinom
    rng = np.random.default_rng(seed)
    n = len(zbar); out = np.zeros((n, 3)); g = np.arange(maxg + 1)
    L = np.linalg.cholesky(S + 1e-10 * np.eye(4))
    for k in range(n_draws):
        z = zbar + np.einsum("nij,nj->ni", L, rng.standard_normal((n, 4)))
        mh = np.exp(z[:, 0]) / (1 + np.exp(-z[:, 2])); ma = np.exp(z[:, 1]) / (1 + np.exp(-z[:, 3]))
        ph = nbinom.pmf(g[None, :], r, (r / (r + mh))[:, None]); pa = nbinom.pmf(g[None, :], r, (r / (r + ma))[:, None])
        J = ph[:, :, None] * pa[:, None, :]
        out[:, 0] += np.tril(J, -1).sum((1, 2)); out[:, 1] += np.trace(J, axis1=1, axis2=2); out[:, 2] += np.triu(J, 1).sum((1, 2))
    out /= n_draws
    return out / out.sum(1, keepdims=True)
