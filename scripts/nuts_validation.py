"""Validación del filtro de Laplace contra el posterior exacto (NUTS) en una temporada.

Se toma el estado del filtro al inicio de la temporada como prior gaussiano, se modela el
estado semanal con paseo aleatorio (mismos hiperparámetros) y la verosimilitud Dixon–Coles de
todos los partidos de la temporada hasta hoy, y se muestrea con NUTS (NumPyro). Al final de la
ventana, el posterior "suavizado" en el último instante coincide con el posterior de
*filtrado*, así que la comparación es directa: medias y desviaciones de las fuerzas netas, y
probabilidades de posición final con ambos.

Uso: python scripts/nuts_validation.py eng.1 2025 [semanas_max]
"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from cadlag.data import store
from cadlag.leagues import LEAGUES
from cadlag.model.filter import _loglik_dc, prepare
from cadlag.pipeline import load_or_fit
from cadlag.sim.season import season_state, simulate

jax.config.update("jax_enable_x64", True)
comp, syear = sys.argv[1], int(sys.argv[2])
m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
ld = prepare(m, comp)
mod = load_or_fit(ld)
hp = mod.params()
d = ld.matches
first = int(np.argmax(d["syear"].to_numpy() == syear))
idx_season = np.nonzero(d["syear"].to_numpy() == syear)[0]
# corte a mitad de temporada para que queden partidos por simular
cut = idx_season[len(idx_season) // 2]
Ms, Ps, _ = mod.run_checkpoints([first - 1, cut])
m0, P0 = Ms[0], Ps[0]
mF, PF = Ms[1], Ps[1]
from cadlag.eval.backtest import preseason_transition
m0, P0 = preseason_transition(m0, P0, hp, ld, syear)
N = ld.N
act = np.nonzero(ld.active[ld.seasons.index(syear)])[0]
idx = np.r_[act, N + act, 2 * N, 2 * N + 1]
n = len(act)
loc = {g: i for i, g in enumerate(act)}
games = d.iloc[first:cut + 1]
t0 = games["date"].iloc[0]
week = ((games["date"] - t0).dt.days // 7).to_numpy()
W = int(week.max()) + 1
hi = games["hi"].map(loc).to_numpy(); ai = games["ai"].map(loc).to_numpy()
x = games["hg"].to_numpy(float); y = games["ag"].to_numpy(float); neu = games["neutral"].to_numpy(float)
mu_prior = m0[idx]
L0 = np.linalg.cholesky(P0[np.ix_(idx, idx)] + 1e-6 * np.eye(len(idx)))
qd = np.r_[np.full(2 * n, hp["s_team"] ** 2), hp["s_mu"] ** 2, hp["s_eta"] ** 2] * 7.0


def model():
    e0 = numpyro.sample("e0", dist.Normal(0, 1).expand([len(idx)]).to_event(1))
    x0 = mu_prior + L0 @ e0
    inc = numpyro.sample("inc", dist.Normal(0, 1).expand([W - 1, len(idx)]).to_event(2)) * jnp.sqrt(qd)
    X = jnp.concatenate([x0[None], x0[None] + jnp.cumsum(inc, 0)], 0)  # (W, D)
    Xm = X[week]
    zh = Xm[:, 2 * n] + Xm[:, 2 * n + 1] * (1 - neu) + Xm[jnp.arange(len(hi)), hi] - Xm[jnp.arange(len(hi)), n + ai]
    za = Xm[:, 2 * n] + Xm[jnp.arange(len(hi)), ai] - Xm[jnp.arange(len(hi)), n + hi]
    ll = jax.vmap(lambda a, b, c, e: _loglik_dc(jnp.array([a, b]), c, e, hp["rho"]))(zh, za, x, y)
    numpyro.factor("lik", ll.sum())
    numpyro.deterministic("xT", X[-1])


mcmc = MCMC(NUTS(model, target_accept_prob=0.85), num_warmup=600, num_samples=600, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0))
xT = np.asarray(mcmc.get_samples()["xT"])
diverg = int(mcmc.get_extra_fields()["diverging"].sum())
net_n = xT[:, :n] + xT[:, n:2 * n]
net_f = mF[act] + mF[N + act]
sd_f = np.sqrt(np.diag(PF)[act] + np.diag(PF)[N + act] + 2 * PF[act, N + act])
# las fuerzas netas se comparan centradas (el gauge solo fija la suma)
net_n_c = net_n - net_n.mean(1, keepdims=True)
res = dict(comp=comp, syear=syear, n_matches=len(games), weeks=W, divergences=diverg,
           corr_means=float(np.corrcoef(net_n_c.mean(0), net_f - net_f.mean())[0, 1]),
           max_abs_mean_diff=float(np.abs(net_n_c.mean(0) - (net_f - net_f.mean())).max()),
           sd_ratio_mean=float((net_n_c.std(0) / sd_f).mean()),
           teams=[dict(name=names.get(ld.team_ids[a]), nuts_mean=float(net_n_c.mean(0)[k]), nuts_sd=float(net_n_c.std(0)[k]),
                       filter_mean=float(net_f[k] - net_f.mean()), filter_sd=float(sd_f[k])) for k, a in enumerate(act)])
# probabilidades de posición: filtro (gaussiana) vs muestras NUTS
tix = {t: i for i, t in enumerate(ld.team_ids)}
st = season_state(m, comp, syear, tix, asof=games["date"].iloc[-1])
mi = st.model_index
order = [list(act).index(j) for j in mi]
Xd = np.concatenate([xT[:, :n][:, order], xT[:, n:2 * n][:, order], xT[:, 2 * n:]], 1)
Xd = np.repeat(Xd, 10, axis=0)
rF = simulate(LEAGUES[comp], st, mF, PF, hp, N, n_sims=len(Xd), seed=1)
rN = simulate(LEAGUES[comp], st, mF, PF, hp, N, n_sims=len(Xd), seed=1, x_draw=Xd)
MF, MN = rF.position_matrix(), rN.position_matrix()
res["position_tv_mean"] = float(0.5 * np.abs(MF - MN).sum(1).mean())  # distancia de variación total media por equipo
res["position_tv_max"] = float(0.5 * np.abs(MF - MN).sum(1).max())
out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / f"nuts_validation_{comp}_{syear}.json", "w"), indent=1, default=float)
print({k: v for k, v in res.items() if k != "teams"})
