"""Pendiente 2: traspasos como aristas del grafo entre ligas.

Un traspaso definitivo entre dos ligas del grafo es una comparación: el jugador "sube" hacia la
liga más fuerte. Se modela P(A → B) = σ(β (r_B − r_A)) sobre las mismas fuerzas netas r de la
comparación entre ligas (copas + cambios de división), con cuasi-verosimilitud (dispersión κ por
Pearson). β se elige por validación cruzada en 5 partes sobre los partidos de copa: la pregunta es
si los traspasos mejoran la predicción de partidos que el modelo no vio.

Además: jerarquía de ligas solo por traspasos (Bradley–Terry agregado) y su correlación con las
fuerzas de los partidos; y el efecto sobre la incertidumbre y la resistencia Europa–Sudamérica.

Uso: python scripts/transfer_edges_analysis.py
Escribe outputs/analysis/transfer_edges.json y transfer_edges_offsets.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cadlag.data import store, transfers
from cadlag.model import crossleague as cl
from cadlag.model.filter import DynamicDC, prepare
from cadlag.pipeline import MODELS

m, teams = store.load()
OWN = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "chi.1", "bra.1", "arg.1"]
SECOND = {"eng.2": "eng.1", "esp.2": "esp.1", "ita.2": "ita.1", "ger.2": "ger.1", "fra.2": "fra.1", "bra.2": "bra.1"}
models = {c: DynamicDC(prepare(m, c), np.load(MODELS / f"{(c if c in OWN else SECOND[c])}_h.npy")) for c in OWN + list(SECOND)}
states = cl.collect_states(models)
edges, movers = cl.cup_edges(m, states), cl.mover_edges(states)
TAU = json.loads((store.ROOT / "outputs" / "analysis" / "crossleague_summary.json").read_text())["tau"]
tr = transfers.load(m, teams)
pairs = cl.transfer_pairs(tr)
print(f"{len(edges)} partidos de copa, {len(movers)} cambios de división, {int(pairs.n_ab.sum() + pairs.n_ba.sum())} traspasos entre ligas del grafo", flush=True)


def pearson(f, pairs):
    th = f.theta_full; r = []
    for p in pairs.itertuples():
        a, b = f.kix.get((p.la, p.sa)), f.kix.get((p.lb, p.sb))
        if a is None or b is None:
            continue
        x = th[2 * b] + th[2 * b + 1] - th[2 * a] - th[2 * a + 1]
        pr = 1 / (1 + np.exp(-f.beta * x)); n = p.n_ab + p.n_ba
        r.append((p.n_ab - n * pr) ** 2 / max(n * pr * (1 - pr), 1e-9))
    return float(np.sum(r) / max(len(r) - 1, 1))


def fit_beta(beta, E=edges):
    if beta == 0:
        return cl.fit(E, movers, tau=TAU)
    f = cl.fit(E, movers, tau=TAU, transfers=pairs, beta=beta)
    k = max(1.0, pearson(f, pairs))
    return cl.fit(E, movers, tau=TAU, transfers=pairs, beta=beta, kappa=k)


rng = np.random.default_rng(0)
fold = rng.integers(0, 5, len(edges))
cv = {}
for beta in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
    ll = []
    for k in range(5):
        f = fit_beta(beta, edges[fold != k])
        ll.append(np.nansum(f.loglik(edges[fold == k])))
    full = fit_beta(beta)
    cv[beta] = dict(cv_loglik=float(np.sum(ll)), kappa=full.kappa)
    print(f"β={beta}: log-verosimilitud fuera de muestra {np.sum(ll):.1f}  (κ={full.kappa:.2f})", flush=True)
best = max(cv, key=lambda b: cv[b]["cv_loglik"])
base, aug = fit_beta(0.0), fit_beta(best if best > 0 else 2.0)

# --- comparación de fuerzas y resistencias (última temporada de cada liga) -------------
nb, na = base.net(), aug.net()
lat = nb.sort_values("season").groupby("league").tail(1)[["league", "season"]]
cmp = lat.merge(nb, on=["league", "season"]).merge(na, on=["league", "season"], suffixes=("_base", "_traspasos"))
latest = dict(zip(lat.league, lat.season))
Rb, Ra = base.resistance(latest), aug.resistance(latest)
south = [l for l in ("bra.1", "arg.1", "chi.1") if l in latest]

# --- jerarquía solo con traspasos (Bradley–Terry agregado por liga) --------------------
L = sorted(set(pairs.la) | set(pairs.lb)); ix = {l: i for i, l in enumerate(L)}
agg = pairs.groupby(["la", "lb"])[["n_ab", "n_ba"]].sum().reset_index()


def nll(s):
    s = np.r_[0.0, s]  # la primera liga anclada
    x = np.array([s[ix[b]] - s[ix[a]] for a, b in zip(agg.la, agg.lb)])
    return -(agg.n_ab * -np.logaddexp(0, -x) + agg.n_ba * -np.logaddexp(0, x)).sum() + 0.5 * 1e-4 * (s ** 2).sum()


sol = minimize(nll, np.zeros(len(L) - 1), method="BFGS")
bt = dict(zip(L, np.r_[0.0, sol.x] - np.r_[0.0, sol.x][ix["eng.1"]]))
mean_net = nb[nb.season >= 2015].groupby("league")["net"].mean()
common = [l for l in L if l in mean_net.index]
corr = float(np.corrcoef([bt[l] for l in common], [mean_net[l] for l in common])[0, 1])
spear = float(pd.Series([bt[l] for l in common]).rank().corr(pd.Series([mean_net[l] for l in common]).rank()))

out = store.ROOT / "outputs" / "analysis"
cmp.to_csv(out / "transfer_edges_offsets.csv", index=False)
res = dict(tau=TAU, n_transfers=int(pairs.n_ab.sum() + pairs.n_ba.sum()), n_pairs=int(len(pairs)), cv=cv, beta_best=best,
           kappa=aug.kappa, bt_hierarchy=bt, corr_bt_vs_matches=corr, spearman_bt_vs_matches=spear,
           offsets=cmp.round(4).to_dict("records"),
           resistance_eng_south=dict(base={s: float(Rb.loc["eng.1", s]) for s in south}, traspasos={s: float(Ra.loc["eng.1", s]) for s in south}),
           sd_ratio=dict(zip(cmp.league, (cmp.net_sd_traspasos / cmp.net_sd_base).round(3))))
print(json.dumps(res, indent=1, default=float))
(out / "transfer_edges.json").write_text(json.dumps(res, indent=1, default=float))
