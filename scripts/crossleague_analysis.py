"""Comparación entre ligas: offsets de liga, resistencia efectiva, Fiedler y ranking global.

Uso: python scripts/crossleague_analysis.py
Escribe outputs/analysis/crossleague_*.csv y crossleague_summary.json
"""
import json

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.model import crossleague as cl
from cadlag.model.filter import DynamicDC, prepare
from cadlag.pipeline import MODELS

m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
OWN = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "chi.1", "bra.1", "arg.1"]
SECOND = {"eng.2": "eng.1", "esp.2": "esp.1", "ita.2": "ita.1", "ger.2": "ger.1", "fra.2": "fra.1", "bra.2": "bra.1"}
models = {}
for c in OWN + list(SECOND):
    h = np.load(MODELS / f"{(c if c in OWN else SECOND[c])}_h.npy")
    models[c] = DynamicDC(prepare(m, c), h)
    print("filtro", c, flush=True)
states = cl.collect_states(models)
edges = cl.cup_edges(m, states)
movers = cl.mover_edges(states)
print(f"{len(edges)} partidos de copa con ambos equipos ubicados; {len(movers)} cambios de división", flush=True)

# --- elegir tau por evidencia de Laplace ---------------------------------------------
best = None
for tau in (0.03, 0.05, 0.08, 0.12, 0.2):
    f = cl.fit(edges, movers, tau=tau)
    print(f"tau={tau}: log evidencia {f.log_evidence:.1f}", flush=True)
    if best is None or f.log_evidence > best.log_evidence:
        best = f
fit = best
print("tau elegido:", fit.tau, "copas:", fit.cups)

out = store.ROOT / "outputs" / "analysis"
out.mkdir(parents=True, exist_ok=True)
net = fit.net()
net.to_csv(out / "crossleague_offsets.csv", index=False)

# --- resistencia efectiva entre ligas en la temporada más reciente con datos -----------
latest = {l: g["season"].max() for l, g in net.groupby("league")}
R = fit.resistance(latest)
R.to_csv(out / "crossleague_resistance.csv")

# --- grafo agregado y vector de Fiedler ------------------------------------------------
G = cl.league_graph(edges, movers)
G.to_csv(out / "crossleague_graph.csv", index=False)
nodes, vals, fv = cl.fiedler(G)

# --- ranking global de equipos actuales ------------------------------------------------
rows = []
for comp, mod in models.items():
    mT, PT, _ = mod.run()
    ld = mod.ld
    syear = int(ld.matches["syear"].max())
    act = ld.active[ld.seasons.index(syear)]
    ls = cl.league_season(comp, ld.matches["date"].iloc[-1].tz_convert(None))
    row = net[(net["league"] == comp) & (net["season"] == ls)]
    if row.empty:
        row = net[net["league"] == comp].sort_values("season").tail(1)
    if row.empty:
        continue
    r, rsd = float(row["net"].iloc[0]), float(row["net_sd"].iloc[0])
    N = ld.N
    for i in np.nonzero(act)[0]:
        v = PT[i, i] + PT[N + i, N + i] + 2 * PT[i, N + i]
        rows.append(dict(team=ld.team_ids[i], name=names.get(ld.team_ids[i]), league=comp,
                         within=float(mT[i] + mT[N + i]), league_offset=r,
                         global_net=float(mT[i] + mT[N + i] + r), sd=float(np.sqrt(v + rsd ** 2))))
rank = pd.DataFrame(rows).sort_values("global_net", ascending=False).reset_index(drop=True)
rank.to_csv(out / "crossleague_global_ranking.csv", index=False)

summary = dict(tau=fit.tau, log_evidence=fit.log_evidence, n_cup_matches=int(len(edges)), n_movers=int(len(movers)),
               cups=fit.cups, latest=net.sort_values("season").groupby("league").tail(1).to_dict("records"),
               fiedler=dict(nodes=nodes, eigenvalues=vals[:4].tolist(), vector=fv.tolist()),
               resistance=R.round(5).to_dict(), top30=rank.head(30).to_dict("records"))
json.dump(summary, open(out / "crossleague_summary.json", "w"), indent=1, default=float)
print(net.sort_values("season").groupby("league").tail(1).sort_values("net", ascending=False).round(3).to_string())
print(rank.head(25).round(3).to_string())
print("Fiedler:", dict(zip(nodes, np.round(fv, 3))))
print(R.round(4).to_string())

# --- salida para la web -------------------------------------------------------------------
web = store.ROOT / "outputs" / "web"
web.mkdir(parents=True, exist_ok=True)
lat = net.sort_values("season").groupby("league").tail(1).sort_values("net", ascending=False)
json.dump(dict(reference="eng.1", tau=fit.tau,
               leagues=lat.to_dict("records"),
               ranking=rank.head(200).to_dict("records"),
               resistance=R.round(5).to_dict(),
               fiedler=dict(zip(nodes, map(float, fv)))),
          open(web / "global.json", "w"), ensure_ascii=False, default=float)
