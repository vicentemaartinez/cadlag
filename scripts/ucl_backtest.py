"""Champions League: pronósticos fuera de muestra de las ediciones 2018-19 a 2025-26.

Hiperparámetros: la capa europea ajustada con temporadas 2011-2017 y los filtros domésticos con
temporadas anteriores a 2016 (``outputs/backtest``). Para cada momento (inicio de la primera fase,
tras las fechas 2 y 4 —y 6 en el formato nuevo—, al terminar la primera fase y antes de cada ronda
de eliminación) se simula el torneo con lo que se sabía ese día y se mide:

  * log score de la etapa alcanzada por cada equipo, promedio por equipo (formato nuevo: fase liga,
    playoff, octavos, cuartos, semis, final, campeón; con grupos: grupos, octavos, ..., campeón);
  * −log P(campeón real).

Compara la capa europea (fuerzas domésticas + residuo europeo) con la misma capa solo con partidos
de copa. Ediciones 2018-19 a 2023-24 con fase de grupos (``sim.ucl_groups``), 2024-25 y 2025-26
con fase liga (``sim.ucl``).

Uso: python scripts/ucl_backtest.py [n_sims]  → outputs/analysis/ucl_backtest.json
"""
import json
import sys

import numpy as np
import pandas as pd

from cadlag.data import store
from cadlag.model import europe as eu
from cadlag.sim import ucl, ucl_groups as ug

S = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
dom = eu.load_domestic(m, "backtest")
fitb = json.load(open(store.ROOT / "outputs" / "europe" / "fit_backtest.json"))
VARIANTS = {"europa": (120, fitb["dom"]["h"]), "solo_copas": (-1, fitb["solo_copas"]["h"])}
comp = "uefa.champions"
PEN_CHAMP = {2025: "Paris Saint-Germain"}   # finales por penales (ESPN no trae la tanda)
rows = []


def winner_of_final(d, syear):
    fin = d[d.slug == "final"].iloc[0]
    if fin.hg == fin.ag:
        return next(k for k, v in names.items() if v == PEN_CHAMP[syear])
    return fin.home if fin.hg > fin.ag else fin.away


for syear in range(2018, 2026):
    new = syear >= 2024
    d = m[(m.comp == comp) & (m.syear == syear)].copy()
    d["slug"] = d["slug"].astype(str).replace(ug.ALIASES)
    first = d[d.slug == ("league-phase" if new else "group-stage")].sort_values("date")
    days = pd.Series(np.sort(first["date"].dt.normalize().unique()))
    jor = (days.diff().dt.days.fillna(99) > 4).cumsum()
    md_end = days.groupby(jor).max().tolist()
    pts = [("inicio", first["date"].min() - pd.Timedelta(hours=1))]
    pts += [(f"fecha {k}", md_end[k - 1] + pd.Timedelta(hours=23, minutes=59)) for k in ((2, 4, 6) if new else (2, 4))]
    pts += [("fin primera fase", first["date"].max() + pd.Timedelta(hours=3))]
    rounds = ucl.ROUNDS[1:] if new else ug.ROUNDS[1:]
    for r in rounds:
        pts.append((f"antes de {r}", d[d.slug == r]["date"].min() - pd.Timedelta(hours=1)))
    full = ucl.cup_state(m, comp, syear, names) if new else ug.group_state(m, comp, syear)
    T = full.teams; n = len(T)
    real = np.zeros(n, int)
    ladder = (["knockout-round-playoffs", "round-of-16", "quarterfinals", "semifinals", "final"] if new
              else ["round-of-16", "quarterfinals", "semifinals", "final"])
    for s, r in enumerate(ladder, start=1):
        k = full.ko.get(r)
        if k is not None:
            for t in set(k.hi) | set(k.ai):
                real[int(t)] = max(real[int(t)], s)
    if new:   # los 8 primeros van directo a octavos (etapa 2)
        k = full.ko["round-of-16"]
        for t in set(k.hi) | set(k.ai):
            real[int(t)] = max(real[int(t)], 2)
    champ = T.index(winner_of_final(d, syear))
    real[champ] = len(ladder) + 1
    for label, asof in pts:
        asof = pd.Timestamp(asof)
        asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof
        cs = ucl.cup_state(m, comp, syear, names, asof=asof) if new else ug.group_state(m, comp, syear, asof=asof)
        for name, (md_, h) in VARIANTS.items():
            ed = eu.build(m, dom, max_days=md_, upto=asof)
            ef = eu.EuroFilter(ed, np.array(h))
            final, _ = ef.run()
            st = eu.strengths_at(ef, final, dom, cs.teams, asof, comp, max_days=md_)
            stg = (ucl.simulate_cup(cs, st.draw, S=S, seed=0, r=st.r)["stage"] if new
                   else ug.simulate_groups(cs, st.draw, S=S, seed=0, r=st.r))
            K = len(ladder) + 2
            P = np.stack([(stg == k).mean(0) for k in range(K)], 1)
            row = dict(syear=syear, formato="liga" if new else "grupos", momento=label, modelo=name,
                       log_etapa=float(-np.log(np.clip(P[np.arange(n), real], 1e-4, 1)).mean()),
                       log_campeon=float(-np.log(max(P[champ, -1], 1e-4))), p_campeon=float(P[champ, -1]),
                       favorito=names[T[int(P[:, -1].argmax())]], p_fav=float(P[:, -1].max()))
            rows.append(row)
            print(row, flush=True)
D = pd.DataFrame(rows)
summ = D.groupby(["formato", "modelo"])[["log_etapa", "log_campeon"]].mean()
print(summ)
out = dict(resumen={f"{a}/{b}": v for (a, b), v in summ.to_dict("index").items()})
for metric in ("log_etapa", "log_campeon"):
    piv = D.pivot_table(index=["syear", "momento"], columns="modelo", values=metric, sort=False)
    dd = piv["europa"] - piv["solo_copas"]
    by_ed = dd.groupby(level=0).mean()          # una cifra por edición: los momentos están correlacionados
    out[f"dif_{metric}"] = dict(media=float(by_ed.mean()), se_ediciones=float(by_ed.std() / np.sqrt(len(by_ed))),
                                por_edicion={int(k): float(v) for k, v in by_ed.items()})
out["filas"] = rows
(store.ROOT / "outputs" / "analysis" / "ucl_backtest.json").write_text(json.dumps(out, indent=1, default=str))
print({k: v for k, v in out.items() if k.startswith("dif")})
