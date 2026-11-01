"""Libertadores (2017-2025) y Sudamericana (2023-2025): pronósticos fuera de muestra.

Hiperparámetros de la capa sudamericana ajustados con 2011-2020 (``scripts/europe_fit.py sudamerica``)
y filtros domésticos de backtest. En cada edición, al empezar los grupos, tras las fechas 2 y 4, al
terminar los grupos y antes de cada ronda de eliminación, se simulan las dos copas juntas con lo que
se sabía ese día y se mide el log score de la etapa alcanzada por cada equipo y −log P(campeón).
Compara la capa anclada a las ligas con modelo doméstico con la misma capa solo con partidos de copa.

Uso: python scripts/conmebol_backtest.py [n_sims]  → outputs/analysis/conmebol_backtest.json
"""
import json
import sys

import numpy as np
import pandas as pd

from cadlag.cup_conmebol import strengths_for
from cadlag.data import store
from cadlag.model import europe as eu
from cadlag.sim import conmebol as cm

S = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
m, teams = store.load()
names = dict(zip(teams["team_id"], teams["name"]))
dom = eu.load_domestic(m, "backtest", region=eu.SOUTH)
fitb = json.load(open(eu.SOUTH.cache / "fit_backtest.json"))
VARIANTS = {"sudamerica": (120, fitb["dom"]["h"]), "solo_copas": (-1, fitb["solo_copas"]["h"])}
CHAMP = {}   # campeón real por (copa, año), de la final (y de shootouts.csv si fue por penales)
sh = cm._shootout_winners()
ids = {v: k for k, v in names.items()}
rows = []
for syear in [int(y) for y in __import__("os").environ.get("YEARS", "").split()] or range(2017, 2026):
    comps = (cm.LIB, cm.SUD) if syear >= 2023 else (cm.LIB,)
    d = m[m["comp"].isin(comps) & (m["syear"] == syear)].copy()
    d["slug"] = d["slug"].astype(str).replace({"finals": "final"})
    full = cm.conmebol_state(m, syear, names, comps=comps)
    T = full.teams; n = len(T)
    real, fin_date = {}, {}
    for comp in comps:
        dc = d[d["comp"] == comp]
        fin = dc[dc["slug"] == "final"].sort_values("date")
        fin_date[comp] = fin["date"].iloc[-1] - pd.Timedelta(hours=2)
        gh = fin.groupby("home")["hg"].sum().add(fin.groupby("away")["ag"].sum(), fill_value=0)
        a, b = fin["home"].iloc[-1], fin["away"].iloc[-1]
        champ = a if gh.get(a, 0) > gh.get(b, 0) else b if gh.get(b, 0) > gh.get(a, 0) else None
        if champ is None:
            day = fin["date"].iloc[-1].strftime("%Y-%m-%d")
            champ = ids[sh[(sh["comp"] == comp) & (sh["date"] == day)]["winner"].iloc[0]]
        ladder = cm.KO[comp]
        st_real = np.full(n, -1)
        for g in full.cups[comp].groups:
            st_real[g] = 0
        for s_, rnd in enumerate(ladder, start=1):
            k = dc[dc["slug"] == rnd]
            for t in set(k["home"]) | set(k["away"]):
                if t in T:
                    st_real[T.index(t)] = max(st_real[T.index(t)], s_)
        if comp == cm.SUD:   # los primeros de grupo entran directo a octavos (etapa 2)
            k = dc[dc["slug"] == "round-of-16"]
            for t in set(k["home"]) | set(k["away"]):
                st_real[T.index(t)] = max(st_real[T.index(t)], 2)
        st_real[T.index(champ)] = len(ladder) + (1 if comp == cm.LIB else 1)
        real[comp] = (st_real, T.index(champ))
    lib = d[(d["comp"] == cm.LIB)]
    gs = lib[lib["slug"] == "group-stage"].sort_values("date")
    days = pd.Series(np.sort(gs["date"].dt.normalize().unique()))
    jor = (days.diff().dt.days.fillna(99) > 4).cumsum()
    md_end = days.groupby(jor).max().tolist()
    pts = [("inicio grupos", gs["date"].min() - pd.Timedelta(hours=1))]
    pts += [(f"fecha {k}", md_end[k - 1] + pd.Timedelta(hours=23, minutes=59)) for k in (2, 4) if k <= len(md_end)]
    pts += [("fin grupos", gs["date"].max() + pd.Timedelta(hours=3))]
    for rnd in cm.KO[cm.LIB]:
        k = lib[lib["slug"] == rnd]
        pts.append((f"antes de {rnd}", k["date"].min() - pd.Timedelta(hours=1)))
    for label, asof in pts:
        asof = pd.Timestamp(asof)
        asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof
        st = cm.conmebol_state(m, syear, names, asof=asof, comps=comps)
        for name, (md_, h) in VARIANTS.items():
            strg = strengths_for(m, dom, st, asof, h, max_days=md_)
            res = cm.simulate(st, strg.draw, S=S, seed=0, r=strg.r)
            for comp in comps:
                if asof >= fin_date[comp]:      # esa copa ya terminó (la final de la Sudamericana es antes)
                    continue
                stg = res[comp]["stage"]; st_real, ci = real[comp]
                inc = st_real >= 0
                K = len(cm.STAGES[comp])
                P = np.stack([(stg == k).mean(0) for k in range(K)], 1)
                ls = -np.log(np.clip(P[np.where(inc)[0], st_real[inc]], 1e-4, 1)).mean()
                row = dict(syear=syear, copa=comp, momento=label, modelo=name, log_etapa=float(ls),
                           log_campeon=float(-np.log(max(P[ci, -1], 1e-4))), p_campeon=float(P[ci, -1]),
                           favorito=names[T[int(P[:, -1].argmax())]], p_fav=float(P[:, -1].max()))
                rows.append(row)
                print(row, flush=True)
D = pd.DataFrame(rows)
out = dict(resumen={f"{a}/{b}": v for (a, b), v in D.groupby(["copa", "modelo"])[["log_etapa", "log_campeon"]].mean().to_dict("index").items()})
for metric in ("log_etapa", "log_campeon"):
    piv = D.pivot_table(index=["copa", "syear", "momento"], columns="modelo", values=metric, sort=False)
    dd = (piv["sudamerica"] - piv["solo_copas"]).groupby(level=[0, 1]).mean()
    out[f"dif_{metric}"] = dict(media=float(dd.mean()), se_ediciones=float(dd.std() / np.sqrt(len(dd))),
                                por_edicion={f"{a} {b}": float(v) for (a, b), v in dd.items()})
out["filas"] = rows
(store.ROOT / "outputs" / "analysis" / "conmebol_backtest.json").write_text(json.dumps(out, indent=1, default=str))
print({k: v for k, v in out.items() if k.startswith("dif")})
