"""Capa continental (``model.europe``): ajuste y prueba fuera de muestra, y ajuste de producción.

1. Con las fuerzas domésticas de los hiperparámetros de backtest, se ajustan los hiperparámetros de
   la capa con las temporadas de entrenamiento y se mide la log-verosimilitud predictiva del resto,
   para tres variantes:
     * ``dom``: equipos de todas las ligas con modelo doméstico de la región, anclados a su fuerza
       doméstica (+ offset de su liga + residuo);
     * ``dom5``: lo mismo, solo con las ligas fuertes (``region.top``: las cinco grandes en Europa,
       Brasil y Argentina en Sudamérica); el resto con estado propio;
     * ``solo_copas``: todos los equipos con estado propio, solo partidos de copa.
   Europa: entrenamiento 2011-12 a 2017-18, prueba desde 2018-19. Sudamérica: entrenamiento 2011 a
   2020, prueba desde 2021.
2. Con las fuerzas domésticas de producción y todos los partidos, ajusta los hiperparámetros que
   usa el pronóstico (``models/europe_h.npy`` o ``models/sudamerica_h.npy``).

Uso: python scripts/europe_fit.py [europe|sudamerica]
     → outputs/<región>/fit_backtest.json, outputs/analysis/<región>_fit.json
"""
import json
import sys

import numpy as np

from cadlag.data import store
from cadlag.model import europe as eu

rg = eu.REGIONS[sys.argv[1] if len(sys.argv) > 1 else "europe"]
TRAIN, TEST = ((2011, 2017), 2018) if rg is eu.EUROPE else ((2011, 2020), 2021)
m, _ = store.load()
res = {}
dom = eu.load_domestic(m, "backtest", refresh=True, region=rg)
eds = {}
for name, md, dd in (("dom", 120, dom), ("dom5", 120, dom[dom["comp"].isin(rg.top)]), ("solo_copas", -1, dom)):
    ed = eu.build(m, dd, max_days=md, region=rg); f = eu.EuroFilter(ed)
    s = ed.matches["season"].to_numpy()
    f.fit(mask=((s >= TRAIN[0]) & (s <= TRAIN[1])).astype(float))
    _, o = f.run(); lp = o["logp"]; w = ed.arr["w"]; te = (s >= TEST) & (w > 0)
    res[name] = dict(h=f.h.tolist(), test=float(-lp[te].mean()), n=int(te.sum()), lp=lp[te].tolist())
    eds[name] = (ed, te)
    print(name, round(res[name]["test"], 4), dict(zip(eu.HYPER_NAMES, np.round(f.h, 3))), flush=True)
d = np.array(res["dom"]["lp"]) - np.array(res["solo_copas"]["lp"])
res["dif"] = dict(media=float(d.mean()), se=float(d.std() / np.sqrt(len(d))))
d5 = np.array(res["dom"]["lp"]) - np.array(res["dom5"]["lp"])
res["dif_vs_dom5"] = dict(media=float(d5.mean()), se=float(d5.std() / np.sqrt(len(d5))))
# por tipo de partido y copa
ed, te = eds["dom"]
x = ed.matches[te]
typ = np.where((x.core_h > 0) & (x.core_a > 0), "liga-liga", np.where((x.core_h > 0) | (x.core_a > 0), "liga-otro", "otro-otro"))
by = {}
for ty in ("liga-liga", "liga-otro", "otro-otro"):
    for c in rg.comps:
        k = (typ == ty) & (x.comp == c).to_numpy()
        if k.sum() > 20:
            by[f"{ty} {c}"] = dict(n=int(k.sum()), media=float(d[k].mean()), se=float(d[k].std() / np.sqrt(k.sum())))
res["por_tipo"] = by
print("dom - solo_copas", res["dif"], "| dom - dom5", res["dif_vs_dom5"], flush=True)
# ¿dónde gana anclar las ligas que no son de las fuertes?
ed5, _ = eds["dom5"]
x5 = ed5.matches[te]
nuevo = ((x5.core_h == 0).to_numpy() & (x.core_h > 0).to_numpy()) | ((x5.core_a == 0).to_numpy() & (x.core_a > 0).to_numpy())
res["vs_dom5_por_tipo"] = {k: dict(n=int(v.sum()), media=float(d5[v].mean()) if v.any() else 0.0,
                                   se=float(d5[v].std() / np.sqrt(max(v.sum(), 1))) if v.any() else 0.0)
                           for k, v in (("con equipo de liga nueva", nuevo), ("sin", ~nuevo))}
print(res["vs_dom5_por_tipo"], flush=True)
rg.cache.mkdir(parents=True, exist_ok=True)
json.dump(res, open(rg.cache / "fit_backtest.json", "w"))
# producción
domp = eu.load_domestic(m, "prod", refresh=True, region=rg)
ed = eu.build(m, domp, region=rg); f = eu.EuroFilter(ed, np.array(res["dom"]["h"]))
s = ed.matches["season"].to_numpy(); f.fit(mask=(s >= TRAIN[0]).astype(float))
np.save(store.ROOT / "models" / rg.hyper_file, f.h)
hp = dict(zip(eu.HYPER_NAMES, [float(v) for v in f.h]))
print("producción", {k: round(v, 3) for k, v in hp.items()}, flush=True)
summary = dict(test=dict(dom=res["dom"]["test"], dom5=res["dom5"]["test"], solo_copas=res["solo_copas"]["test"], n=res["dom"]["n"]),
               dif=res["dif"], dif_vs_dom5=res["dif_vs_dom5"], vs_dom5_por_tipo=res["vs_dom5_por_tipo"], por_tipo=by,
               hiper_produccion=hp, hiper_backtest=dict(zip(eu.HYPER_NAMES, res["dom"]["h"])))
name = "europe_fit.json" if rg is eu.EUROPE else f"{rg.name}_fit.json"
(store.ROOT / "outputs" / "analysis" / name).write_text(json.dumps(summary, indent=1))
