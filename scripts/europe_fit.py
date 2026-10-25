"""Capa europea (``model.europe``): ajuste y prueba fuera de muestra, y ajuste de producción.

1. Con las fuerzas domésticas de los hiperparámetros de backtest (temporadas anteriores a 2016), se
   ajustan los hiperparámetros de la capa con los partidos UEFA de 2011-12 a 2017-18 y se mide la
   log-verosimilitud predictiva de los de 2018-19 en adelante, para tres variantes:
     * ``dom``: equipos de las 16 ligas con modelo doméstico (las cinco grandes y ``EUROPE_EXTRA``)
       anclados a su fuerza doméstica (+ offset de su liga + residuo);
     * ``dom5``: lo mismo, solo con las cinco grandes ligas (el resto con estado propio);
     * ``solo_copas``: todos los equipos con estado propio, solo partidos de copa.
2. Con las fuerzas domésticas de producción y todos los partidos, ajusta los hiperparámetros que
   usa el pronóstico (``models/europe_h.npy``).

Uso: python scripts/europe_fit.py   → outputs/europe/fit_backtest.json, outputs/analysis/europe_fit.json
"""
import json

import numpy as np

from cadlag.data import store
from cadlag.model import europe as eu

m, _ = store.load()
res = {}
dom = eu.load_domestic(m, "backtest", refresh=True)
eds = {}
for name, md, dd in (("dom", 120, dom), ("dom5", 120, dom[dom["comp"].isin(eu.BIG5)]), ("solo_copas", -1, dom)):
    ed = eu.build(m, dd, max_days=md); f = eu.EuroFilter(ed)
    s = ed.matches["season"].to_numpy()
    f.fit(mask=((s >= 2011) & (s <= 2017)).astype(float))
    _, o = f.run(); lp = o["logp"]; w = ed.arr["w"]; te = (s >= 2018) & (w > 0)
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
    for c in eu.UEFA:
        k = (typ == ty) & (x.comp == c).to_numpy()
        if k.sum() > 20:
            by[f"{ty} {c}"] = dict(n=int(k.sum()), media=float(d[k].mean()), se=float(d[k].std() / np.sqrt(k.sum())))
res["por_tipo"] = by
print("dom - solo_copas", res["dif"], "| dom - dom5", res["dif_vs_dom5"], flush=True)
# por tipo según la variante de 5 ligas: ¿dónde gana anclar las ligas nuevas?
ed5, _ = eds["dom5"]
x5 = ed5.matches[te]
nuevo = ((x5.core_h == 0).to_numpy() & (x.core_h > 0).to_numpy()) | ((x5.core_a == 0).to_numpy() & (x.core_a > 0).to_numpy())
res["vs_dom5_por_tipo"] = {k: dict(n=int(v.sum()), media=float(d5[v].mean()), se=float(d5[v].std() / np.sqrt(max(v.sum(), 1))))
                           for k, v in (("con equipo de liga nueva", nuevo), ("sin", ~nuevo))}
print(res["vs_dom5_por_tipo"], flush=True)
(store.ROOT / "outputs" / "europe").mkdir(parents=True, exist_ok=True)
json.dump(res, open(store.ROOT / "outputs" / "europe" / "fit_backtest.json", "w"))
# producción
domp = eu.load_domestic(m, "prod", refresh=True)
ed = eu.build(m, domp); f = eu.EuroFilter(ed, np.array(res["dom"]["h"]))
s = ed.matches["season"].to_numpy(); f.fit(mask=(s >= 2011).astype(float))
np.save(store.ROOT / "models" / "europe_h.npy", f.h)
hp = dict(zip(eu.HYPER_NAMES, [float(v) for v in f.h]))
print("producción", {k: round(v, 3) for k, v in hp.items()}, flush=True)
summary = dict(test=dict(dom=res["dom"]["test"], dom5=res["dom5"]["test"], solo_copas=res["solo_copas"]["test"], n=res["dom"]["n"]),
               dif=res["dif"], dif_vs_dom5=res["dif_vs_dom5"], vs_dom5_por_tipo=res["vs_dom5_por_tipo"], por_tipo=by, hiper_produccion=hp,
               hiper_backtest=dict(zip(eu.HYPER_NAMES, res["dom"]["h"])))
(store.ROOT / "outputs" / "analysis" / "europe_fit.json").write_text(json.dumps(summary, indent=1))
