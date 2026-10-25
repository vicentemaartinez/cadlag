"""Capa europea (``model.europe``): ajuste y prueba fuera de muestra, y ajuste de producción.

1. Con las fuerzas domésticas de los hiperparámetros de backtest (temporadas anteriores a 2016), se
   ajustan los hiperparámetros de la capa con los partidos UEFA de 2011-12 a 2017-18 y se mide la
   log-verosimilitud predictiva de los de 2018-19 en adelante, para dos variantes:
     * ``dom``: equipos de las cinco grandes ligas anclados a su fuerza doméstica (+ offset + residuo);
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
dom = eu.load_domestic(m, "backtest")
eds = {}
for name, md in (("dom", 120), ("solo_copas", -1)):
    ed = eu.build(m, dom, max_days=md); f = eu.EuroFilter(ed)
    s = ed.matches["season"].to_numpy()
    f.fit(mask=((s >= 2011) & (s <= 2017)).astype(float))
    _, o = f.run(); lp = o["logp"]; w = ed.arr["w"]; te = (s >= 2018) & (w > 0)
    res[name] = dict(h=f.h.tolist(), test=float(-lp[te].mean()), n=int(te.sum()), lp=lp[te].tolist())
    eds[name] = (ed, te)
    print(name, round(res[name]["test"], 4), dict(zip(eu.HYPER_NAMES, np.round(f.h, 3))), flush=True)
d = np.array(res["dom"]["lp"]) - np.array(res["solo_copas"]["lp"])
res["dif"] = dict(media=float(d.mean()), se=float(d.std() / np.sqrt(len(d))))
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
print("dom - solo_copas", res["dif"], flush=True)
(store.ROOT / "outputs" / "europe").mkdir(parents=True, exist_ok=True)
json.dump(res, open(store.ROOT / "outputs" / "europe" / "fit_backtest.json", "w"))
# producción
domp = eu.load_domestic(m, "prod")
ed = eu.build(m, domp); f = eu.EuroFilter(ed, np.array(res["dom"]["h"]))
s = ed.matches["season"].to_numpy(); f.fit(mask=(s >= 2011).astype(float))
np.save(store.ROOT / "models" / "europe_h.npy", f.h)
hp = dict(zip(eu.HYPER_NAMES, [float(v) for v in f.h]))
print("producción", {k: round(v, 3) for k, v in hp.items()}, flush=True)
summary = dict(test=dict(dom=res["dom"]["test"], solo_copas=res["solo_copas"]["test"], n=res["dom"]["n"]),
               dif=res["dif"], por_tipo=by, hiper_produccion=hp,
               hiper_backtest=dict(zip(eu.HYPER_NAMES, res["dom"]["h"])))
(store.ROOT / "outputs" / "analysis" / "europe_fit.json").write_text(json.dumps(summary, indent=1))
