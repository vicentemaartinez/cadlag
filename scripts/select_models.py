"""Confirma el modelo de cada liga a partir de los backtests pareados fuera de muestra.

El modelo oficial es el de dos latentes (docs/teoria.md §21). Una liga vuelve al núcleo solo si
hay evidencia de que el de dos latentes pierde: dos errores estándar en contra en la
log-verosimilitud del marcador (scripts/shots2_analysis.py) o en la de la posición final
(scripts/season_compare.py). Escribe models/model_selection.json, que lee ``cadlag.model.engine``.

Uso: python scripts/select_models.py
"""
import json

from cadlag.data import store
from cadlag.leagues import CORE_LEAGUES

an = store.ROOT / "outputs" / "analysis"
mt = json.loads((an / "shots2_backtest.json").read_text())
sc_path = an / "season_compare.json"
sc = json.loads(sc_path.read_text()) if sc_path.exists() else {}
sel = {}
for comp in CORE_LEAGUES:
    r, s = mt.get(comp), sc.get(comp)
    e = {}
    loses = False
    if r:
        z = r["delta_logp"] / r["delta_se"]
        e.update(partidos=dict(delta_logp=round(r["delta_logp"], 5), se=round(r["delta_se"], 5), z=round(z, 2),
                               rps_dc=round(r["rps_base"], 5), rps_shots2=round(r["rps_two"], 5), n=r["n"], desde=r["test_from"]))
        loses |= z <= -2
    if s:
        z = s["delta_log_pos"] / s["se"]
        e.update(temporadas=dict(delta_log_pos=round(s["delta_log_pos"], 5), se=round(s["se"], 5), z=round(z, 2),
                                 n_temporadas=s["n_seasons"]))
        loses |= z <= -2
    sel[comp] = dict(kind="dc" if loses else "shots2", **e)
    p, t = e.get("partidos", {}), e.get("temporadas", {})
    print(f"{comp}: {sel[comp]['kind']:6s}  partidos z={p.get('z', float('nan')):+.1f}  temporadas z={t.get('z', float('nan')):+.1f}")
out = store.ROOT / "models" / "model_selection.json"
out.write_text(json.dumps(sel, indent=1))
print("→", out.relative_to(store.ROOT))
