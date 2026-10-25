"""Trae de ESPN el historial completo de ligas nuevas (una vez) y lo suma al almacén.

Pensado para correr en GitHub Actions (``.github/workflows/backfill.yml``), que sí llega a ESPN:
pide cada año calendario completo (``dates=YYYY``) de cada liga, con resultados y tiros al arco,
y lo mezcla con lo que ya hay. Los códigos que ESPN no reconoce se informan y se saltan.

Uso: python scripts/backfill_espn.py [--remote] [--desde 2010] [ligas...]
     → data/processed/{matches,teams,stats}.parquet (y, con --remote, al estado de Supabase)
"""
import argparse
import datetime as dt
import json

from cadlag.data import store

# Ligas de los clubes que juegan copas UEFA y que no modelamos todavía (código de ESPN)
LIGAS = ["por.1", "ned.1", "tur.1", "bel.1", "gre.1", "aut.1", "cze.1", "nor.1", "ukr.1", "svk.1",
         "aze.1", "sco.1", "den.1", "sui.1", "swe.1", "cro.1", "srb.1", "pol.1", "cyp.1", "isr.1",
         "rou.1", "hun.1", "bul.1"]

ap = argparse.ArgumentParser()
ap.add_argument("ligas", nargs="*")
ap.add_argument("--desde", type=int, default=2010)
ap.add_argument("--remote", action="store_true")
a = ap.parse_args()
ligas = a.ligas or LIGAS
if a.remote:
    from cadlag import remote
    remote.pull_state()
years = [str(y) for y in range(a.desde, dt.date.today().year + 2)]
resumen = {}
for comp in ligas:
    try:
        store.update([comp], days_back=0, days_forward=0, months=years)
    except RuntimeError as e:      # código desconocido para ESPN u otro error de red
        resumen[comp] = f"error: {e}"
        print(f"{comp}: {e}", flush=True)
        continue
    m, _ = store.load()
    d = m[m["comp"] == comp]
    por_temp = d.groupby("syear").size().to_dict()
    resumen[comp] = dict(partidos=int(len(d)), terminados=int(d["finished"].sum()),
                         temporadas={int(k): int(v) for k, v in por_temp.items()})
    print(f"{comp}: {len(d)} partidos, temporadas {min(por_temp, default='-')}–{max(por_temp, default='-')}", flush=True)
try:
    from cadlag.data.events import load_stats
    st = load_stats()
    m, _ = store.load()
    ids = set(st["id"].astype(str))
    for comp, r in resumen.items():
        if isinstance(r, dict):
            d = m[(m["comp"] == comp) & m["finished"]]
            r["con_tiros"] = int(d["id"].astype(str).isin(ids).sum())
except Exception as e:  # el resumen de tiros es informativo
    print("sin resumen de tiros:", e)
(store.PROCESSED / "backfill_resumen.json").write_text(json.dumps(resumen, indent=1))
print(json.dumps({k: (v if isinstance(v, str) else {kk: vv for kk, vv in v.items() if kk != "temporadas"})
                  for k, v in resumen.items()}, indent=1))
if a.remote:
    remote.push_state()
