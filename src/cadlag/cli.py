"""Línea de comandos.

    cadlag build                 reconstruye data/processed desde data/raw
    cadlag update [ligas...]     trae resultados recientes de ESPN (necesita internet libre)
    cadlag fit [ligas...]        reajusta hiperparámetros del modelo de cada liga
                                     (models/<liga>_shots2_h.npy; el núcleo en models/<liga>_h.npy)
    cadlag forecast [ligas...]   simula y escribe outputs/web/*.json
    cadlag run                   lo que corre la actualización automática: update y, si hubo
                                     resultados nuevos, forecast + exportación al sitio
                                     (--force para rehacer todo, --no-update para no ir a ESPN)
    cadlag site                  solo la exportación al sitio (sims + trayectoria + site/public/data)
    cadlag run --remote          igual que run, pero lee y guarda el estado en Supabase Storage,
                                     sube los datos del sitio y avisa a Vercel (GitHub Actions)
    cadlag remote-init           crea los buckets en Supabase (una vez)
"""
from __future__ import annotations

import argparse

from .leagues import CORE_LEAGUES, CUP_COMPETITIONS, EUROPE_EXTRA


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cadlag")
    ap.add_argument("cmd", choices=["build", "update", "fit", "forecast", "run", "site", "remote-init"])
    ap.add_argument("comps", nargs="*")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--days-back", type=int, default=3)
    ap.add_argument("--months", nargs="*", default=None, help="meses YYYYMM a re-sincronizar")
    ap.add_argument("--force", action="store_true", help="re-simular aunque no haya resultados nuevos")
    ap.add_argument("--no-update", action="store_true", help="no consultar ESPN")
    ap.add_argument("--site-sims", type=int, default=20000, help="temporadas simuladas para el laboratorio")
    ap.add_argument("--remote", action="store_true", help="estado y datos del sitio en Supabase Storage")
    a = ap.parse_args(argv)
    from .cup import CUPS
    # ligas y copas pronosticadas: sin argumentos, todas; con argumentos, las nombradas
    cups = [c for c in a.comps if c in CUPS] if a.comps else list(CUPS)
    comps = [c for c in a.comps if c not in CUPS] if a.comps else CORE_LEAGUES
    a.cups = cups

    if a.cmd == "build":
        from .data import store
        m, t = store.build_from_raw()
        store.save(m, t)
        print(f"{len(m)} partidos, {len(t)} equipos")
    elif a.cmd == "update":
        from .data import store
        ch = store.update(a.comps or (CORE_LEAGUES + CUP_COMPETITIONS + EUROPE_EXTRA), days_back=a.days_back, months=a.months)
        print(f"{len(ch)} partidos nuevos o actualizados")
    elif a.cmd == "run":
        run(comps, a)
    elif a.cmd == "site":
        site(comps, a, trajectory=True)
    elif a.cmd == "remote-init":
        from . import remote
        remote.init()
    elif a.cmd == "fit":
        from .data import store
        from .model.engine import Engine
        m, _ = store.load()
        for c in comps:
            Engine(m, c, refit=True)
    elif a.cmd == "forecast":
        from .pipeline import forecast
        forecast(comps, n_sims=a.sims, cups=cups)


def site(comps, a, trajectory=True):
    from . import export
    print("exportando al sitio", flush=True)
    export.export_sims(S=a.site_sims, comps=comps, cups=a.cups)
    if trajectory:
        export.export_trajectory("chi.1")
    export.export_web()


def run(comps, a):
    """Actualización automática: si no hay resultados nuevos en las ligas núcleo, no hace nada."""
    import time
    from .data import store
    t0 = time.time()
    if a.remote:
        from . import remote
        remote.pull_state()
    changed_comps: set[str] = set()
    if not a.no_update:
        # las ligas de EUROPE_EXTRA solo alimentan la capa europea de las copas: se actualizan, no se pronostican
        ch = store.update(comps + CUP_COMPETITIONS + EUROPE_EXTRA, days_back=a.days_back, months=a.months)
        changed_comps = set(ch["comp"]) & (set(comps) | set(a.cups)) if len(ch) else set()
        print(f"{len(ch)} partidos nuevos o actualizados; ligas y copas afectadas: {sorted(changed_comps) or 'ninguna'}", flush=True)
        if a.remote and len(ch):
            remote.push_state()  # también los cambios de copas o fixtures, aunque no se re-simule
    if not changed_comps and not a.force:
        print("sin cambios: no se re-simula")
        return
    from .pipeline import forecast
    forecast(comps, n_sims=a.sims, cups=a.cups)
    site(comps, a, trajectory=a.force or "chi.1" in changed_comps)
    if a.remote:
        remote.push_site()
        remote.redeploy()
    print(f"listo en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
