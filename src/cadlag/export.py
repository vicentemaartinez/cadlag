"""Exportación para el sitio web (site/public/data/).

Tres pasos, en orden, después de ``pipeline.forecast`` (que escribe outputs/web/<liga>.json):

1. ``export_sims``: simula de nuevo cada liga y guarda, por temporada simulada, el resultado (0/1/2)
   de los próximos K partidos, sus probabilidades dadas las fuerzas de esa temporada, los puntos y
   la posición final de cada equipo. Con eso el navegador condiciona escenarios ("¿y si gana X?")
   reponderando todas las temporadas, sin re-simular (docs/teoria.md §22).
2. ``export_trajectory``: probabilidad de título después de cada día con partidos (la figura càdlàg).
3. ``export_web``: separa todo en los archivos que lee el sitio. Las probabilidades de posición se
   recalculan desde las mismas simulaciones que usa el laboratorio, para que la matriz no cambie
   cuando el navegador termina de cargarlas.
"""
from __future__ import annotations

import base64
import gzip
import json

import numpy as np
import pandas as pd

from .data import store
from .eval.backtest import preseason_transition
from .leagues import CORE_LEAGUES, LEAGUES
from .model.filter import prepare
from .pipeline import current_season, load_or_fit
from .sim.analysis import outcomes
from .sim.season import season_state, simulate

WEB = store.ROOT / "outputs" / "web"
SITE = store.ROOT / "site" / "public" / "data"


LAYOUT = "o[K] sqrtq[3K] pts[n] pos[n]"   # bytes por temporada simulada (ver docstring de export_sims)
# Argentina: además los puntos del torneo en curso y, en un byte, el puesto en la zona (bits 0-3), si
# sale campeón (bit 4), si desciende (bit 5) y si va a la Libertadores (bit 6) o a la Sudamericana (bit 7).
LAYOUT_SPLIT = LAYOUT + " zpts[n] zinfo[n]"
# copas: además la etapa alcanzada (0 fase liga, 1 playoff, 2 octavos, 3 cuartos, 4 semis, 5 final, 6 campeón)
LAYOUT_CUP = LAYOUT + " stage[n]"


def export_sims(S: int = 20000, K: int = 10, comps=None, seed: int = 7, cups=None) -> dict:
    """Por temporada simulada, en bytes: el resultado sorteado de los próximos K partidos (0/1/2),
    P(L, E, V) de esos partidos dadas las fuerzas de esa temporada (como 255·√p, para tener precisión
    relativa en probabilidades chicas), los puntos finales y la
    posición final de cada equipo. Con P(·|fuerzas) el laboratorio condiciona reponderando todas
    las temporadas: w_s = Π_k P(o_k fijado | x_s), que es exacto porque, dadas las fuerzas, los
    partidos son independientes."""
    from .model.engine import Engine
    m, _ = store.load()
    out = {}
    for comp in (CORE_LEAGUES if comps is None else comps):
        league = LEAGUES[comp]
        f = json.loads((WEB / f"{comp}.json").read_text())
        eng = Engine(m, comp, verbose=False)
        ld, hp = eng.ld, eng.params()
        mT, PT = eng.run()
        tix = {t: i for i, t in enumerate(ld.team_ids)}
        st = season_state(m, comp, current_season(m, comp), tix)
        if len(st.done) == 0:
            mT, PT = eng.preseason(mT, PT, current_season(m, comp))
        k = min(K, len(st.todo))
        r = simulate(league, st, mT, PT, hp, ld.N, n_sims=S, seed=seed, probs_k=k)
        O = outcomes(r)[:, :k].astype(np.uint8)
        Q = np.clip(np.round(np.sqrt(r.probs if k else np.zeros((S, 0, 3))) * 255), 0, 255).astype(np.uint8).reshape(S, 3 * k)
        pos = r.pos.astype(np.uint8)
        if r.promoted is not None:     # bit 7 de la posición: asciende (directo o por playoffs)
            pos = pos | (r.promoted.astype(np.uint8) << 7)
        parts = [O, Q, np.clip(r.pts, 0, 255).astype(np.uint8), pos]
        split = None
        if r.extra is not None:
            X = r.extra
            zinfo = (X["zpos"].astype(np.uint8) | (X["title"].astype(np.uint8) << 4) | (X["rel"].astype(np.uint8) << 5))
            if "lib" in X:   # bits 6 y 7: Libertadores y Sudamericana del año siguiente
                zinfo = zinfo | (X["lib"].astype(np.uint8) << 6) | (X["sud"].astype(np.uint8) << 7)
            parts += [np.clip(X["zpts"], 0, 255).astype(np.uint8), zinfo]
            sp = st.split
            prev = sp["prev"] if sp["prev"] is not None else np.zeros((len(st.teams), 2))
            split = dict(zone=[int(z) for z in X["zone"]], prev_pts=[float(x) for x in prev[:, 0]],
                         prev_games=[float(x) for x in prev[:, 1]], games=[float(x) for x in sp["games"]],
                         promedios=sp["prev"] is not None, tournament=["Apertura", "Clausura"][sp["cur"]],
                         **({"cups": dict(fixed=[int(i) for i in sp["cups"]["fixed"]], lib=sp["cups"]["lib"], sud=sp["cups"]["sud"])}
                            if sp.get("cups") else {}))
        blob = np.concatenate(parts, axis=1).tobytes()
        fixtures = []
        for j in range(k):
            row = st.todo.iloc[j]
            fixtures.append(dict(home=st.teams[int(row.hi)], away=st.teams[int(row.ai)], date=str(row.date)[:16],
                                 p=[float((O[:, j] == c).mean()) for c in range(3)]))
        out[comp] = dict(
            meta={k2: f[k2] for k2 in ("comp", "name", "country", "season_label", "matches_played", "matches_total",
                                        "zones", "suspense", "last_result")},
            order=st.teams, teams=f["teams"], upcoming=f["upcoming"], decisive=f["decisive_matches"],
            model=dict(home_advantage=f["model"]["home_advantage"], goal_level=f["model"]["goal_level"]),
            sims=dict(S=S, K=k, n=len(st.teams), layout=LAYOUT_SPLIT if split else LAYOUT, asc=r.promoted is not None,
                      **({"split": split} if split else {}), fixtures=fixtures, b64=base64.b64encode(gzip.compress(blob, 9)).decode()),
        )
        print(f"  sims {comp}: {S} temporadas, {k} partidos fijables", flush=True)
    from . import cup as cupm
    for comp in (list(cupm.CUPS) if cups is None else cups):
        p = WEB / f"{comp}.json"
        if not p.exists():
            continue
        f = json.loads(p.read_text())
        if comp not in cupm._CACHE:      # `cadlag site` sin pronóstico previo en este proceso
            cupm.forecast_cup(comp, m, store.load()[1], n_sims=S)
        blob, k, order, fixtures = cupm.sims_blob(comp, K)
        Sc = cupm._CACHE[comp][2]["pos"].shape[0]
        out[comp] = dict(
            meta={k2: f[k2] for k2 in ("comp", "name", "country", "season_label", "matches_played", "matches_total",
                                        "zones", "suspense", "last_result")},
            order=order, teams=f["teams"], upcoming=[], decisive=f["decisive_matches"],
            model=dict(home_advantage=f["model"]["home_advantage"], goal_level=f["model"]["goal_level"],
                       offsets=f["model"]["offsets"]),
            sims=dict(S=Sc, K=k, n=len(order), layout=LAYOUT_CUP, cup=True, fixtures=fixtures,
                      b64=base64.b64encode(gzip.compress(blob, 9)).decode()),
        )
        print(f"  sims {comp}: {Sc} ediciones, {k} partidos fijables", flush=True)
    an = store.ROOT / "outputs" / "analysis"
    g = json.loads((WEB / "global.json").read_text())
    graph = [dict(a=a, b=b, w=int(w)) for a, b, w in pd.read_csv(an / "crossleague_graph.csv")[["a", "b", "w"]].itertuples(index=False)]
    summ = json.loads((an / "crossleague_summary.json").read_text())
    out["_global"] = dict(leagues=g["leagues"], ranking=g["ranking"][:40], fiedler=g["fiedler"], graph=graph,
                          eigen=summ["fiedler"]["eigenvalues"][:3])
    bt = {}
    s2p = an / "shots2_vs_market.json"   # dos latentes en los mismos partidos con cuotas
    s2 = json.loads(s2p.read_text()) if s2p.exists() else {}
    for c in CORE_LEAGUES:
        p = store.ROOT / "outputs" / "backtest" / f"{c}_summary.json"
        if p.exists():
            j = json.loads(p.read_text())
            bt[c] = {r["pronostico"]: dict(rps=r["rps"], log=r["log"], n=r["n"]) for r in j["matches"]}
            if c in s2:  # modelo oficial (dos latentes), mismos partidos
                bt[c]["Dos latentes"] = s2[c]["Dos latentes"]
    out["_backtest"] = bt
    (WEB / "sims_data.json").write_text(json.dumps(out, ensure_ascii=False, default=float))
    return out


def export_trajectory(comp: str = "chi.1", syear: int | None = None, S: int = 3000) -> dict:
    from .model.engine import Engine
    m, teams = store.load()
    syear = syear or current_season(m, comp)
    names = dict(zip(teams["team_id"], teams["name"]))
    league = LEAGUES[comp]
    eng = Engine(m, comp, verbose=False)
    ld, mod, hp = eng.ld, eng, eng.params()
    tix = {t: i for i, t in enumerate(ld.team_ids)}
    reg = m[(m["comp"] == comp) & (m["syear"] == syear) & (m["stage"] == "regular") & m["finished"]].sort_values("date")
    if reg.empty:
        return {}
    days = reg["date"].dt.tz_convert("America/Santiago").dt.date
    ends = reg.groupby(days)["date"].max().sort_values()
    plan = [("pre", reg["date"].iloc[0] - pd.Timedelta(seconds=1))] + [("d", t) for t in ends]
    dn = ld.matches["date"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    idx = [int(np.searchsorted(dn, a.tz_convert("UTC").tz_localize(None).to_datetime64(), side="right") - 1) for _, a in plan]
    Ms, Ps, _ = mod.run_checkpoints(idx)
    rows = []
    for (kind, asof), mv, Pm in zip(plan, Ms, Ps):
        if kind == "pre":
            mv, Pm = eng.preseason(mv, Pm, syear)
        st = season_state(m, comp, syear, tix, asof=asof)
        r = simulate(league, st, mv, Pm, hp, ld.N, n_sims=S, seed=11)
        p1 = (r.pos == 0).mean(0)
        rows.append(dict(date=str(asof)[:16], played=int((reg["date"] <= asof).sum()),
                         p={t: round(float(p), 4) for t, p in zip(st.teams, p1)}))
    peak: dict[str, float] = {}
    for r in rows:
        for t, p in r["p"].items():
            peak[t] = max(peak.get(t, 0), p)
    keep = [t for t, p in sorted(peak.items(), key=lambda x: -x[1]) if p >= 0.10][:6]
    total = int(len(m[(m["comp"] == comp) & (m["syear"] == syear) & (m["stage"] == "regular")]))
    out = dict(comp=comp, syear=syear, S=S, total=total, teams=[dict(id=t, name=names.get(t, t)) for t in keep],
               points=[dict(date=r["date"], played=r["played"], p=[r["p"].get(t, 0.0) for t in keep]) for r in rows])
    (WEB / f"trajectory_{comp}.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"  trayectoria {comp} {syear}: {len(rows)} puntos", flush=True)
    return out


def _positions_from_blob(raw: bytes, S: int, K: int, n: int) -> np.ndarray:
    a = np.frombuffer(raw, np.uint8).reshape(S, -1)[:, 4 * K + n:4 * K + 2 * n] & 0x7F
    M = np.zeros((n, n))
    for t in range(n):
        M[t] = np.bincount(a[:, t], minlength=n)[:n] / S
    return M


def export_web() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    data = json.loads((WEB / "sims_data.json").read_text())
    r4 = lambda x: round(float(x), 4)  # noqa: E731
    index = []
    from .cup import CUPS, REACH
    for c in CORE_LEAGUES + [x for x in CUPS if x in data]:
        d = data[c]
        sims = d["sims"]
        gz = base64.b64decode(sims["b64"])
        (SITE / f"{c}.sims.gz").write_bytes(gz)
        raw = gzip.decompress(gz)
        M = _positions_from_blob(raw, sims["S"], sims["K"], sims["n"])
        pos = {t: M[i] for i, t in enumerate(d["order"])}
        asc = None
        if sims.get("asc"):   # P(ascenso) desde las mismas simulaciones que la matriz
            b = np.frombuffer(raw, np.uint8).reshape(sims["S"], -1)[:, 4 * sims["K"] + sims["n"]:]
            asc = {t: float((b[:, i] >> 7).mean()) for i, t in enumerate(d["order"])}
        for t in d["teams"]:
            t["position_probs"] = [r4(p) for p in pos[t["id"]]]
            if asc is not None:
                t["zones"]["promoted"] = r4(asc[t["id"]])
        if sims.get("split"):  # campeón, playoffs y descenso desde las mismas simulaciones
            z = np.frombuffer(raw, np.uint8).reshape(sims["S"], -1)[:, 4 * sims["K"] + 3 * sims["n"]:]
            for i, tid in enumerate(d["order"]):
                t = next(x for x in d["teams"] if x["id"] == tid)
                t["zones"]["title"] = r4(((z[:, i] >> 4) & 1).mean())
                t["zones"]["playoffs"] = r4(((z[:, i] & 0x0F) < 8).mean())
                t["zones"]["relegated"] = r4(((z[:, i] >> 5) & 1).mean())
                if sims["split"].get("cups"):
                    t["zones"]["libertadores"] = r4(((z[:, i] >> 6) & 1).mean())
                    t["zones"]["sudamericana"] = r4(((z[:, i] >> 7) & 1).mean())
        if sims.get("cup"):   # etapas de la copa desde las mismas simulaciones
            n_ = sims["n"]; b = np.frombuffer(raw, np.uint8).reshape(sims["S"], -1)
            stg = b[:, 4 * sims["K"] + 2 * n_:4 * sims["K"] + 3 * n_]
            for i, tid in enumerate(d["order"]):
                t = next(x for x in d["teams"] if x["id"] == tid)
                for key, v in REACH.items():
                    t["zones"][key] = r4((stg[:, i] >= v).mean())
                for z in d["meta"]["zones"]:
                    t["zones"][z["key"]] = r4(sum(t["position_probs"][p - 1] for p in z["positions"]))
                t["stage_probs"] = [r4((stg[:, i] == k).mean()) for k in range(7)]
        league = dict(meta=d["meta"], order=d["order"], teams=d["teams"], upcoming=d["upcoming"], decisive=d["decisive"],
                      model=d["model"], sims={k: v for k, v in sims.items() if k != "b64"})
        (SITE / f"{c}.json").write_text(json.dumps(league, ensure_ascii=False, separators=(",", ":")))
        mt = d["meta"]
        index.append(dict(comp=c, season_label=mt["season_label"], **({"cup": True} if sims.get("cup") else {}), matches_played=mt["matches_played"],
                          matches_total=mt["matches_total"], last_result=mt["last_result"], S=sims["S"],
                          teams=[dict(id=t["id"], name=t["name"], p=t["position_probs"],
                                      **({"asc": t["zones"]["promoted"]} if "promoted" in t["zones"] else {}),
                                      **({"title": t["zones"]["title"]} if "title" in t["zones"] else {}),
                                      **({"champion": t["zones"]["champion"]} if sims.get("cup") else {}))
                                 for t in d["teams"]]))
    (SITE / "index.json").write_text(json.dumps(dict(leagues=index, backtest=data.get("_backtest", {})),
                                                ensure_ascii=False, separators=(",", ":")))
    (SITE / "global.json").write_text(json.dumps(data["_global"], ensure_ascii=False, separators=(",", ":")))
    for p in WEB.glob("trajectory_*.json"):
        (SITE / p.name).write_text(p.read_text())
    print(f"  sitio: {len(list(SITE.iterdir()))} archivos en {SITE.relative_to(store.ROOT)}", flush=True)
