"""Almacén local de partidos: ``data/processed/matches.parquet`` y ``teams.parquet``.

``build_from_raw`` reconstruye todo desde los volcados de ``data/raw/espn``.
``update`` trae de ESPN lo reciente (días alrededor de hoy y los meses que falten) y lo
mezcla con lo que ya hay. Está pensado para correr en un cron (Mac, GitHub Actions, etc.).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import espn

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "espn"
PROCESSED = ROOT / "data" / "processed"
MATCHES = PROCESSED / "matches.parquet"
TEAMS = PROCESSED / "teams.parquet"
STATS = PROCESSED / "stats.parquet"   # estadísticas traídas por la actualización (se suman a data/raw)


def build_from_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    df, teams = espn.read_dump(RAW / "espn_dump_2010-2026.txt.gz")
    fx = RAW / "espn_fixtures_2027.csv.gz"
    if fx.exists():
        df = pd.concat([df, pd.read_csv(fx, dtype=str, keep_default_na=False)], ignore_index=True)
    matches = espn.clean(df)
    t = pd.DataFrame(sorted(teams.items()), columns=["team_id", "name"])
    return matches, t


def save(matches: pd.DataFrame, teams: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(MATCHES, index=False)
    teams.to_parquet(TEAMS, index=False)


MANUAL = ROOT / "data" / "manual" / "results.csv"


def apply_manual(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Resultados cargados a mano (``data/manual/results.csv``) para partidos que ESPN dejó sin
    terminar. Cada fila (comp, date, home, away, hg, ag; equipos por nombre de ESPN) se cruza con el
    partido de esa competición y esos equipos más cercano a la fecha (±4 días); solo se aplica si
    ESPN todavía no lo da por terminado, así que cuando ESPN se pone al día manda ESPN."""
    if not MANUAL.exists():
        return matches
    man = pd.read_csv(MANUAL, comment="#")
    if man.empty:
        return matches
    ids = {}
    for tid, nm in zip(teams["team_id"], teams["name"]):
        ids.setdefault(nm, tid)
    m = matches.copy()
    for r in man.itertuples():
        h, a = ids.get(r.home), ids.get(r.away)
        if h is None or a is None:
            print(f"resultado manual sin equipo: {r.home} – {r.away}")
            continue
        k = m[(m["comp"] == r.comp) & (m["home"] == h) & (m["away"] == a)]
        if not len(k):
            continue
        gap = (k["date"] - pd.Timestamp(r.date, tz="UTC")).abs()
        i = gap.idxmin()
        if gap[i] > pd.Timedelta(days=4) or bool(m.at[i, "finished"]):
            continue
        m.loc[i, ["hg", "ag"]] = [float(r.hg), float(r.ag)]
        m.loc[i, ["finished", "pending", "live"]] = [True, False, False]
        m.loc[i, "status"] = "FULL_TIME"
    return m


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MATCHES.exists():
        m, t = build_from_raw()
        save(m, t)
    t = pd.read_parquet(TEAMS)
    return apply_manual(pd.read_parquet(MATCHES), t), t


def _stale_days(matches: pd.DataFrame, comps: list[str], today, lookback: int = 30, cap: int = 10) -> list[str]:
    """Días de los últimos ``lookback`` con partidos que ya debieron jugarse pero siguen "programados":
    ESPN a veces cierra el resultado después de que la ventana de ``days_back`` ya pasó."""
    d = matches[matches["comp"].isin(comps) & ~matches["finished"]]
    now = pd.Timestamp.now(tz="UTC")
    d = d[(d["date"] < now - pd.Timedelta(hours=3)) & (d["date"] >= pd.Timestamp(today - dt.timedelta(days=lookback), tz="UTC"))]
    days = sorted(set(d["date"].dt.tz_convert("America/New_York").dt.strftime("%Y%m%d")), reverse=True)
    return days[:cap]


def _missing_stat_days(matches: pd.DataFrame, comps: list[str], today, lookback: int = 60, cap: int = 12) -> list[str]:
    """Días (fecha de ESPN, hora del Este de EE. UU.) con partidos terminados de los últimos ``lookback``
    días que todavía no tienen estadísticas (tiros al arco). Así la actualización se completa sola si
    una corrida falló o si se perdieron días; ``cap`` limita cuántos días extra se piden por corrida."""
    try:
        from .events import load_stats
        have = set(load_stats()["id"].astype(str))
    except Exception:
        return []
    d = matches[matches["comp"].isin(comps) & matches["finished"]]
    d = d[d["date"] >= pd.Timestamp(today - dt.timedelta(days=lookback), tz="UTC")]
    d = d[~d["id"].astype(str).isin(have)]
    days = sorted(set(d["date"].dt.tz_convert("America/New_York").dt.strftime("%Y%m%d")), reverse=True)
    return days[:cap]


def update(comps: list[str], days_back: int = 3, days_forward: int = 1,
           months: list[str] | None = None) -> pd.DataFrame:
    """Actualiza el almacén con datos frescos de ESPN.

    - Siempre: día por día en ``[hoy - days_back, hoy + days_forward]`` (resultados recientes).
    - Opcional: meses completos ``YYYYMM`` (para fixtures reprogramados).
    Devuelve los partidos que cambiaron (nuevos resultados), útil para decidir si re-simular.
    """
    import requests

    matches, teams = load()
    today = dt.datetime.now(dt.timezone.utc).date()
    keys = [(today + dt.timedelta(days=k)).strftime("%Y%m%d") for k in range(-days_back, days_forward + 1)]
    keys += months or []
    keys += _missing_stat_days(matches, comps, today)
    keys += _stale_days(matches, comps, today)
    keys = list(dict.fromkeys(keys))
    s = requests.Session()
    frames, names, stats = [], {}, []
    for comp in comps:
        for k in keys:
            rows, n, st = espn.fetch_rows(comp, k, s, with_stats=True)
            frames.append(rows)
            names.update(n)
            stats.append(st)
    if stats:
        new = pd.concat(stats, ignore_index=True)
        if len(new):
            old = pd.read_parquet(STATS) if STATS.exists() else new.iloc[0:0]
            pd.concat([old[~old["id"].isin(new["id"])], new], ignore_index=True).to_parquet(STATS, index=False)
    if not frames:
        return matches.iloc[0:0]
    fresh = espn.clean(pd.concat(frames, ignore_index=True))
    old = matches.set_index("id")
    changed = fresh[~fresh["id"].isin(old.index) |
                    fresh.set_index("id")["finished"].ne(old.reindex(fresh["id"])["finished"].fillna(False)).values]
    merged = pd.concat([matches[~matches["id"].isin(fresh["id"])], fresh], ignore_index=True)
    merged = merged.sort_values(["date", "comp", "id"]).reset_index(drop=True)
    t = pd.concat([teams, pd.DataFrame(sorted(names.items()), columns=["team_id", "name"])])
    t = t.drop_duplicates("team_id", keep="last").sort_values("team_id").reset_index(drop=True)
    save(merged, t)
    return changed


def league_matches(matches: pd.DataFrame, comp: str, stages=("regular",)) -> pd.DataFrame:
    d = matches[(matches["comp"] == comp)]
    if stages is not None:
        d = d[d["stage"].isin(stages)]
    return d.reset_index(drop=True)
