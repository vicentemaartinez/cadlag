"""Adaptador para la API pública (no oficial) de marcadores de ESPN.

Es la misma fuente que usa *marcador*: gratis, sin clave, con resultados minutos después
de cada partido y, sobre todo, con **identificadores de equipo globales**: el Colo-Colo de
la liga chilena es el mismo ``id`` que el de la Libertadores. Eso es lo que hace posible el
grafo de partidos entre ligas.

Lo que acepta ESPN (verificado en octubre de 2026):
  - ``dates=YYYY&limit=1000``   un año calendario completo (incluye partidos por jugar)
  - ``dates=YYYYMM&limit=1000`` un mes
  - ``dates=YYYYMMDD``          un día
  - ``dates=YYYYMMDD-YYYYMMDD`` NO: responde 400

Al no ser oficial puede cambiar sin aviso: todo sale normalizado por ``parse_event`` y la
limpieza (``clean``) está cubierta por pruebas.
"""
from __future__ import annotations

import gzip
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PATH = "/apis/site/v2/sports/soccer"
# Desde algunos servidores site.api.espn.com responde 403 y site.web.api.espn.com funciona.
HOSTS = ["https://site.web.api.espn.com", "https://site.api.espn.com"]
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "es-CL,es;q=0.9,en;q=0.8",
    "user-agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"),
    "referer": "https://www.espn.cl/",
    "origin": "https://www.espn.cl",
}

COLUMNS = ["comp", "id", "date", "home", "away", "hg", "ag", "status", "slug", "syear",
           "neutral", "hls", "als"]
FINISHED = {"FULL_TIME", "FINAL", "FINAL_PEN", "FINAL_AET", "FINAL_AGT"}
EXTRA_TIME = {"FINAL_PEN", "FINAL_AET", "FINAL_AGT"}
PENDING = {"SCHEDULED", "POSTPONED", "TBD", "DELAYED"}
_PLAYOFF_RE = re.compile(
    r"final|semi|quarter|round-of|playoff|play-off|knockout|liguilla|relegation|promotion|"
    r"repechage|runners-up|copa-libertadores-playoffs|octavos|cuartos", re.I)


# --------------------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------------------
def fetch_scoreboard(comp: str, dates: str, session=None, retries: int = 3) -> dict:
    """Descarga un scoreboard. ``dates`` es 'YYYY', 'YYYYMM' o 'YYYYMMDD'."""
    import requests

    s = session or requests.Session()
    last = None
    for host in HOSTS:
        url = f"{host}{BASE_PATH}/{comp}/scoreboard"
        params = {"dates": dates}
        if len(dates) < 8:
            params["limit"] = 1000
        for k in range(retries):
            try:
                r = s.get(url, params=params, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code} en {url}"
            except Exception as exc:  # red intermitente
                last = repr(exc)
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"ESPN no respondió para {comp} {dates}: {last}")


def parse_event(comp: str, e: dict) -> tuple[list, dict]:
    """Un evento de ESPN -> una fila con las columnas de ``COLUMNS`` y los nombres de equipo."""
    c = e["competitions"][0]
    h = next(x for x in c["competitors"] if x["homeAway"] == "home")
    a = next(x for x in c["competitors"] if x["homeAway"] == "away")
    st = (e.get("status") or {}).get("type") or {}
    fin = st.get("state") == "post" and st.get("completed")
    season = e.get("season") or {}
    row = [comp, e["id"], e["date"], h["team"]["id"], a["team"]["id"],
           h.get("score", "") if fin else "", a.get("score", "") if fin else "",
           (st.get("name") or "").replace("STATUS_", ""), season.get("slug", ""),
           season.get("year", ""), 1 if c.get("neutralSite") else 0,
           "|".join(str(x.get("value")) for x in (h.get("linescores") or [])),
           "|".join(str(x.get("value")) for x in (a.get("linescores") or []))]
    names = {h["team"]["id"]: h["team"]["displayName"], a["team"]["id"]: a["team"]["displayName"]}
    return row, names


STAT_COLUMNS = ["id", "sh_h", "sh_a", "sot_h", "sot_a", "poss_h", "poss_a", "cor_h", "cor_a", "events"]
_STAT_NAMES = {"sh": ("totalShots", "shotsTotal", "shots"), "sot": ("shotsOnTarget", "onTargetShots", "shotsOnGoal"),
               "poss": ("possessionPct", "possession"), "cor": ("wonCorners", "cornerKicks", "corners")}


def _stat(comp: dict, key: str):
    st = {x.get("name"): x.get("displayValue", x.get("value")) for x in (comp.get("statistics") or [])}
    for nm in _STAT_NAMES[key]:
        v = st.get(nm)
        if v not in (None, ""):
            try:
                return float(str(v).replace("%", ""))
            except ValueError:
                return np.nan
    return np.nan


def parse_stats(e: dict) -> dict | None:
    """Tiros, tiros al arco, posesión, córners y la línea de tiempo de goles y tarjetas, en el
    mismo formato que ``data/raw/espn/espn_events_*.csv.gz`` (ver ``data.events``)."""
    c = e["competitions"][0]
    st = ((e.get("status") or {}).get("type") or {})
    if not (st.get("state") == "post" and st.get("completed")):
        return None
    h = next(x for x in c["competitors"] if x["homeAway"] == "home")
    a = next(x for x in c["competitors"] if x["homeAway"] == "away")
    toks = []
    for d in c.get("details") or []:
        typ = "P" if d.get("penaltyKick") and d.get("scoringPlay") else "O" if d.get("ownGoal") else \
              "G" if d.get("scoringPlay") else "R" if d.get("redCard") else "Y" if d.get("yellowCard") else None
        if typ is None:
            continue
        sec = (d.get("clock") or {}).get("value")
        tid = (d.get("team") or {}).get("id")
        side = "h" if tid == h["team"]["id"] else "a" if tid == a["team"]["id"] else "?"
        toks.append(f"{typ}{int(sec) if sec is not None else ''}{side}")
    return dict(id=str(e["id"]), sh_h=_stat(h, "sh"), sh_a=_stat(a, "sh"), sot_h=_stat(h, "sot"), sot_a=_stat(a, "sot"),
                poss_h=_stat(h, "poss"), poss_a=_stat(a, "poss"), cor_h=_stat(h, "cor"), cor_a=_stat(a, "cor"),
                events=";".join(toks))


def fetch_rows(comp: str, dates: str, session=None, with_stats: bool = False):
    js = fetch_scoreboard(comp, dates, session)
    rows, names, stats = [], {}, []
    for e in js.get("events", []) or []:
        try:
            r, n = parse_event(comp, e)
        except (StopIteration, KeyError, IndexError):
            continue
        rows.append(r)
        names.update(n)
        if with_stats:
            try:
                sr = parse_stats(e)
                if sr:
                    stats.append(sr)
            except (StopIteration, KeyError, IndexError, TypeError):
                pass
    df = pd.DataFrame(rows, columns=COLUMNS).astype(str)
    if with_stats:
        return df, names, pd.DataFrame(stats, columns=STAT_COLUMNS)
    return df, names


# --------------------------------------------------------------------------------------
# Lectura de los volcados crudos (data/raw/espn)
# --------------------------------------------------------------------------------------
def read_dump(path: Path) -> tuple[pd.DataFrame, dict]:
    """Lee el volcado ``@@comp`` / ``@@TEAMS`` generado al descargar el histórico."""
    raw = gzip.open(path, "rt", encoding="utf-8").read()
    rows, teams = [], {}
    for block in raw.split("@@")[1:]:
        name, _, body = block.partition("\n")
        if name == "TEAMS":
            for line in body.strip().split("\n"):
                k, v = line.split("\t", 1)
                teams[k] = v
            continue
        for line in body.strip().split("\n"):
            if line:
                parts = line.split(",")
                if len(parts) == len(COLUMNS) - 1:
                    rows.append([name] + parts)
    return pd.DataFrame(rows, columns=COLUMNS), teams


def read_csv_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# --------------------------------------------------------------------------------------
# Limpieza y normalización
# --------------------------------------------------------------------------------------
def _goals_90(total: str, ls: str) -> float:
    """Goles a los 90' a partir de los parciales por tiempo (si los hay)."""
    parts = [p for p in ls.split("|") if p not in ("", "None")]
    if len(parts) >= 2:
        try:
            return float(parts[0]) + float(parts[1])
        except ValueError:
            pass
    return float(total) if total not in ("", "None") else np.nan


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos, quita duplicados y eventos mal etiquetados, marca fases.

    Reglas (cada una documentada porque la API no es oficial):
      1. Duplicados por ``id``: se queda la última versión.
      2. Si el ``slug`` de la temporada contiene un año y no coincide con ``syear``, el
         evento es un duplicado fantasma (ESPN repite partidos de 2009-10 con slug 2013-14).
      3. El marcador oficial para el modelo es el de los 90 minutos: en partidos con alargue
         se usan los parciales del primer y segundo tiempo.
    """
    d = df.copy()
    d = d.drop_duplicates("id", keep="last")
    first_year = d["slug"].str.extract(r"(20\d\d)", expand=False)
    ghost = first_year.notna() & (d["syear"] != "") & (first_year != d["syear"])
    # En ligas de año calendario el slug puede decir 2020 para una temporada que ESPN cierra en 2021.
    ghost &= ~d["slug"].str.contains("primera-division-de-chile|transicion|clausura|apertura", case=False)
    d = d[~ghost]

    d["date"] = pd.to_datetime(d["date"], utc=True)
    d["finished"] = d["status"].isin(FINISHED)
    d["extra_time"] = d["status"].isin(EXTRA_TIME)
    d["pending"] = d["status"].isin(PENDING)
    hg = pd.to_numeric(d["hg"], errors="coerce")
    ag = pd.to_numeric(d["ag"], errors="coerce")
    hg90 = np.where(d["extra_time"], [_goals_90(t, l) for t, l in zip(d["hg"], d["hls"])], hg)
    ag90 = np.where(d["extra_time"], [_goals_90(t, l) for t, l in zip(d["ag"], d["als"])], ag)
    d["hg"] = np.where(d["finished"], hg90, np.nan)
    d["ag"] = np.where(d["finished"], ag90, np.nan)
    # 4. Partidos cancelados/abandonados no cuentan; un postergado que luego se jugó aparece
    #    como dos eventos con el mismo cruce: se queda el que terminó.
    d = d[~d["status"].isin({"CANCELED", "CANCELLED", "ABANDONED"})]
    pair = [d["comp"], d["slug"], d["home"], d["away"]]
    any_fin = d.groupby(pair)["finished"].transform("any")
    d = d[d["finished"] | ~any_fin]
    d["live"] = ~d["finished"] & ~d["pending"]
    d["neutral"] = d["neutral"].astype(str).isin(["1", "True", "true"])
    d["syear"] = pd.to_numeric(d["syear"], errors="coerce").astype("Int64")
    d["stage"] = np.where(d["slug"].str.contains(_PLAYOFF_RE), "playoff", "regular")
    d = d.drop(columns=["hls", "als"])
    return d.sort_values(["date", "comp", "id"]).reset_index(drop=True)


def season_key(comp: str, syear: int, style: str) -> str:
    if style == "calendar":
        return str(syear)
    return f"{syear}-{str(syear + 1)[2:]}"
