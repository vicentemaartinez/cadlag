"""Estadísticas y eventos por partido (ESPN): tiros, tiros al arco, posesión, córners y la
línea de tiempo de goles y tarjetas (minuto exacto en segundos).

Formato crudo de ``events``: ``<tipo><segundo><lado>`` separados por ``;``
  tipo: G gol, P penal convertido, O autogol, R roja, Y amarilla
  lado: h local, a visita, ? desconocido
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .store import RAW

EVENTS = RAW / "espn_events_2010-2027.csv.gz"
_TOK = re.compile(r"^([GPORY])(\d*)([ha?]?)$")


def load_stats() -> pd.DataFrame:
    """Histórico (data/raw) más lo que trae la actualización automática (data/processed/stats.parquet)."""
    from .store import STATS
    d = pd.read_csv(EVENTS, dtype={"id": str, "events": str}, keep_default_na=False)
    if STATS.exists():
        new = pd.read_parquet(STATS)
        new["id"] = new["id"].astype(str); new["events"] = new["events"].fillna("")
        d = pd.concat([d, new], ignore_index=True)
    for c in ["sh_h", "sh_a", "sot_h", "sot_a", "poss_h", "poss_a", "cor_h", "cor_a"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop_duplicates("id", keep="last")


def load_events(stats: pd.DataFrame | None = None, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Formato largo: id, type, t (segundos), side ('h'/'a'), con la convención de autogoles resuelta.

    ESPN atribuye el autogol a un equipo; se verifica contra el marcador final cuál convención
    (equipo que lo hace o equipo que se beneficia) reproduce el resultado, y se normaliza a
    "lado que suma el gol".
    """
    s = load_stats() if stats is None else stats
    rows = []
    for mid, ev in zip(s["id"], s["events"]):
        if not ev:
            continue
        for tok in ev.split(";"):
            m = _TOK.match(tok)
            if not m:
                continue
            typ, sec, side = m.groups()
            rows.append((mid, typ, float(sec) if sec else np.nan, side or "?"))
    e = pd.DataFrame(rows, columns=["id", "type", "t", "side"])
    if matches is not None:
        e = _fix_own_goals(e, matches)
    return e


def _fix_own_goals(e: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    fin = matches[matches["finished"]].set_index("id")[["hg", "ag"]]
    goals = e[e["type"].isin(["G", "P", "O"])].copy()
    # dos convenciones para los autogoles
    flip = {"h": "a", "a": "h", "?": "?"}
    best, best_ok = None, -1
    for conv in ("beneficiary", "scorer"):
        g = goals.copy()
        if conv == "scorer":
            m = g["type"] == "O"
            g.loc[m, "side"] = g.loc[m, "side"].map(flip)
        cnt = g.groupby(["id", "side"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(columns=["h", "a"], fill_value=0).join(fin, how="inner")
        ok = int(((cnt["h"] == cnt["hg"]) & (cnt["a"] == cnt["ag"])).sum())
        if ok > best_ok:
            best, best_ok = conv, ok
    if best == "scorer":
        m = e["type"] == "O"
        e.loc[m, "side"] = e.loc[m, "side"].map(flip)
    e.attrs["own_goal_convention"] = best
    return e
