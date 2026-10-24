"""Espejo público (GitHub) de football-data.co.uk con cuotas Bet365/máximas y Elo de ClubElo.

Fuente: github.com/xgabora/Club-Football-Match-Data-2000-2025 (datos de football-data.co.uk
y clubelo.com). Se usa solo para *evaluar*: comparar el modelo contra el mercado y contra
Elo. Chile no está; Brasil y Argentina llegan hasta 2024.

Los nombres de equipo no coinciden con los de ESPN, así que el cruce se aprende solo:
cada partido de ESPN se empareja con los del espejo del mismo día (±1, por husos horarios)
con el mismo marcador; cada emparejamiento vota "id ESPN -> nombre", y gana la mayoría.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .store import ROOT

MIRROR = ROOT / "data" / "raw" / "footballdata_mirror" / "matches_core.csv.gz"
DIVISION = {"eng.1": "E0", "eng.2": "E1", "esp.1": "SP1", "ita.1": "I1", "ger.1": "D1", "fra.1": "F1",
            "bra.1": "BRA", "arg.1": "ARG"}


def load_mirror() -> pd.DataFrame:
    d = pd.read_csv(MIRROR, low_memory=False)
    d["day"] = pd.to_datetime(d["MatchDate"]).dt.normalize()
    return d


def link(espn: pd.DataFrame, comp: str, mirror: pd.DataFrame | None = None) -> pd.DataFrame:
    """Agrega a los partidos ESPN de ``comp`` las columnas del espejo (cuotas, Elo, tiros)."""
    mirror = load_mirror() if mirror is None else mirror
    mm = mirror[mirror["Division"] == DIVISION[comp]].copy()
    e = espn[(espn["comp"] == comp) & espn["finished"]].copy()
    e["day"] = e["date"].dt.tz_convert(None).dt.normalize()
    # 1) votos por día ±1 y marcador idéntico
    votes = defaultdict(Counter)
    for shift in (-1, 0, 1):
        k = e.assign(day=e["day"] + pd.Timedelta(days=shift))
        j = k.merge(mm, left_on=["day", "hg", "ag"], right_on=["day", "FTHome", "FTAway"])
        for h, a, hn, an in zip(j["home"], j["away"], j["HomeTeam"], j["AwayTeam"]):
            votes[h][hn] += 1
            votes[a][an] += 1
    name = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    e["m_home"] = e["home"].map(name)
    e["m_away"] = e["away"].map(name)
    # 2) cruce exacto por nombres y día ±1
    out = []
    for shift in (0, -1, 1):
        k = e.assign(day=e["day"] + pd.Timedelta(days=shift))
        j = k.merge(mm, left_on=["day", "m_home", "m_away"], right_on=["day", "HomeTeam", "AwayTeam"])
        out.append(j)
    j = pd.concat(out).drop_duplicates("id")
    cols = ["id", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "OddHome", "OddDraw", "OddAway",
            "MaxHome", "MaxDraw", "MaxAway", "HomeElo", "AwayElo", "HomeShots", "AwayShots",
            "HomeTarget", "AwayTarget", "HomeRed", "AwayRed"]
    res = espn[espn["comp"] == comp].merge(j[cols], on="id", how="left")
    ok = res["FTHome"].notna()
    bad = ok & ((res["FTHome"] != res["hg"]) | (res["FTAway"] != res["ag"]))
    res.loc[bad, cols[1:]] = np.nan  # marcadores en conflicto: no se usa ese partido
    return res
