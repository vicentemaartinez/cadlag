"""Traspasos de Transfermarkt (github.com/eordo/transfermarkt-data, ligas top 5, 2009-10 a 2025-26).

Cada fila es un movimiento de un club de las cinco grandes ligas: ``movement`` = in (llega desde
``dealing_club``) u out (se va a ``dealing_club``), con valor de mercado del jugador y tarifa.

``load()`` devuelve los traspasos sin duplicados (un traspaso entre dos clubes top 5 aparece en
ambos archivos) y con el club y la liga de cada extremo ubicados en los ids de ESPN, cuando se
puede: el club propio por nombre dentro de su liga y temporada, y el otro club dentro de las ligas
de su país que tenemos (primera y segunda división, Brasil, Argentina, Chile).
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
import pandas as pd

from . import store

RAW = store.ROOT / "data" / "raw" / "transfermarkt" / "transfers_top5_2009-2025.csv.gz"
SOURCE = "https://raw.githubusercontent.com/eordo/transfermarkt-data/HEAD/{folder}/{year}.csv"
FOLDERS = ["bundesliga", "laliga", "ligue_1", "premier_league", "serie_a"]


def download(years=range(2009, 2026)) -> None:
    """Arma el archivo local desde github.com/eordo/transfermarkt-data. No se publica en el
    repositorio: son datos de Transfermarkt sin licencia de redistribución."""
    import io

    import requests
    frames = []
    for f in FOLDERS:
        for y in years:
            r = requests.get(SOURCE.format(folder=f, year=y), timeout=60)
            if r.ok:
                frames.append(pd.read_csv(io.StringIO(r.text)))
    RAW.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(RAW, index=False, compression="gzip")
LEAGUE_COMP = {"Premier League": "eng.1", "Laliga": "esp.1", "Serie A": "ita.1", "Bundesliga": "ger.1", "Ligue 1": "fra.1"}
COUNTRY_COMPS = {"England": ["eng.1", "eng.2"], "Spain": ["esp.1", "esp.2"], "Italy": ["ita.1", "ita.2"],
                 "Germany": ["ger.1", "ger.2"], "France": ["fra.1", "fra.2"], "Brazil": ["bra.1", "bra.2"],
                 "Argentina": ["arg.1"], "Chile": ["chi.1"]}
CALENDAR = {"bra.1", "bra.2", "arg.1", "chi.1"}
_DROP = {"fc", "cf", "ac", "as", "sc", "afc", "club", "de", "del", "la", "cd", "ud", "sd", "rc", "rcd", "ss", "us",
         "ssc", "ogc", "sco", "fk", "sv", "vfl", "vfb", "tsg", "bv", "bsc", "fsv", "1", "calcio", "futbol", "football",
         "clube", "esporte", "regatas", "ca", "cr", "ec", "se", "real", "athletic", "atletico", "sporting", "stade",
         "olympique", "racing", "the", "and", "hotspur", "albion", "wanderers", "town", "city", "united", "county"}


_ALIAS = {"1.FC Köln": "Cologne", "Stade Brestois 29": "Brest", "Athletic Bilbao": "Athletic Club"}


def norm(s: str) -> str:
    s = _ALIAS.get(str(s), str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("ü", "u"))
    return " ".join(w for w in s.split() if w not in _DROP) or s.strip()


def _match(name: str, cands: dict[str, str], cutoff=0.72):
    """cands: nombre normalizado -> id. Devuelve el id más parecido o None."""
    n = norm(name)
    if n in cands:
        return cands[n]
    for k, v in cands.items():  # contención de tokens (p. ej. "inter" ⊂ "internazionale")
        a, b = set(n.split()), set(k.split())
        if a and b and (a <= b or b <= a) and min(len(n), len(k)) >= 4:
            return v
    best = difflib.get_close_matches(n, list(cands), n=1, cutoff=cutoff)
    return cands[best[0]] if best else None


def _season_key(comp: str, season: int, window: str) -> int:
    """Temporada de la liga en que ocurre el traspaso (las sudamericanas van por año calendario)."""
    if comp in CALENDAR:
        return season + 1 if window == "winter" else season
    return season


def load(matches: pd.DataFrame | None = None, teams: pd.DataFrame | None = None, loans: bool = False,
         dedupe: bool = True) -> pd.DataFrame:
    """``dedupe=False`` conserva la fila de cada club (para balances por club: la venta de un club top 5
    a otro club top 5 cuenta como salida para uno y llegada para el otro)."""
    if matches is None:
        matches, teams = store.load()
    names = dict(zip(teams["team_id"], teams["name"]))
    roster = {}  # (comp, syear) -> {nombre normalizado: id}
    for (c, s), g in matches[matches["stage"] == "regular"].groupby(["comp", "syear"]):
        ids = set(g["home"]) | set(g["away"])
        roster[(c, int(s))] = {norm(names.get(t, t)): t for t in ids}
    if not RAW.exists():
        download()
    d = pd.read_csv(RAW)
    if not loans:  # para la jerarquía entre ligas: los préstamos bajan por desarrollo, no por nivel
        d = d[d["is_loan"] == 0]
    d = d.copy()
    d["comp"] = d["league"].map(LEAGUE_COMP)
    own = [_match(r.club, roster.get((r.comp, int(r.season)), {})) for r in d.itertuples()]
    d["club_id"] = own
    other_comp, other_id = [], []
    for r in d.itertuples():
        found = (None, None)
        for c in COUNTRY_COMPS.get(r.dealing_country, []):
            s = _season_key(c, int(r.season), r.window)
            t = _match(r.dealing_club, roster.get((c, s), {}))
            if t is not None:
                found = (c, t); break
        other_comp.append(found[0]); other_id.append(found[1])
    d["other_comp"], d["other_id"] = other_comp, other_id
    d["from_comp"] = np.where(d["movement"] == "in", d["other_comp"], d["comp"])
    d["to_comp"] = np.where(d["movement"] == "in", d["comp"], d["other_comp"])
    d["from_id"] = np.where(d["movement"] == "in", d["other_id"], d["club_id"])
    d["to_id"] = np.where(d["movement"] == "in", d["club_id"], d["other_id"])
    d["from_season"] = [(_season_key(c, int(s), w) if isinstance(c, str) else np.nan) for c, s, w in zip(d["from_comp"], d["season"], d["window"])]
    d["to_season"] = [(_season_key(c, int(s), w) if isinstance(c, str) else np.nan) for c, s, w in zip(d["to_comp"], d["season"], d["window"])]
    # un traspaso entre dos clubes top 5 aparece dos veces (una por club)
    if dedupe:
        d = d.sort_values("movement").drop_duplicates(["player_id", "season", "window", "from_comp", "to_comp"])
    return d.reset_index(drop=True)
