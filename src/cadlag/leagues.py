"""Reglas de cada liga: formato, zonas de la tabla y orden de desempate.

Los criterios de desempate se codifican como una lista ordenada de códigos que el
simulador aplica de forma recursiva (ver ``cadlag.sim.tiebreak``):

    PTS      puntos
    GD       diferencia de gol
    GF       goles a favor
    W        partidos ganados
    AGF      goles a favor de visita
    H2H_PTS  puntos en la minitabla entre los empatados
    H2H_GD   diferencia de gol en la minitabla
    H2H_GF   goles a favor en la minitabla
    H2H_AGF  goles de visita en la minitabla
    LOTS     sorteo

Fuentes de las reglas: ver la sección "simulación de temporada" del informe de research
(``docs/research.md``). Los puntos sin verificar contra el reglamento oficial están marcados.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CRITERIA = ["PTS", "GD", "GF", "W", "AGF", "H2H_PTS", "H2H_GD", "H2H_GF", "H2H_AGF", "LOTS"]
CRIT_CODE = {c: i for i, c in enumerate(CRITERIA)}


@dataclass(frozen=True)
class Zone:
    key: str          # identificador estable (para la web)
    label: str        # etiqueta en español
    positions: tuple  # posiciones (1-indexadas) que cubre la zona


@dataclass(frozen=True)
class League:
    code: str                    # código interno = código de ESPN (eng.1, chi.1, ...)
    name: str
    country: str
    n_teams: int
    season_style: str            # "split" (2025-26) o "calendar" (2026)
    tiebreak: tuple              # criterios en orden
    h2h_reapply: bool = False    # si un subgrupo sigue empatado, ¿se recalcula la minitabla solo con él?
    title_playoff: bool = False  # empate de dos en puntos por el título -> partido de definición
    relegation_playoff_pos: int | None = None  # empate de dos en puntos en el límite del descenso -> definición (Serie A)
    zones: tuple = field(default_factory=tuple)
    mirror_division: str | None = None  # código de división en el espejo de football-data.co.uk
    parent: str | None = None    # división inmediatamente superior (de ahí vienen los descendidos)
    promotion_playoff: tuple = ()  # ((desde_temporada, formato), ...): "efl4" (3.º-6.º) o "efl6" (3.º-8.º)
    split_since: int | None = None  # Argentina: desde este año, Apertura y Clausura con zonas y playoffs
    zone_tiebreak: tuple = ()       # desempate dentro de cada zona del torneo
    notes: str = ""

    def zone_of(self, pos: int) -> list[str]:
        return [z.key for z in self.zones if pos in z.positions]

    def playoff_format(self, syear: int) -> str | None:
        fmt = None
        for since, f in self.promotion_playoff:
            if syear >= since:
                fmt = f
        return fmt


def _rng(a: int, b: int) -> tuple:
    return tuple(range(a, b + 1))


LEAGUES: dict[str, League] = {
    "eng.1": League(
        code="eng.1", name="Premier League", country="Inglaterra", n_teams=20, season_style="split",
        tiebreak=("PTS", "GD", "GF", "H2H_PTS", "H2H_AGF", "LOTS"),
        zones=(Zone("champion", "Campeón", (1,)), Zone("ucl", "Champions League", _rng(1, 4)),
               Zone("relegation", "Descenso", _rng(18, 20))),
        mirror_division="E0",
        notes="El quinto cupo de Champions (European Performance Spot) depende del coeficiente UEFA; no se modela.",
    ),
    "esp.1": League(
        code="esp.1", name="LaLiga", country="España", n_teams=20, season_style="split",
        tiebreak=("PTS", "H2H_PTS", "H2H_GD", "GD", "GF", "LOTS"), h2h_reapply=True,
        zones=(Zone("champion", "Campeón", (1,)), Zone("ucl", "Champions League", _rng(1, 4)),
               Zone("relegation", "Descenso", _rng(18, 20))),
        mirror_division="SP1",
    ),
    "ita.1": League(
        code="ita.1", name="Serie A", country="Italia", n_teams=20, season_style="split",
        tiebreak=("PTS", "H2H_PTS", "H2H_GD", "GD", "GF", "LOTS"),
        title_playoff=True, relegation_playoff_pos=17,
        zones=(Zone("champion", "Campeón", (1,)), Zone("ucl", "Champions League", _rng(1, 4)),
               Zone("relegation", "Descenso", _rng(18, 20))),
        mirror_division="I1",
        notes="Desempate de dos equipos por el título: partido único; por el descenso: ida y vuelta (FIGC CU 331/A, 2025-26).",
    ),
    "ger.1": League(
        code="ger.1", name="Bundesliga", country="Alemania", n_teams=18, season_style="split",
        tiebreak=("PTS", "GD", "GF", "H2H_PTS", "H2H_AGF", "AGF", "LOTS"),
        zones=(Zone("champion", "Campeón", (1,)), Zone("ucl", "Champions League", _rng(1, 4)),
               Zone("relegation_playoff", "Promoción", (16,)), Zone("relegation", "Descenso", _rng(17, 18))),
        mirror_division="D1",
    ),
    "fra.1": League(
        code="fra.1", name="Ligue 1", country="Francia", n_teams=18, season_style="split",
        tiebreak=("PTS", "GD", "H2H_PTS", "H2H_GD", "H2H_GF", "H2H_AGF", "GF", "AGF", "LOTS"),
        zones=(Zone("champion", "Campeón", (1,)), Zone("ucl", "Champions League", _rng(1, 3)),
               Zone("relegation_playoff", "Barrage", (16,)), Zone("relegation", "Descenso", _rng(17, 18))),
        mirror_division="F1",
    ),
    "chi.1": League(
        code="chi.1", name="Liga de Primera", country="Chile", n_teams=16, season_style="calendar",
        tiebreak=("PTS", "GD", "W", "GF", "AGF", "LOTS"), title_playoff=True,
        zones=(Zone("champion", "Campeón", (1,)), Zone("libertadores", "Copa Libertadores", _rng(1, 2)),
               Zone("libertadores_playoff", "Repechaje Libertadores", (3,)),
               Zone("sudamericana", "Copa Sudamericana", _rng(4, 6)),
               Zone("relegation", "Descenso", _rng(15, 16))),
        notes="Orden de desempate según Wikipedia (sin verificar contra las Bases ANFP 2026).",
    ),
    "bra.1": League(
        code="bra.1", name="Brasileirão Série A", country="Brasil", n_teams=20, season_style="calendar",
        tiebreak=("PTS", "W", "GD", "GF", "H2H_PTS", "LOTS"),
        zones=(Zone("champion", "Campeón", (1,)), Zone("libertadores", "Copa Libertadores", _rng(1, 5)),
               Zone("sudamericana", "Copa Sudamericana", _rng(6, 11)),
               Zone("relegation", "Descenso", _rng(17, 20))),
        mirror_division="BRA",
        notes="Orden de desempate de 2025, asumido para 2026.",
    ),
    "eng.2": League(
        code="eng.2", name="Championship", country="Inglaterra", n_teams=24, season_style="split",
        tiebreak=("PTS", "GD", "GF", "H2H_PTS", "H2H_GD", "W", "AGF", "LOTS"),
        zones=(Zone("champion", "Campeón", (1,)), Zone("promotion", "Ascenso", _rng(1, 2)),
               Zone("promotion_playoff", "Playoffs de ascenso", _rng(3, 8)),
               Zone("relegation", "Descenso", _rng(22, 24))),
        mirror_division="E1", parent="eng.1",
        promotion_playoff=((2009, "efl4"), (2026, "efl6")),
        notes=("Desempate según el reglamento EFL (puntos, diferencia, goles, resultados entre sí, victorias, "
               "goles de visita). Playoffs desde 2026-27: 5.º-8.º y 6.º-7.º a partido único en casa del mejor "
               "ubicado; semifinales ida y vuelta (el 3.º contra el peor ubicado que siga vivo, vuelta en casa "
               "del mejor ubicado); final única en Wembley. Hasta 2025-26: 3.º-6.º y 4.º-5.º, ida y vuelta. "
               "Empates en llaves: alargue y penales (supuesto; el reglamento 2026-27 no lo detalla)."),
    ),
    "arg.1": League(
        code="arg.1", name="Liga Profesional", country="Argentina", n_teams=30, season_style="calendar",
        tiebreak=("PTS", "GD", "GF", "LOTS"),   # tabla anual (luego fair play, que no se modela)
        zone_tiebreak=("PTS", "GD", "GF", "LOTS"),
        zones=(Zone("relegation", "Descenso (tabla anual)", (30,)),),
        mirror_division="ARG", split_since=2025,
        notes=("Dos torneos por año (Apertura y Clausura), 30 equipos en dos zonas de 15: 14 partidos de zona y "
               "2 interzonales (el clásico y uno sorteado); el Clausura repite el fixture del Apertura con las "
               "localías invertidas. Los 8 primeros de cada zona juegan octavos (1.ºA-8.ºB, 1.ºB-8.ºA, 2.ºA-7.ºB, "
               "...), cuartos y semis a partido único en casa del mejor ubicado y final neutral; alargue y penales. "
               "Desempate en zona: puntos, diferencia, goles, resultados entre sí, fair play, sorteo (los dos "
               "últimos no se modelan). Descienden dos: el último de los promedios (puntos de las tres últimas "
               "temporadas sobre partidos jugados, sin playoffs) y el último de la tabla anual; si es el mismo "
               "equipo, el penúltimo de la tabla anual. Empates en esos puestos: partido de desempate."),
    ),
}

# Base de los promedios (Argentina): puntos y partidos de las dos temporadas anteriores en Primera,
# fase regular de todos los torneos (2024: Copa de la Liga + Liga Profesional, 41 partidos; 2025:
# Apertura + Clausura, 32). ESPN no trae la Copa de la Liga 2024, así que se carga desde la tabla
# oficial de promedios (verificada contra los partidos de 2026: mismos puntos y partidos).
# Copas del año siguiente por la temporada argentina (desde 2026, cambio aprobado por la LPF en marzo
# de 2026): a la Libertadores van los campeones del Apertura, del Clausura y de la Copa Argentina, y
# por la tabla anual —sin contar a los ya clasificados— el 1.º y el 2.º (fase de grupos) y el 9.º
# (fase 2); a la Sudamericana, del 3.º al 8.º. ``fixed`` son los clasificados ya conocidos (nombres
# de ESPN): el campeón del torneo ya jugado y, cuando se defina, el de la Copa Argentina (final el 4
# de noviembre de 2026). El campeón del torneo en curso sale de cada simulación.
CUP_SPOTS: dict[tuple[str, int], dict] = {
    ("arg.1", 2026): dict(fixed=["Belgrano (Córdoba)"], lib=(0, 1, 8), sud=(2, 3, 4, 5, 6, 7)),
}

PROMEDIO_BASE: dict[tuple[str, int], dict[str, tuple[int, int]]] = {
    ("arg.1", 2026): {
        "Boca Juniors": (129, 73), "River Plate": (123, 73), "Vélez Sarsfield": (116, 73),
        "Argentinos Juniors": (113, 73), "Rosario Central": (113, 73), "Racing Club": (123, 73),
        "Independiente": (110, 73), "Estudiantes de La Plata": (105, 73), "Lanús": (109, 73),
        "Huracán": (109, 73), "Talleres (Córdoba)": (106, 73), "Gimnasia (Mendoza)": (0, 0),
        "Independiente Rivadavia": (89, 73), "Unión (Santa Fe)": (99, 73), "Defensa y Justicia": (96, 73),
        "Barracas Central": (98, 73), "San Lorenzo": (96, 73), "Gimnasia La Plata": (86, 73),
        "Instituto (Córdoba)": (87, 73), "Belgrano (Córdoba)": (86, 73), "Deportivo Riestra": (100, 73),
        "Tigre": (88, 73), "Platense": (92, 73), "Atlético Tucumán": (84, 73), "Newell's Old Boys": (82, 73),
        "Central Córdoba (Santiago del Estero)": (84, 73), "Sarmiento (Junín)": (70, 73), "Banfield": (76, 73),
        "Aldosivi": (33, 32), "Estudiantes de Río Cuarto": (0, 0),
    },
}

# Ajustes de puntos (descuentos) por liga y temporada: {(liga, temporada): [(id ESPN, puntos, fecha), ...]}.
# ESPN no los aplica a los resultados, así que se cargan a mano. La fecha es la del anuncio: un
# pronóstico hecho antes no lo conoce (importa en los backtests); la tabla final los incluye todos.
POINT_ADJUSTMENTS: dict[tuple[str, int], list[tuple[str, int, str]]] = {
    ("eng.2", 2009): [("384", -10, "2010-01-27")],                       # Crystal Palace, administración
    ("eng.2", 2011): [("385", -10, "2012-02-17")],                       # Portsmouth, administración
    ("eng.2", 2018): [("392", -9, "2019-03-22")],                        # Birmingham, pérdidas (P&S)
    ("eng.2", 2019): [("350", -12, "2020-07-01")],                       # Wigan, administración
    ("eng.2", 2020): [("399", -6, "2020-08-01")],                        # Sheffield Wednesday, P&S (12 → 6 en apelación)
    ("eng.2", 2021): [("374", -12, "2021-09-22"), ("374", -9, "2021-11-16"),  # Derby: administración + contabilidad
                      ("338", -6, "2021-11-17")],                        # Reading, plan de negocios
    ("eng.2", 2022): [("338", -6, "2023-04-04")],                        # Reading, plan de negocios
    ("eng.2", 2025): [("399", -12, "2025-10-24"), ("399", -6, "2025-12-01"),  # Sheffield Wednesday
                      ("375", -6, "2026-02-05"),                         # Leicester, PSR
                      ("383", -2, "2026-04-24")],                        # West Brom, PSR
    ("eng.2", 2026): [("376", -4, "2026-07-01")],                        # Southampton, grabación de entrenamientos
}


def point_adjustments(comp: str, syear: int, asof=None) -> dict[str, int]:
    """Puntos a sumar por equipo (negativos = descuentos) conocidos a la fecha ``asof``."""
    import pandas as pd
    out: dict[str, int] = {}
    for team, pts, date in POINT_ADJUSTMENTS.get((comp, int(syear)), []):
        if asof is None or pd.Timestamp(date, tz="UTC") <= asof:
            out[team] = out.get(team, 0) + pts
    return out


# Ligas que el proyecto modela hoy (las demás están descargadas para el grafo entre ligas).
CORE_LEAGUES = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "eng.2", "chi.1", "bra.1", "arg.1"]
# Ligas que solo alimentan la capa europea de las copas (model.europe): se actualizan y se filtran,
# pero no se pronostican ni se muestran. ESPN no tiene (con ese código) Chequia, Ucrania, Eslovaquia,
# Azerbaiyán, Croacia, Serbia, Polonia, Hungría ni Bulgaria.
EUROPE_EXTRA = ["por.1", "ned.1", "tur.1", "bel.1", "gre.1", "aut.1", "nor.1", "sco.1", "den.1", "sui.1", "swe.1"]
CUP_COMPETITIONS = ["uefa.champions", "uefa.europa", "uefa.europa.conf",
                    "conmebol.libertadores", "conmebol.sudamericana", "fifa.cwc"]


def season_label(league: League, date) -> str:
    """Etiqueta de temporada para una fecha (pandas.Timestamp)."""
    if league.season_style == "calendar":
        return str(date.year)
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y}-{str(y + 1)[2:]}"
