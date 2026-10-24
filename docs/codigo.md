# Código

Mapa del repositorio para quien quiera leerlo o extenderlo.

## Módulos (`src/cadlag/`)

| Módulo | Qué hace |
|---|---|
| `data/espn.py`, `store.py` | Descarga y limpieza (ESPN, IDs de equipo globales), ~106 mil partidos 2010–2027 |
| `data/events.py` | Tiros, tiros al arco, posesión y línea de tiempo de goles y tarjetas por partido |
| `data/mirror.py` | Cruce automático con football-data.co.uk (cuotas Bet365, Elo) para evaluar |
| `model/filter.py` | Núcleo de un latente (referencia): espacio de estados + Dixon–Coles, filtro de Laplace, hiperparámetros por verosimilitud prequential con gradientes exactos (JAX) |
| `sim/split.py` | Formato argentino: zonas inferidas del fixture, llave del torneo con resultados ya jugados, promedios y descensos |
| `sim/season.py`, `tiebreak.py` | Predictiva posterior de la temporada, desempates recursivos, partidos de definición, playoffs de ascenso, descuentos de puntos |
| `sim/analysis.py` | Importancia (información mutua), "qué necesita mi equipo", cotas por programación entera, suspenso |
| `model/crossleague.py` | Offsets de liga en el grafo de copas y ascensos; resistencia efectiva; Fiedler; ranking global |
| `model/hodge.py` | HodgeRank: gradiente / curl / armónico con nulo calibrado |
| `model/styles.py` | Interacción bilineal u_iᵀv_j (NumPyro) y prueba espectral |
| `model/filter_shots.py` | Tiros al arco como segunda medición del estado |
| `model/filter_shots2.py` | **Modelo oficial**: dos latentes, crear y evitar tiros al arco y convertirlos (NegBin adelgazada), mismo filtro y ajuste |
| `model/engine.py` | Motor por liga: el oficial, salvo que `models/model_selection.json` devuelva una liga al núcleo |
| `sim/rare.py` | Eventos raros por muestreo por importancia (fuerza + resultados inclinados) |
| `data/transfers.py` | Traspasos de Transfermarkt (top 5, 2009-2025) ubicados en los ids de ESPN |
| `model/inplay.py` | Proceso de conteo en vivo (marcador, rojas, descuentos), Hawkes, Kolmogorov hacia adelante |
| `model/changepoint.py` | BOCPD (Adams–MacKay) sobre residuos por equipo |
| `eval/` | Reglas de puntuación propias, de-vig (Shin), backtest walk-forward |
| `pipeline.py`, `export.py`, `cli.py` | Pronóstico, exportación al sitio y línea de comandos |
| `remote.py` | Estado y datos del sitio en Supabase Storage, aviso a Vercel |

## Estructura

```
src/cadlag/       motor: datos, modelo, simulación, evaluación, exportación al sitio
scripts/          análisis del artículo (backtests, Hodge, estilos, en vivo, entre ligas…)
site/             sitio web (Next.js, estático)
docs/             teoria.md (formulación y resultados), research.md (estado del arte), codigo.md, operacion.md
data/             raw/ (descargas) y processed/ (almacén; el vigente vive en Supabase)
models/           hiperparámetros por liga y modelo elegido (model_selection.json)
outputs/          análisis, backtests y global.json (comparación entre ligas)
.github/workflows actualización automática
PRODUCT.md, DESIGN.md  producto y sistema de diseño
```

## Comandos

```bash
cadlag update                # trae resultados recientes de ESPN
cadlag forecast              # simula las ligas (outputs/web/<liga>.json)
cadlag site                  # exporta al sitio: simulaciones, trayectoria, site/public/data
cadlag run                   # update y, si hubo resultados nuevos, forecast + site
cadlag fit                   # reajusta hiperparámetros (models/<liga>_shots2_h.npy)
pytest                       # pruebas
```

## Análisis

Cada script escribe en `outputs/analysis/`; los resultados están resumidos en `docs/teoria.md`.

```bash
python scripts/backtest.py eng.1 2016          # partidos y temporadas, walk-forward
sh scripts/backtest_all.sh                     # las cinco ligas europeas
python scripts/crossleague_analysis.py         # comparación entre ligas + outputs/web/global.json
python scripts/hodge_analysis.py 200           # intransitividad
python scripts/styles_analysis.py 2            # estilos latentes
python scripts/shots_analysis.py eng.1:2016    # tiros al arco
python scripts/inplay_analysis.py              # modelo en vivo y contagio
python scripts/changepoint_analysis.py         # cambios de régimen
python scripts/suspense_analysis.py 2016 20    # competitividad y suspenso
python scripts/nuts_validation.py eng.1 2025   # filtro de Laplace vs. NUTS
python scripts/plot_matrix.py chi.1            # matriz de posiciones (PNG)
python scripts/rare_events.py                  # eventos casi imposibles (y rare_refine.py)
python scripts/shots2_analysis.py eng.1:2016   # dos latentes para los tiros
python scripts/season_compare.py               # núcleo vs. oficial en posiciones finales
python scripts/shots2_vs_market.py             # oficial vs. Bet365 en los mismos partidos
python scripts/compare_models.py               # pronóstico actual con ambos modelos
python scripts/select_models.py                # confirma el modelo oficial por liga
python scripts/market_prior_analysis.py        # mercado de pases en la pretemporada
python scripts/transfer_edges_analysis.py      # traspasos como aristas entre ligas
python scripts/tournament_backtest.py          # Argentina: campeones y playoffs de cada torneo
python scripts/shots2_norel_fit.py eng.2:2016  # Championship: prior propio de los descendidos
```
