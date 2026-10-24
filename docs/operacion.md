# Operación

Cómo se mantiene actualizado el sitio y qué hay que tocar a mano.

## Actualización automática

GitHub Actions (`.github/workflows/update.yml`) corre `cadlag run --remote` cada 2 horas:

1. descarga el almacén de partidos desde Supabase Storage (bucket privado `cadlag-state`);
2. trae de ESPN los resultados de los últimos 3 días, con tiros al arco (y completa días de los
   últimos dos meses a los que les falten);
3. si cambió alguna liga núcleo, re-simula, sube los datos del sitio al bucket público
   `cadlag-data` y llama al deploy hook de Vercel, que vuelve a compilar el sitio descargando
   esos datos (`site/scripts/fetch-data.mjs`).

Así el repositorio solo guarda código: los datos que cambian viven en Supabase.

Configuración (una vez):

- Supabase: crear un proyecto (o usar uno existente) y, con `SUPABASE_URL` y `SUPABASE_SECRET_KEY`
  en el entorno, correr `cadlag remote-init` para crear los buckets.
- GitHub → Settings → Secrets and variables → Actions: `SUPABASE_URL`, `SUPABASE_SECRET_KEY`,
  `VERCEL_DEPLOY_HOOK`.
- Vercel: Root Directory `site`; variable de entorno
  `CADLAG_DATA_URL = <SUPABASE_URL>/storage/v1/object/public/cadlag-data`;
  Settings → Git → Deploy Hooks → crear uno y copiar la URL al secreto `VERCEL_DEPLOY_HOOK`.
- Primera corrida: Actions → "actualizar datos" → Run workflow con *force* marcado.

## Fuentes y mantenimiento

- ESPN, API no oficial `site.api.espn.com/apis/site/v2/sports/soccer/<liga>/scoreboard`
  (misma fuente que *marcador*). Desde algunos servidores conviene `site.web.api.espn.com`.
- football-data.co.uk y clubelo.com vía github.com/xgabora/Club-Football-Match-Data-2000-2025 (licencia MIT,
  ver `data/raw/footballdata_mirror/LICENSE`).
- Traspasos: github.com/eordo/transfermarkt-data (Transfermarkt, ligas top 5, 2009-10 a 2025-26). No
  se incluyen en el repositorio: `cadlag.data.transfers.load()` los descarga la primera vez.
- Descuentos de puntos (EFL): no vienen en ESPN; se cargan a mano con su fecha en
  `src/cadlag/leagues.py` (`POINT_ADJUSTMENTS`). Cuando la EFL anuncie uno nuevo, se agrega ahí.
- Promedios de Argentina: los puntos y partidos de las dos temporadas anteriores van en
  `PROMEDIO_BASE` (mismo archivo); para 2026 vienen de la tabla oficial; desde 2027 se calculan solos con los datos.

## Notas

- En repositorios públicos GitHub desactiva los flujos programados tras 60 días sin commits; si el
  sitio deja de actualizarse, se reactiva en Actions → "actualizar datos".
- Una actualización completa (nueve ligas, con las curvas de "¿Qué necesita?") toma unos 15 minutos.
