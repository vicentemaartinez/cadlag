# càdlàg · sitio web

Next.js 15 (App Router) exportado como sitio estático.

```bash
npm install
npm run dev      # usa los datos de public/data (genéralos con `cadlag site` en la raíz)
npm run build    # descarga los datos de Supabase si CADLAG_DATA_URL está definida, y compila a out/
```

En Vercel: Root Directory `site`, variable `CADLAG_DATA_URL = <SUPABASE_URL>/storage/v1/object/public/cadlag-data`.
La actualización automática (GitHub Actions) sube datos nuevos y llama al deploy hook; ver el README de la raíz.

Rutas: `/`, `/liga/{premier-league,laliga,serie-a,bundesliga,ligue-1,chile,brasileirao}/`, `/mundo/`.
El tema parte en claro; la elección de oscuro se guarda en el navegador. Diseño: `../DESIGN.md`, producto: `../PRODUCT.md`.
