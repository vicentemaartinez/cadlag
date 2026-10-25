// Descarga los datos del sitio desde Supabase Storage (bucket público cadlag-data) antes de compilar.
// Sin CADLAG_DATA_URL usa lo que haya en public/data (generado localmente con `cadlag site`).
import { mkdir, writeFile, access, readFile } from "node:fs/promises";
import { join } from "node:path";

const base = (process.env.CADLAG_DATA_URL || "").replace(/\/$/, "");
const dir = join(process.cwd(), "public", "data");
const leagues = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "eng.2", "chi.1", "bra.1", "arg.1", "uefa.champions"];

if (!base) {
  try { await access(join(dir, "index.json")); console.log("CADLAG_DATA_URL no definida: se usan los datos locales de public/data"); process.exit(0); }
  catch { console.error("Faltan datos: define CADLAG_DATA_URL o genera public/data con `cadlag site`"); process.exit(1); }
}
await mkdir(dir, { recursive: true });
const get = async (f) => {
  const r = await fetch(`${base}/${f}?t=${Date.now()}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${f}: ${r.status}`);
  await writeFile(join(dir, f), Buffer.from(await r.arrayBuffer()));
};
// Primero el índice: define qué ligas tienen datos publicados. Una liga recién agregada al código
// todavía no está en Supabase hasta que corre la actualización; no debe romper el build.
await get("index.json");
const index = JSON.parse(await readFile(join(dir, "index.json"), "utf8"));
const have = new Set(index.leagues.map((l) => l.comp));
const missing = leagues.filter((c) => !have.has(c));
if (missing.length) console.warn(`aún sin datos (se omiten): ${missing.join(", ")}`);
const files = ["global.json", "trajectory_chi.1.json", ...leagues.filter((c) => have.has(c)).flatMap((c) => [`${c}.json`, `${c}.sims.gz`])];
await Promise.all(files.map(get));
console.log(`datos: ${files.length + 1} archivos desde ${base}`);
