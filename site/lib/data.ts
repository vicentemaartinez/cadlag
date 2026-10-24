// Solo en el servidor (build): lee los JSON exportados por scripts/export_web.py.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { LEAGUES } from "./leagues";
import type { Global, Index, League, Trajectory } from "./types";

const dir = join(process.cwd(), "public", "data");
const read = <T,>(f: string): T => JSON.parse(readFileSync(join(dir, f), "utf8")) as T;

export const getIndex = () => read<Index>("index.json");
export const getLeague = (comp: string) => read<League>(`${comp}.json`);
// Ligas con datos publicados (una liga recién agregada aparece cuando la actualización sube sus datos).
export const availableLeagues = (): string[] => {
  const have = new Set(getIndex().leagues.map((l) => l.comp));
  return LEAGUES.filter((c) => have.has(c) && existsSync(join(dir, `${c}.json`)));
};
export const getGlobal = () => read<Global>("global.json");
export const getTrajectory = (comp: string): Trajectory | null => {
  try { return read<Trajectory>(`trajectory_${comp}.json`); } catch { return null; }
};
