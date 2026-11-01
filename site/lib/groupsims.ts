import { BASE } from "./leagues";

// Laboratorio de las copas con fase de grupos (Libertadores y Sudamericana). Las dos copas comparten
// un archivo (ver cadlag.cup_conmebol.lab_blob); cada edición simulada ocupa W bytes:
//   por copa, desde su offset: o[K] resultado sorteado de sus próximos K partidos de grupo (0/1/2) y
//   q[3K] P(L, E, V | fuerzas de esa edición) como 255·√p
//   desde base, para los n equipos de las dos copas: pts[n] puntos de grupo, gpos[n] puesto en su
//   grupo (0-3), lib[n] y sud[n] etapa alcanzada en cada copa + 1 (0 = no la juega).
export type GroupSimsMeta = {
  file: string; S: number; n: number; W: number; base: number; order: string[];
  cup_index: number; K: number; off: number;
  team_cup: (number | null)[]; team_group: (number | null)[];
  fixtures: { home: string; away: string; hi: number; ai: number; date: string; p: number[] }[];
};
export type GroupSim = {
  meta: GroupSimsMeta; buf: Uint8Array;
  // P(etapa ≥ k en esta copa | equipo, puesto en su grupo): n×4×8, encogida hacia la de cualquier equipo
  // de la misma copa en ese puesto (50 ediciones de peso)
  qstage: Float64Array;
  col: number;   // offset de la columna de etapas de esta copa (lib o sud) dentro del bloque base
};

const DEC = Float64Array.from({ length: 256 }, (_, b) => (b / 255) ** 2);
const NS = 8;
const raw = new Map<string, Promise<Uint8Array>>();

export function loadGroupSims(meta: GroupSimsMeta, comp: string): Promise<GroupSim> {
  if (!raw.has(meta.file)) {
    raw.set(meta.file, (async () => {
      const res = await fetch(`${BASE}/data/${meta.file}`);
      let bytes = new Uint8Array(await res.arrayBuffer());
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        const ds = new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")));
        bytes = new Uint8Array(await ds.arrayBuffer());
      }
      return bytes;
    })());
  }
  return raw.get(meta.file)!.then((buf) => {
    const { S, n, W, base } = meta;
    const col = base + (comp.endsWith("libertadores") ? 2 : 3) * n;
    const cnt = new Float64Array(n * 4), hit = new Float64Array(n * 4 * NS);
    for (let s = 0; s < S; s++) {
      const o = s * W;
      for (let t = 0; t < n; t++) {
        const p = buf[o + base + n + t], g = buf[o + col + t];
        cnt[t * 4 + p]++;
        for (let k = 0; k <= g && k < NS; k++) hit[(t * 4 + p) * NS + k]++;
      }
    }
    // tasas agrupadas por copa del equipo (la etapa en la Sudamericana de un equipo de la Libertadores
    // depende de su puesto allá: el tercero va al playoff)
    const pc = new Float64Array(2 * 4), ph = new Float64Array(2 * 4 * NS);
    for (let t = 0; t < n; t++) {
      const c = meta.team_cup[t] ?? 0;
      for (let p = 0; p < 4; p++) { pc[c * 4 + p] += cnt[t * 4 + p]; for (let k = 0; k < NS; k++) ph[(c * 4 + p) * NS + k] += hit[(t * 4 + p) * NS + k]; }
    }
    const qstage = new Float64Array(n * 4 * NS);
    for (let t = 0; t < n; t++) {
      const c = meta.team_cup[t] ?? 0;
      for (let p = 0; p < 4; p++) for (let k = 0; k < NS; k++) {
        const pooled = pc[c * 4 + p] ? ph[(c * 4 + p) * NS + k] / pc[c * 4 + p] : 0;
        qstage[(t * 4 + p) * NS + k] = (hit[(t * 4 + p) * NS + k] + 50 * pooled) / (cnt[t * 4 + p] + 50);
      }
    }
    return { meta, buf, qstage, col };
  });
}

export type GroupScenario = {
  G: Float64Array;      // n×4: P(equipo t termina k.º en su grupo)
  reach: Float64Array;  // n×8: P(etapa ≥ k en esta copa), con la etapa codificada + 1 (1 = la juega)
  xpts: Float64Array;   // puntos de grupo esperados
  probs: Float64Array;  // K×3: P(L, E, V) de cada partido dado el escenario
  ess: number; cnt: number;
};

const PH = [3, 1, 0], PA = [0, 1, 3];

/** Igual que el laboratorio de las ligas: w_s = Π_k P(o_k fijado | fuerzas de s); los partidos fijados
 *  se reemplazan por el resultado elegido, se corrigen los puntos y se reordena cada grupo tocado por
 *  puntos (desempate: el puesto que tenía cada equipo en esa simulación). La etapa alcanzada se toma de
 *  la simulación si el puesto no cambió y, si cambió, de P(etapa | equipo, nuevo puesto). */
export function groupScenario(sim: GroupSim, fixed: Record<number, number>): GroupScenario {
  const { meta, buf, qstage, col } = sim;
  const { S, n, W, base, K, off, fixtures } = meta;
  const G = new Float64Array(n * 4), reach = new Float64Array(n * NS), xpts = new Float64Array(n), probs = new Float64Array(K * 3);
  const keys = Object.keys(fixed).map(Number), vals = keys.map((k) => fixed[k]);
  const ci = meta.cup_index;
  // miembros de cada grupo de esta copa
  const groups: number[][] = [];
  meta.team_cup.forEach((c, t) => { if (c === ci) { const g = meta.team_group[t]!; (groups[g] ??= []).push(t); } });
  const touched = new Set<number>(keys.map((k) => meta.team_group[fixtures[k].hi]!));
  const pts = new Float64Array(n), orig = new Uint8Array(n), pos = new Uint8Array(n);
  let sw = 0, sw2 = 0, cnt = 0;
  for (let s = 0; s < S; s++) {
    const o = s * W;
    let w = 1;
    for (let i = 0; i < keys.length; i++) w *= DEC[buf[o + off + K + 3 * keys[i] + vals[i]]];
    if (w <= 0) continue;
    cnt++;
    for (let t = 0; t < n; t++) { pts[t] = buf[o + base + t]; orig[t] = buf[o + base + n + t]; }
    pos.set(orig);
    let changed = false;
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i], so = buf[o + off + k], v = vals[i];
      if (so !== v) { changed = true; const f = fixtures[k]; pts[f.hi] += PH[v] - PH[so]; pts[f.ai] += PA[v] - PA[so]; }
    }
    if (changed) for (const g of touched) {
      const ord = groups[g].slice().sort((a, b) => pts[b] - pts[a] || orig[a] - orig[b]);
      ord.forEach((t, r) => (pos[t] = r));
    }
    for (let t = 0; t < n; t++) {
      if (meta.team_cup[t] === ci) { G[t * 4 + pos[t]] += w; xpts[t] += w * pts[t]; }
      if (pos[t] === orig[t]) { const g = buf[o + col + t]; for (let k = 1; k <= g && k < NS; k++) reach[t * NS + k] += w; }
      else for (let k = 1; k < NS; k++) reach[t * NS + k] += w * qstage[(t * 4 + pos[t]) * NS + k];
    }
    for (let j = 0; j < K * 3; j++) probs[j] += w * DEC[buf[o + off + K + j]];
    sw += w; sw2 += w * w;
  }
  if (sw > 0) {
    for (let i = 0; i < G.length; i++) G[i] /= sw;
    for (let i = 0; i < reach.length; i++) reach[i] /= sw;
    for (let t = 0; t < n; t++) xpts[t] /= sw;
    for (let k = 0; k < K; k++) { const z = probs[3 * k] + probs[3 * k + 1] + probs[3 * k + 2]; for (let c = 0; c < 3; c++) probs[3 * k + c] /= z || 1; }
  }
  return { G, reach, xpts, probs, ess: sw2 > 0 ? (sw * sw) / sw2 : 0, cnt };
}
