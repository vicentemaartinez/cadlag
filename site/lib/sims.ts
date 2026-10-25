import { BASE } from "./leagues";

// Cada temporada simulada ocupa W bytes (ver cadlag.export.export_sims):
//   o[K]   resultado sorteado de los próximos K partidos (0 local, 1 empate, 2 visita)
//   q[3K]  P(L, E, V | fuerzas de esa temporada) como 255·√p (precisión relativa en p chicas)
//   pts[n] puntos finales      pos[n] posición final (0-based); con ascenso, el bit 7 marca "asciende"
// Argentina agrega zpts[n] (puntos del torneo en curso) y zinfo[n]: puesto en la zona (bits 0-3),
// campeón del torneo (bit 4) y descenso (bit 5). Las copas agregan stage[n]: etapa alcanzada
// (0 fase liga, 1 playoff, 2 octavos, 3 cuartos, 4 semis, 5 final, 6 campeón).
export type SplitMeta = {
  zone: number[]; prev_pts: number[]; prev_games: number[]; games: number[]; promedios: boolean; tournament: string;
  cups?: { fixed: number[]; lib: number[]; sud: number[] };   // Libertadores y Sudamericana por la tabla anual
};
export type Sim = {
  S: number; K: number; n: number; buf: Uint8Array; W: number;
  asc: boolean;
  qwin: Float64Array | null;   // n×n: P(gana los playoffs de ascenso | equipo, posición)
  split: SplitMeta | null;
  qtitle: Float64Array | null; // n×16: P(campeón del torneo | equipo, puesto en su zona)
  cup: boolean;
  qstage: Float64Array | null; // n×n×7: P(llega al menos a la etapa k | equipo, puesto en la fase liga)
};

const cache = new Map<string, Promise<Sim>>();
export const LAYOUT = "o[K] sqrtq[3K] pts[n] pos[n]";
export const LAYOUTS = [LAYOUT, LAYOUT + " zpts[n] zinfo[n]", LAYOUT + " stage[n]"];
// columnas "llega al menos a": octavos, cuartos, semis, final, campeón
export const REACH = { r16: 2, qf: 3, sf: 4, final: 5, champion: 6 } as const;
export type ReachKey = keyof typeof REACH;
// tabla de decodificación: byte → p
const DEC = Float64Array.from({ length: 256 }, (_, b) => (b / 255) ** 2);

export function loadSims(comp: string, S: number, K: number, n: number, asc = false, split: SplitMeta | null = null, cup = false): Promise<Sim> {
  if (!cache.has(comp)) {
    cache.set(comp, (async () => {
      const res = await fetch(`${BASE}/data/${comp}.sims.gz`);
      let bytes = new Uint8Array(await res.arrayBuffer());
      // algunos servidores ya lo entregan descomprimido (Content-Encoding: gzip)
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        const ds = new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")));
        bytes = new Uint8Array(await ds.arrayBuffer());
      }
      const W = 4 * K + 2 * n + (split ? 2 * n : 0) + (cup ? n : 0);
      let qwin: Float64Array | null = null, qtitle: Float64Array | null = null;
      if (asc) {
        const cnt = new Float64Array(n * n), win = new Float64Array(n * n);
        for (let s = 0; s < S; s++) for (let t = 0; t < n; t++) {
          const b = bytes[s * W + 4 * K + n + t], p = b & 0x7f;
          if (p < 2) continue;
          cnt[t * n + p]++; if (b & 0x80) win[t * n + p]++;
        }
        qwin = new Float64Array(n * n);
        for (let i = 0; i < n * n; i++) qwin[i] = cnt[i] ? win[i] / cnt[i] : 0;
      }
      if (split) {
        const cnt = new Float64Array(n * 16), win = new Float64Array(n * 16);
        for (let s = 0; s < S; s++) for (let t = 0; t < n; t++) {
          const b = bytes[s * W + 4 * K + 3 * n + t], z = b & 0x0f;
          cnt[t * 16 + z]++; if (b & 0x10) win[t * 16 + z]++;
        }
        qtitle = new Float64Array(n * 16);
        for (let i = 0; i < n * 16; i++) qtitle[i] = cnt[i] ? win[i] / cnt[i] : 0;
      }
      let qstage: Float64Array | null = null;
      if (cup) {
        // P(etapa ≥ k | equipo, puesto), encogida hacia la de cualquier equipo en ese puesto (50 simulaciones de peso)
        const cnt = new Float64Array(n * n), hit = new Float64Array(n * n * 7), pc = new Float64Array(n), ph = new Float64Array(n * 7);
        for (let s = 0; s < S; s++) for (let t = 0; t < n; t++) {
          const p = bytes[s * W + 4 * K + n + t] & 0x7f, g = bytes[s * W + 4 * K + 2 * n + t];
          cnt[t * n + p]++; pc[p]++;
          for (let k = 0; k <= g; k++) { hit[(t * n + p) * 7 + k]++; ph[p * 7 + k]++; }
        }
        qstage = new Float64Array(n * n * 7);
        for (let t = 0; t < n; t++) for (let p = 0; p < n; p++) for (let k = 0; k < 7; k++) {
          const pooled = pc[p] ? ph[p * 7 + k] / pc[p] : 0;
          qstage[(t * n + p) * 7 + k] = (hit[(t * n + p) * 7 + k] + 50 * pooled) / (cnt[t * n + p] + 50);
        }
      }
      return { S, K, n, buf: bytes, W, asc, qwin, split, qtitle, cup, qstage };
    })());
  }
  return cache.get(comp)!;
}

export type Extras = Partial<Record<"asc" | "title" | "playoffs" | "rel" | "lib" | "sud" | ReachKey, Float64Array>>;

export type Scenario = {
  M: Float64Array;      // n×n, P(equipo t termina en la posición k)
  xpts: Float64Array;   // puntos finales esperados
  epos: Float64Array;   // posición esperada (1-based)
  probs: Float64Array;  // K×3, P(L, E, V) de cada partido dado el escenario
  ex: Extras;           // ascenso; campeón del torneo, playoffs, descenso y copas (Argentina)
  zM?: Float64Array;    // Argentina: n×16, P(equipo t termina en el puesto k de su zona en el torneo en curso)
  zx?: Float64Array;    // puntos esperados en el torneo en curso
  zepos?: Float64Array; // puesto esperado en su zona (1-based)
  ess: number;          // tamaño efectivo de muestra (Σw)² / Σw²
  cnt: number;          // temporadas con peso > 0
};

const PH = [3, 1, 0], PA = [0, 1, 3];

/** Condiciona en resultados fijados reponderando todas las temporadas:
 *  w_s = Π_k P(o_k fijado | fuerzas de s). Dadas las fuerzas los partidos son independientes, así
 *  que el resto de la temporada s sigue siendo una muestra válida; los partidos fijados se
 *  reemplazan por el resultado elegido (se corrigen los puntos y se reordena la tabla por puntos,
 *  desempatando con la posición que tenía cada equipo en esa simulación). Lo que depende de una
 *  llave (ascenso por playoffs, campeón del torneo) se toma del sorteo de esa temporada si el orden
 *  no cambió, y si cambió, de la probabilidad de ganarla desde la nueva posición. */
export function scenario(sim: Sim, fixed: Record<number, number>, hi: number[], ai: number[]): Scenario {
  const { S, K, n, buf, W, split } = sim;
  const M = new Float64Array(n * n), xpts = new Float64Array(n), epos = new Float64Array(n), probs = new Float64Array(K * 3);
  const ex: Extras = {};
  if (sim.asc) ex.asc = new Float64Array(n);
  if (split) { ex.title = new Float64Array(n); ex.playoffs = new Float64Array(n); if (split.promedios) ex.rel = new Float64Array(n); }
  if (split?.cups) { ex.lib = new Float64Array(n); ex.sud = new Float64Array(n); }
  const zM = split ? new Float64Array(n * 16) : undefined, zx = split ? new Float64Array(n) : undefined, zepos = split ? new Float64Array(n) : undefined;
  const qd = new Uint8Array(n);
  const rkeys = Object.keys(REACH) as ReachKey[];
  if (sim.cup) for (const k of rkeys) ex[k] = new Float64Array(n);
  const qwin = sim.qwin, qtitle = sim.qtitle;
  const flag = new Uint8Array(n);
  const keys = Object.keys(fixed).map(Number), vals = keys.map((k) => fixed[k]);
  const pts = new Float64Array(n), orig = new Int32Array(n), pos = new Int32Array(n), idx = Array.from({ length: n }, (_, i) => i);
  const zp = new Float64Array(n), zo = new Int32Array(n), zn = new Int32Array(n), zinfo = new Uint8Array(n);
  const zones = split ? [0, 1].map((z) => split.zone.map((v, i) => (v === z ? i : -1)).filter((i) => i >= 0)) : [];
  let sw = 0, sw2 = 0, cnt = 0;
  for (let s = 0; s < S; s++) {
    const o = s * W;
    let w = 1;
    for (let i = 0; i < keys.length; i++) w *= DEC[buf[o + K + 3 * keys[i] + vals[i]]];
    if (w <= 0) continue;
    cnt++;
    let changed = false;
    for (let t = 0; t < n; t++) { pts[t] = buf[o + 4 * K + t]; const b = buf[o + 4 * K + n + t]; orig[t] = b & 0x7f; flag[t] = b >> 7; }
    if (split) for (let t = 0; t < n; t++) { zp[t] = buf[o + 4 * K + 2 * n + t]; zinfo[t] = buf[o + 4 * K + 3 * n + t]; zo[t] = zinfo[t] & 0x0f; }
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i], so = buf[o + k], v = vals[i];
      if (so !== v) {
        changed = true;
        pts[hi[k]] += PH[v] - PH[so]; pts[ai[k]] += PA[v] - PA[so];
        if (split) { zp[hi[k]] += PH[v] - PH[so]; zp[ai[k]] += PA[v] - PA[so]; }
      }
    }
    if (changed) {
      idx.sort((a, b) => pts[b] - pts[a] || orig[a] - orig[b]);
      for (let r = 0; r < n; r++) pos[idx[r]] = r;
    } else pos.set(orig);
    for (let t = 0; t < n; t++) { M[t * n + pos[t]] += w; xpts[t] += w * pts[t]; epos[t] += w * (pos[t] + 1); }
    if (sim.cup) {
      // misma etapa si el puesto no cambió; si cambió, P(etapa | equipo, nuevo puesto)
      for (let t = 0; t < n; t++) {
        const g = buf[o + 4 * K + 2 * n + t];
        for (const k of rkeys) ex[k]![t] += w * (pos[t] === orig[t] ? (g >= REACH[k] ? 1 : 0) : sim.qstage![(t * n + pos[t]) * 7 + REACH[k]]);
      }
    }
    if (ex.asc) {
      let same = true; for (let t = 0; t < n && same; t++) same = pos[t] === orig[t];
      for (let t = 0; t < n; t++) ex.asc[t] += w * (same ? flag[t] : pos[t] < 2 ? 1 : qwin![t * n + pos[t]]);
    }
    if (split) {
      let zsame = true;
      if (changed) {
        for (const zt of zones) {
          const ord = zt.slice().sort((a, b) => zp[b] - zp[a] || zo[a] - zo[b]);
          ord.forEach((t, r) => { zn[t] = r; if (r !== zo[t]) zsame = false; });
        }
      } else zn.set(zo);
      for (let t = 0; t < n; t++) {
        ex.playoffs![t] += w * (zn[t] < 8 ? 1 : 0);
        ex.title![t] += w * (zsame ? (zinfo[t] >> 4) & 1 : qtitle![t * 16 + zn[t]]);
        zM![t * 16 + zn[t]] += w; zx![t] += w * zp[t]; zepos![t] += w * (zn[t] + 1);
      }
      if (split.cups) {
        if (!changed) for (let t = 0; t < n; t++) { ex.lib![t] += w * ((zinfo[t] >> 6) & 1); ex.sud![t] += w * ((zinfo[t] >> 7) & 1); }
        else {
          // se recorre la tabla anual saltando a los ya clasificados (campeones conocidos y el del torneo
          // en curso de esta simulación; si cambió su llave, se mantiene el campeón sorteado)
          qd.fill(0); for (const f of split.cups.fixed) qd[f] = 1;
          for (let t = 0; t < n; t++) if ((zinfo[t] >> 4) & 1) qd[t] = 1;
          for (let t = 0; t < n; t++) if (qd[t]) ex.lib![t] += w;
          let e = 0;
          for (let r = 0; r < n; r++) {
            const t = idx[r]; if (qd[t]) continue;
            if (split.cups.lib.includes(e)) ex.lib![t] += w; else if (split.cups.sud.includes(e)) ex.sud![t] += w;
            e++;
          }
        }
      }
      if (ex.rel) {
        if (!changed) for (let t = 0; t < n; t++) ex.rel[t] += w * ((zinfo[t] >> 5) & 1);
        else {
          // último de los promedios y último de la tabla anual (sin él)
          let r1 = 0, best = Infinity;
          for (let t = 0; t < n; t++) {
            const g = split.prev_games[t] + split.games[t]; const p = g ? (split.prev_pts[t] + pts[t]) / g : Infinity;
            if (p < best - 1e-12) { best = p; r1 = t; }
          }
          let r2 = -1;
          for (let r = n - 1; r >= 0; r--) if (idx[r] !== r1) { r2 = idx[r]; break; }
          ex.rel[r1] += w; if (r2 >= 0) ex.rel[r2] += w;
        }
      }
    }
    for (let j = 0; j < K * 3; j++) probs[j] += w * DEC[buf[o + K + j]];
    sw += w; sw2 += w * w;
  }
  if (sw > 0) {
    for (let i = 0; i < M.length; i++) M[i] /= sw;
    for (let t = 0; t < n; t++) { xpts[t] /= sw; epos[t] /= sw; }
    for (const a of Object.values(ex)) for (let t = 0; t < n; t++) a![t] /= sw;
    if (zM) { for (let i = 0; i < zM.length; i++) zM[i] /= sw; for (let t = 0; t < n; t++) { zx![t] /= sw; zepos![t] /= sw; } }
    for (let k = 0; k < K; k++) { const z = probs[3 * k] + probs[3 * k + 1] + probs[3 * k + 2]; for (let c = 0; c < 3; c++) probs[3 * k + c] /= z || 1; }
  }
  return { M, xpts, epos, probs, ex, zM, zx, zepos, ess: sw2 > 0 ? (sw * sw) / sw2 : 0, cnt };
}

// Matriz base sin simulaciones (desde los JSON), para el primer render.
export function baseMatrix(order: string[], teams: { id: string; position_probs: number[] }[]) {
  const n = order.length; const M = new Float64Array(n * n);
  for (const t of teams) { const i = order.indexOf(t.id); t.position_probs.forEach((p, k) => (M[i * n + k] = p)); }
  return M;
}
