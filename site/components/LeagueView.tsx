"use client";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { League, Zone } from "@/lib/types";
import { COUNTRY, NAMES } from "@/lib/leagues";
import { dateShort, fmt, fmtInt, initials, pct, rounds } from "@/lib/format";
import { heat, onTheme, reloadHeat } from "@/lib/heat";
import { baseMatrix, LAYOUTS, loadSims, REACH, scenario, type Extras, type ReachKey, type Scenario, type Sim } from "@/lib/sims";
import Roll from "./Roll";
import TeamDrawer from "./TeamDrawer";

const zoneProb = (M: Float64Array, n: number, t: number, z?: Zone) => (z ? z.positions.reduce((a, p) => a + M[t * n + p - 1], 0) : 0);
// columnas extra después de la tabla: ascenso (Championship); playoffs, campeón y descenso (Argentina)
type XKey = keyof Extras;
const XCOLS: Record<XKey, { label: string; title: string; json: string }> = {
  asc: { label: "Asc.", title: "Probabilidad de ascender (directo o por playoffs)", json: "promoted" },
  playoffs: { label: "Play.", title: "Probabilidad de clasificar a los playoffs del torneo", json: "playoffs" },
  title: { label: "Camp.", title: "Probabilidad de salir campeón del torneo", json: "title" },
  rel: { label: "Desc.", title: "Probabilidad de descender (promedios o tabla anual)", json: "relegated" },
  // copas: llega al menos a esa ronda
  r16: { label: "8vos", title: "Probabilidad de jugar los octavos (directo o por el playoff)", json: "r16" },
  qf: { label: "4tos", title: "Probabilidad de llegar a cuartos de final", json: "qf" },
  sf: { label: "Semi", title: "Probabilidad de llegar a semifinales", json: "sf" },
  final: { label: "Final", title: "Probabilidad de jugar la final", json: "final" },
  champion: { label: "Camp.", title: "Probabilidad de ganar la copa", json: "champion" },
};
const XORDER: XKey[] = ["asc", "playoffs", "title", "rel", "r16", "qf", "sf", "final", "champion"];
const zoneClass = (z: Zone) => (["relegation", "out"].includes(z.key) ? "zone-bot" : ["champion", "ucl", "libertadores", "promotion", "r16"].includes(z.key) ? "zone-top" : "zone-mid");

function tween(dur: number, fn: (e: number) => void) {
  if (!dur || matchMedia("(prefers-reduced-motion: reduce)").matches) { fn(1); return; }
  const t0 = performance.now(); const step = (now: number) => { const t = Math.min(1, (now - t0) / dur); fn(1 - Math.pow(1 - t, 4)); if (t < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

export default function LeagueView({ D }: { D: League }) {
  const comp = D.meta.comp, n = D.order.length, S = D.sims.S;
  const idx = useCallback((id: string) => D.order.indexOf(id), [D.order]);
  // primer render (antes de bajar las simulaciones): matriz y puntos esperados de los JSON
  const base0 = useMemo<Scenario>(() => {
    const M = baseMatrix(D.order, D.teams);
    const xpts = new Float64Array(n), epos = new Float64Array(n);
    const ex: Extras = {};
    const keys: XKey[] = D.sims.asc ? ["asc"] : D.sims.split ? (D.sims.split.promedios ? ["playoffs", "title", "rel"] : ["playoffs", "title"])
      : D.sims.cup ? (Object.keys(REACH) as ReachKey[]) : [];
    for (const k of keys) ex[k] = new Float64Array(n);
    D.teams.forEach((t) => { const i = D.order.indexOf(t.id); xpts[i] = t.xpts; epos[i] = t.expected_position; for (const k of keys) ex[k]![i] = t.zones[XCOLS[k].json] ?? 0; });
    return { M, xpts, epos, probs: new Float64Array(D.sims.K * 3), ex, ess: S, cnt: S };
  }, [D, S, n]);
  const fxHi = useMemo(() => D.sims.fixtures.map((f) => D.order.indexOf(f.home)), [D]);
  const fxAi = useMemo(() => D.sims.fixtures.map((f) => D.order.indexOf(f.away)), [D]);
  const [sim, setSim] = useState<Sim | null>(null);
  const [fixed, setFixed] = useState<Record<number, number>>({});
  const [sort, setSort] = useState<"table" | "exp">("table");
  const [drawer, setDrawer] = useState<number | null>(null);
  const [themeTick, setThemeTick] = useState(0);

  useEffect(() => { if (!LAYOUTS.includes(D.sims.layout ?? "")) return; let alive = true; loadSims(comp, S, D.sims.K, n, !!D.sims.asc, D.sims.split ?? null, !!D.sims.cup).then((s) => alive && setSim(s)); return () => { alive = false; }; }, [comp, S, D.sims.K, n, D.sims.layout]);
  useEffect(() => onTheme(() => setThemeTick((x) => x + 1)), []);

  const base = useMemo(() => (sim ? scenario(sim, {}, fxHi, fxAi) : base0), [sim, base0, fxHi, fxAi]);
  const cur = useMemo(() => {
    if (!sim || !Object.keys(fixed).length) return base;
    return scenario(sim, fixed, fxHi, fxAi);
  }, [sim, fixed, base, fxHi, fxAi]);
  // puntos y partidos jugados con los resultados fijados
  const extra = useMemo(() => {
    const pts = new Float64Array(n), pj = new Float64Array(n);
    Object.entries(fixed).forEach(([k, o]) => { const h = fxHi[+k], a = fxAi[+k]; pts[h] += [3, 1, 0][o]; pts[a] += [0, 1, 3][o]; pj[h]++; pj[a]++; });
    return { pts, pj };
  }, [fixed, fxHi, fxAi, n]);
  const M = cur.M;
  const any = Object.keys(fixed).length > 0;
  const zones = D.meta.zones;
  const cup = !!D.sims.cup;
  const zTop = zones.find((z) => z.key === "champion"), zRel = zones.find((z) => z.key === "relegation" || z.key === "out");

  // --- héroe: el objetivo de arriba es el título, ascender (Championship) o ganar el torneo (Argentina)
  const isAsc = !!D.sims.asc && !!cur.ex.asc;
  const split = D.sims.split ?? null;
  const xkeys = (Object.keys(XCOLS) as XKey[]).filter((k) => cur.ex[k]).sort((a, b) => XORDER.indexOf(a) - XORDER.indexOf(b));
  const headKey: XKey | null = isAsc ? "asc" : split ? "title" : cup ? "champion" : null;
  const top = (t: number) => (headKey ? cur.ex[headKey]![t] : M[t * n]);
  const baseTop = (t: number) => (headKey ? base.ex[headKey]![t] : zoneProb(base.M, n, t, zTop));
  const relP = (sc: Scenario, t: number) => (sc.ex.rel ? sc.ex.rel[t] : zoneProb(sc.M, n, t, zRel));
  let best = 0; for (let t = 1; t < n; t++) if (top(t) > top(best)) best = t;
  const favName = D.teams.find((x) => x.id === D.order[best])!.name; const pFav = top(best);
  const FIX = 6 + xkeys.length; // columnas fijas antes de la matriz
  const [r, R] = rounds(D.meta.matches_played, D.meta.matches_total, n);
  const relTop = [...Array(n).keys()].map((t) => [t, relP(cur, t)] as const).sort((a, b) => b[1] - a[1])[0];
  // copas: el equipo más en duda de seguir en carrera (P(fuera en la fase liga) más cerca de 1/2)
  const edge = [...Array(n).keys()].map((t) => [t, relP(cur, t)] as const).sort((a, b) => Math.abs(a[1] - 0.5) - Math.abs(b[1] - 0.5))[0];
  const H = D.meta.suspense.champion_entropy_bits, Hm = D.meta.suspense.champion_max_bits;

  // --- orden de filas
  // tabla actual + resultados fijados: puntos; en empate, el orden actual (aproxima la diferencia de gol)
  const tableOrder = useMemo(() => D.teams.map((t, r) => [idx(t.id), t.points + extra.pts[idx(t.id)], r] as const)
    .sort((a, b) => b[1] - a[1] || a[2] - b[2]).map((x) => x[0]), [D.teams, idx, extra]);
  const rankNow = useMemo(() => { const m = new Map<number, number>(); tableOrder.forEach((t, i) => m.set(t, i + 1)); return m; }, [tableOrder]);
  const order = useMemo(() => (sort === "table" ? tableOrder : [...Array(n).keys()].sort((a, b) => cur.epos[a] - cur.epos[b])), [sort, tableOrder, n, cur]);

  // --- matriz: FLIP de filas + interpolación de celdas
  const tbody = useRef<HTMLTableSectionElement>(null);
  const tops = useRef(new Map<number, number>());
  const prevCells = useRef<Float64Array | null>(null);
  useLayoutEffect(() => {
    const tb = tbody.current; if (!tb) return; const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    tb.querySelectorAll<HTMLTableRowElement>("tr").forEach((tr) => {
      const t = +tr.dataset.t!; const top = tr.offsetTop; const old = tops.current.get(t);
      if (old !== undefined && old !== top && !reduced) {
        tr.style.transition = "none"; tr.style.transform = `translateY(${old - top}px)`;
        requestAnimationFrame(() => { tr.style.transition = "transform .55s cubic-bezier(.77,0,.175,1)"; tr.style.transform = ""; });
      }
      tops.current.set(t, top);
    });
  }, [order]);
  useEffect(() => {
    const tb = tbody.current; if (!tb) return; reloadHeat();
    const target = Float64Array.from(M); const prev = prevCells.current;
    const cells: [HTMLTableCellElement, number][] = [];
    tb.querySelectorAll<HTMLTableCellElement>("td.cell").forEach((td) => cells.push([td, +td.dataset.t! * n + +td.dataset.k!]));
    const paint = (td: HTMLTableCellElement, v: number) => {
      if (v < 0.0005) { td.textContent = ""; td.style.background = ""; td.classList.add("e"); td.style.color = ""; return; }
      td.classList.remove("e"); td.style.background = heat(v); td.style.color = v > 0.33 ? "var(--heat-text)" : "";
      td.textContent = v >= 0.001 ? (v >= 0.9995 ? "100" : fmt(v * 100, 1)) : "·";
    };
    tween(prev ? 520 : 0, (e) => { for (const [td, i] of cells) { const a = prev ? prev[i] : target[i]; paint(td, a + (target[i] - a) * e); } });
    prevCells.current = target;
  }, [M, n, themeTick]);

  // --- tooltip, cruz y halo que sigue al mouse
  const shell = useRef<HTMLDivElement>(null), tip = useRef<HTMLDivElement>(null), table = useRef<HTMLTableElement>(null);
  const last = useRef<{ tr: Element; td: Element; col?: Element } | null>(null);
  const clearHl = () => { const l = last.current; if (l) { l.tr.classList.remove("rowhl"); l.td.classList.remove("hl"); l.col?.classList.remove("colhl"); last.current = null; } };
  const onOver = (e: React.PointerEvent) => {
    const td = (e.target as HTMLElement).closest<HTMLTableCellElement>("td.cell"); if (!td) return;
    const tr = td.parentElement!; const k = +td.dataset.k!, t = +td.dataset.t!; clearHl();
    const col = table.current!.querySelectorAll("col")[FIX + k]; tr.classList.add("rowhl"); td.classList.add("hl"); col?.classList.add("colhl"); last.current = { tr, td, col };
    const v = M[t * n + k]; const rr = td.getBoundingClientRect(); const tp = tip.current!;
    tp.textContent = `${D.teams.find((x) => x.id === D.order[t])!.name} · ${k + 1}.º · ${v < 0.0005 ? "menos de 0,05" : fmt(v * 100, 1)} %`;
    tp.style.left = rr.left + rr.width / 2 + "px"; tp.style.top = rr.top + "px"; tp.classList.add("on");
  };

  // --- laboratorio
  const fx = D.sims.fixtures;
  const pick = (k: number, o: number) => {
    if (!sim) return;
    const next = { ...fixed }; if (next[k] === o) delete next[k]; else next[k] = o;
    if (scenario(sim, next, fxHi, fxAi).cnt === 0) return; // combinación imposible en el modelo
    setFixed(next);
  };
  const favAll = () => { const f: Record<number, number> = {}; fx.forEach((x, k) => (f[k] = x.p[0] >= x.p[2] ? 0 : 2)); if (sim && scenario(sim, f, fxHi, fxAi).cnt) setFixed(f); };
  const labN = useRef<HTMLElement>(null); const labPrev = useRef(S);
  useEffect(() => { const from = labPrev.current, to = cur.ess; labPrev.current = to; tween(500, (e) => { if (labN.current) labN.current.textContent = fmtInt(from + (to - from) * e); }); }, [cur.ess]);

  const zrow = [...Array(n).keys()].map((p) => zones.filter((z) => z.key !== "champion").find((z) => z.positions.includes(p + 1)) || null);
  const zheads: { z: Zone | null; span: number }[] = []; for (let p = 0; p < n;) { let q = p; while (q + 1 < n && zrow[q + 1] === zrow[p]) q++; zheads.push({ z: zrow[p], span: q - p + 1 }); p = q + 1; }
  const rowsById = new Map(D.teams.map((t) => [idx(t.id), t]));
  const nf = Object.keys(fixed).length;

  return (
    <>
      <header className="lhero wrap">
        <div className="lhero-grid">
          <div className="lead">
            <span className="label">{NAMES[comp]} · {COUNTRY[comp]} · {D.meta.season_label} · fecha {r} de {R}</span>
            <h1>{favName} <span className="soft">{cup ? (pFav >= 0.5 ? `va camino a ganar la ${NAMES[comp]}` : `es el favorito a la ${NAMES[comp]}`) : isAsc ? (pFav >= 0.5 ? "va camino a la Premier" : "es el favorito a subir") : split ? (pFav >= 0.5 ? `va camino a ganar el ${split.tournament}` : `es el favorito al ${split.tournament}`) : pFav >= 0.5 ? "va camino al título" : "lidera, pero nada está decidido"}</span></h1>
            <div className="big">
              <div className="pct tnum"><Roll value={pct(pFav)} /><small>%</small></div>
              <p className="cap">{cup ? (pFav >= 0.5 ? "de probabilidad de ganarla." : "de ganarla; las llaves reparten mucho.") : isAsc ? "de probabilidad de ascender, directo o por playoffs." : split ? (pFav >= 0.5 ? "de probabilidad de salir campeón." : "de salir campeón; los playoffs reparten mucho.") : pFav >= 0.5 ? "de probabilidad de salir campeón." : "es el favorito más alto."}</p>
            </div>
          </div>
          <aside className="side">
            <div className="plate"><div className="k">{split ? `Incertidumbre del ${split.tournament}` : cup ? "Incertidumbre del campeón" : "Incertidumbre del título"}</div><div className="v tnum">{fmt(H, 2)} <i>de {fmt(Hm, 0)} bits</i></div><div className="meter"><span style={{ transform: `scaleX(${H / Hm})` }} /></div><div className="s">{cup ? "0 es campeón decidido." : "0 es liga decidida."}</div></div>
            {cup
              ? <div className="plate"><div className="k">En la cuerda floja</div><div className="v">{D.teams.find((x) => x.id === D.order[edge[0]])!.name}</div><div className="s tnum">{pct(edge[1])} % de quedar fuera en la fase liga</div></div>
              : <div className="plate"><div className="k">Más cerca del descenso</div><div className="v">{D.teams.find((x) => x.id === D.order[relTop[0]])!.name}</div><div className="s tnum">{pct(relTop[1])} % de descender</div></div>}
            <div className="plate"><div className="k">Quedan por jugar</div><div className="v tnum">{D.meta.matches_total - D.meta.matches_played} <i>partidos</i></div><div className="s tnum">{D.meta.matches_played} jugados de {D.meta.matches_total}</div></div>
          </aside>
        </div>
      </header>

      <section className="wrap" id="matriz" style={{ paddingTop: 8 }}>
        <div className="board">
          <div className="matrix-shell" ref={shell} onPointerMove={(e) => { const rr = shell.current!.getBoundingClientRect(); shell.current!.style.setProperty("--mx", e.clientX - rr.left + "px"); shell.current!.style.setProperty("--my", e.clientY - rr.top + "px"); }}>
            <div className="mx-toolbar">
              <h2 className="mx-title">{cup ? "Puesto en la fase liga" : "Posiciones finales"}</h2>
              <div className="seg" role="group" aria-label="Orden de filas">
                <button type="button" aria-pressed={sort === "table"} onClick={() => setSort("table")}>Tabla actual</button>
                <button type="button" aria-pressed={sort === "exp"} onClick={() => setSort("exp")}>Posición esperada</button>
              </div>
            </div>
            <div className="mx-scroll">
              <table className="mx" ref={table} style={{ minWidth: 400 + xkeys.length * 50 + n * 38 }} onPointerOver={onOver} onPointerLeave={() => { tip.current?.classList.remove("on"); clearHl(); }}>
                <colgroup><col style={{ width: 26 }} /><col style={{ width: 210 }} /><col style={{ width: 30 }} /><col style={{ width: 40 }} /><col style={{ width: 38 }} /><col style={{ width: 56 }} />{xkeys.map((k) => <col key={k} style={{ width: 50 }} />)}{[...Array(n)].map((_, k) => <col key={k} />)}</colgroup>
                <thead>
                  <tr className="zones"><th colSpan={FIX} />{zheads.map((h, i) => <th key={i} colSpan={h.span}>{h.z && <div className={zoneClass(h.z)} title={h.z.label}>{h.z.label}</div>}</th>)}</tr>
                  <tr className="cols"><th /><th style={{ textAlign: "left", paddingLeft: 4 }}>Equipo</th><th style={{ textAlign: "right", paddingRight: 6 }}>PJ</th><th style={{ textAlign: "right", paddingRight: 6 }}>Pts</th><th style={{ textAlign: "right", paddingRight: 6 }}>xPts</th><th style={{ textAlign: "right", paddingRight: 8 }} title="Posición esperada">Pos. esp.</th>{xkeys.map((k) => <th key={k} className="asc-h" title={XCOLS[k].title}>{XCOLS[k].label}</th>)}{[...Array(n)].map((_, k) => <th key={k} className="p">{k + 1}</th>)}</tr>
                </thead>
                <tbody ref={tbody}>
                  {order.map((t) => {
                    const T = rowsById.get(t)!;
                    const dTop = top(t) - baseTop(t), dRel = relP(cur, t) - relP(base, t);
                    const [dv, lab] = Math.abs(dTop) >= Math.abs(dRel) ? [dTop, isAsc ? "ascenso" : cup ? "campeón" : "título"] : [-dRel, cup ? "seguir" : "salvarse"];
                    const show = any && Math.abs(dv) >= 0.005;
                    return (
                      <tr key={t} data-t={t}>
                        <td className="pos tnum">{rankNow.get(t)}</td>
                        <td className="team"><button className="team-btn" type="button" onClick={() => setDrawer(t)}>
                          <span className="badge">{initials(T.name)}</span><span className="nm">{T.name}</span>{T.zone && <span className="ztag" title={`Zona ${T.zone}`}>{T.zone}</span>}
                          <span className={"delta" + (show ? " show " + (dv > 0 ? "up" : "down") : "")}>{show ? `${dv > 0 ? "+" : "−"}${fmt(Math.abs(dv) * 100, 1)} ${lab}` : ""}</span>
                        </button></td>
                        <td className="num tnum">{T.played + extra.pj[t]}</td>
                        <td className="num pts tnum" title={T.adj ? `Incluye un descuento de ${-T.adj} puntos` : undefined}>{T.points + extra.pts[t]}{T.adj ? <span className="adj">*</span> : null}{extra.pj[t] > 0 && <sup className="plus">+{extra.pts[t]}</sup>}</td>
                        <td className="num tnum">{fmt(cur.xpts[t], 0)}</td>
                        <td className="num tnum epos">{fmt(cur.epos[t], 1)}{(() => { const dv = cur.epos[t] - base.epos[t]; return any && Math.abs(dv) >= 0.05 ? <sup className={dv < 0 ? "up" : "down"}>{dv < 0 ? "↑" : "↓"}{fmt(Math.abs(dv), 1)}</sup> : null; })()}</td>
                        {xkeys.map((k) => { const v = cur.ex[k]![t]; return <td key={k} className={"num tnum asc-c" + (k === "rel" ? " rel" : "")} style={{ background: v >= 0.0005 ? heat(v) : undefined, color: v > 0.33 ? "var(--heat-text)" : undefined }}>{v >= 0.0005 ? (v >= 0.9995 ? "100" : fmt(v * 100, 1)) : ""}</td>; })}
                        {[...Array(n)].map((_, k) => <td key={k} className="cell tnum" data-t={t} data-k={k} />)}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="mx-foot"><span>{any ? `Escenario con ${nf} resultado${nf > 1 ? "s" : ""} fijado${nf > 1 ? "s" : ""}` : `Pronóstico base · ${D.meta.matches_played} de ${D.meta.matches_total} partidos jugados`}</span><span className="tnum">{any ? `muestra efectiva ${fmtInt(cur.ess)} de ${fmtInt(S)}` : `${fmtInt(S)} ${cup ? "ediciones" : "temporadas"}`}</span></div>
            <p className="note" style={{ marginTop: 4 }}>En % sobre {fmtInt(S)} {cup ? "ediciones" : "temporadas"}. {cup ? "Posiciones de la fase liga: del 1 al 8 a octavos, del 9 al 24 al playoff. Columnas: llega al menos a octavos, cuartos, semis y final, y gana la copa. " : ""}{isAsc ? "Asc.: asciende directo o por playoffs. " : split ? `Tabla anual (Apertura + Clausura). Play.: clasifica a octavos del ${split.tournament}; Camp.: lo gana; Desc.: desciende por promedios o por la tabla anual. ` : ""}Toca un equipo para ver qué necesita.{D.teams.filter((x) => x.adj).map((x) => ` * ${x.name}: descuento de ${-x.adj!} puntos.`).join("")}</p>
          </div>

          <aside className="lab" aria-labelledby="lab-title">
            <div className="lab-head">
              <h3 id="lab-title">¿Y si…?</h3>
              <p>Fija resultados y la matriz se recalcula.</p>
              <div className="counter"><span>Muestra efectiva</span><span className="tnum"><b ref={labN}>{fmtInt(cur.ess)}</b> de {fmtInt(S)}</span></div>
              <div className="sample-bar"><span style={{ transform: `scaleX(${Math.min(1, cur.ess / S)})`, backgroundColor: cur.ess < 400 ? "var(--beni)" : undefined }} /></div>
              <div className={"warn" + (cur.ess < 400 ? " on" : "")}>Muestra efectiva bajo 400: resultados ruidosos.</div>
            </div>
            <ul className="fx-list">
              {fx.map((f, k) => {
                const here = fixed[k] !== undefined;
                const pr = sim && any ? [cur.probs[3 * k], cur.probs[3 * k + 1], cur.probs[3 * k + 2]] : f.p;
                const hn = D.teams.find((x) => x.id === f.home)!.name, an = D.teams.find((x) => x.id === f.away)!.name;
                return (
                  <li className="fx" key={k}>
                    <div className="fx-top"><div className="fx-teams"><span>{hn}</span><span style={{ color: "var(--ink-3)" }}>–</span><span>{an}</span></div><span className="when">{dateShort(f.date, { weekday: "short", day: "numeric", month: "short" })}</span></div>
                    <div className={"pick" + (here ? " locked" : "")} role="group" aria-label={`Resultado de ${hn} contra ${an}`}>
                      {["L", "E", "V"].map((l, o) => (
                        <button key={o} type="button" disabled={!sim} aria-pressed={fixed[k] === o} title={["Gana el local", "Empate", "Gana la visita"][o]}
                          style={{ flexGrow: here ? (fixed[k] === o ? 1 : 0.18) : Math.max(0.15, pr[o]) }} onClick={() => pick(k, o)}>
                          <b>{l}</b><span className="pp">{fmt(pr[o] * 100, 0)} %</span>
                        </button>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ul>
            <div className="lab-foot"><button className="btn small" type="button" onClick={() => setFixed({})}>Reiniciar</button><button className="btn small" type="button" onClick={favAll} disabled={!sim}>Gana el favorito</button></div>
          </aside>
        </div>
      </section>

      <section className="wrap" id="decisivos">
        <div className="sh"><div className="t"><h2>Partidos decisivos</h2></div><p>Cuánta incertidumbre sobre {cup ? "el campeón y quién queda fuera" : <>{isAsc ? "el ascenso" : split ? `el ${split.tournament}` : "el título"} y el descenso</>} resuelve cada resultado, en bits.</p></div>
        <div className="decisive">
          {(() => { const up = (d: (typeof D.decisive)[number]) => (isAsc ? d.mi_promotion ?? 0 : d.mi_title);
            const list = D.decisive.slice(0, 6); const mx = Math.max(...list.map((d) => up(d) + d.mi_relegation), 1e-6);
            return list.map((d, i) => (
              <div className="dm" key={i}>
                <div className="rk tnum">{i + 1}</div>
                <div><div className="m">{d.home_name}<i>contra</i>{d.away_name}</div><div className="sub">{dateShort(d.date)} · L {pct(d.p_home, 0)} · E {pct(d.p_draw, 0)} · V {pct(d.p_away, 0)}</div></div>
                <div className="bits">{fmt(up(d) + d.mi_relegation, 2)} bits
                  <div className="bar"><span className="t" style={{ width: `${(up(d) / mx) * 100}%` }} /><span style={{ left: `${(up(d) / mx) * 100}%`, width: `${(d.mi_relegation / mx) * 100}%` }} /></div>
                  <div className="split">{up(d) >= 0.005 && <span><i className="dot" />{isAsc ? "ascenso" : cup ? "campeón" : "título"} {fmt(up(d), 2)}</span>}{d.mi_relegation >= 0.005 && <span><i className="dot r" />{cup ? "eliminación" : "descenso"} {fmt(d.mi_relegation, 2)}</span>}</div>
                </div>
              </div>
            )); })()}
        </div>
      </section>

      <div className="tip" ref={tip} role="tooltip" />
      <TeamDrawer D={D} t={drawer} M={M} ex={cur.ex} onClose={() => setDrawer(null)} />
    </>
  );
}
