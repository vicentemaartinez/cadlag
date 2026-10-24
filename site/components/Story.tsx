"use client";
import { useEffect, useRef } from "react";
import type { Trajectory } from "@/lib/types";
import { dateShort, pct } from "@/lib/format";

// La trayectoria càdlàg de la probabilidad de título, dibujada con el scroll (solo en la portada).
export default function Story({ T }: { T: Trajectory }) {
  const sec = useRef<HTMLElement>(null), chart = useRef<HTMLDivElement>(null), svg = useRef<SVGSVGElement>(null);
  const num = useRef<HTMLDivElement>(null), when = useRef<HTMLDivElement>(null), hint = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    // doble rueda: total = n(n−1) partidos, n/2 por fecha
    const nTeams = Math.round((1 + Math.sqrt(1 + 4 * T.total)) / 2); const mpr = nTeams / 2;
    let st: { W: number; H: number; X: (v: number) => number; Y: (v: number) => number; L: number; lens: number[]; cw: number } | null = null;

    const build = () => {
      const cw = chart.current!.clientWidth || 760;
      const W = Math.round(Math.max(380, Math.min(760, cw))), H = Math.round(W < 560 ? W * 0.8 : W * 0.56), pad = { l: 42, r: 12, t: 16, b: 50 };
      const total = T.total, P = T.points, last = P[P.length - 1].played;
      const X = (v: number) => pad.l + (v / total) * (W - pad.l - pad.r), Y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
      const step = (j: number) => { let d = ""; P.forEach((q, i) => { const x0 = X(q.played), x1 = X(i + 1 < P.length ? P[i + 1].played : last), y = Y(q.p[j]);
        d += `M${x0.toFixed(1)},${y.toFixed(1)}H${x1.toFixed(1)}`; if (i + 1 < P.length) d += `V${Y(P[i + 1].p[j]).toFixed(1)}`; }); return d; };
      let main = `M${X(P[0].played)},${Y(P[0].p[0])}`;
      P.forEach((q, i) => { main += `H${X(i + 1 < P.length ? P[i + 1].played : last).toFixed(1)}`; if (i + 1 < P.length) main += `V${Y(P[i + 1].p[0]).toFixed(1)}`; });
      const jumps: { x: number; y0: number; y1: number }[] = [];
      for (let i = 1; i < P.length; i++) if (Math.abs(P[i].p[0] - P[i - 1].p[0]) >= 0.045) jumps.push({ x: X(P[i].played), y0: Y(P[i - 1].p[0]), y1: Y(P[i].p[0]) });
      const gx = (W < 560 ? [0, 10, 20, 30] : [0, 5, 10, 15, 20, 25, 30]).map((f) => f * mpr), gy = [0, 0.25, 0.5, 0.75, 1];
      const s = svg.current!; s.setAttribute("viewBox", `0 0 ${W} ${H}`);
      s.innerHTML = `
        ${gy.map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--rule)"/><text x="${pad.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v * 100} %</text>`).join("")}
        ${gx.map((v) => `<text x="${X(v)}" y="${H - pad.b + 18}" text-anchor="middle">${v / mpr}</text>`).join("")}
        <text x="${(pad.l + W - pad.r) / 2}" y="${H - 6}" text-anchor="middle" style="fill:var(--ink-3)">fecha</text>
        <rect x="${X(last)}" y="${pad.t}" width="${X(total) - X(last)}" height="${H - pad.t - pad.b}" fill="var(--paper-2)"/>
        <text x="${(X(last) + X(total)) / 2}" y="${pad.t + 16}" text-anchor="middle" style="fill:var(--ink-3)">por jugar</text>
        <g id="tr-others" opacity="0">${T.teams.slice(1).map((_, j) => `<path d="${step(j + 1)}" fill="none" stroke="var(--ink-3)" stroke-width="1" stroke-opacity=".55"/>`).join("")}</g>
        <path id="tr-main" d="${main}" fill="none" stroke="var(--beni)" stroke-width="2.4" stroke-linejoin="round"/>
        <g id="tr-jumps">${jumps.map((j) => `<g data-x="${j.x}" opacity="0"><circle cx="${j.x}" cy="${j.y0}" r="4" fill="var(--paper)" stroke="var(--beni)" stroke-width="1.6"/><circle cx="${j.x}" cy="${j.y1}" r="4" fill="var(--beni)"/></g>`).join("")}</g>
        <line id="tr-now" x1="0" x2="0" y1="${pad.t}" y2="${H - pad.b}" stroke="var(--ink)" stroke-width="1" stroke-dasharray="2 3"/>
        <text id="tr-head" style="fill:var(--beni);font-weight:600">${T.teams[0].name}</text>
        <text id="tr-oth" x="${X(last) - 6}" y="${Y(0) - 8}" text-anchor="end" style="fill:var(--ink-3)">el resto de los equipos</text>`;
      const p = s.querySelector<SVGPathElement>("#tr-main")!; const L = p.getTotalLength(); p.style.strokeDasharray = String(L);
      const lens = P.map((_, i) => { let acc = 0; for (let k = 0; k <= i; k++) { if (k > 0) acc += Math.abs(Y(P[k].p[0]) - Y(P[k - 1].p[0])); if (k < i) acc += X(k + 1 < P.length ? P[k + 1].played : last) - X(P[k].played); } return acc; });
      st = { W, H, X, Y, L, lens, cw };
    };

    const update = () => {
      if (!st || !sec.current) return; const s = svg.current!; const pin = sec.current.querySelector<HTMLElement>(".story-pin")!;
      const nav = document.getElementById("nav")?.getBoundingClientRect().height ?? 56;
      const r = sec.current.getBoundingClientRect(); const span = Math.max(1, r.height - pin.offsetHeight);
      const t = reduced ? 1 : Math.min(1, Math.max(0, (nav - r.top) / span) * 1.02);
      const Ld = st.L * t; s.querySelector<SVGPathElement>("#tr-main")!.style.strokeDashoffset = String(st.L - Ld);
      const P = T.points; let i = 0; while (i + 1 < P.length && st.lens[i + 1] <= Ld) i++;
      const q = P[i]; const xn = st.X(i + 1 < P.length ? P[i + 1].played : q.played);
      const xNow = st.X(q.played) + Math.max(0, Math.min(xn - st.X(q.played), Ld - st.lens[i]));
      const now = s.querySelector("#tr-now")!; now.setAttribute("x1", String(xNow)); now.setAttribute("x2", String(xNow));
      const op = String(Math.min(1, t * 3)); s.querySelector("#tr-others")!.setAttribute("opacity", op); s.querySelector("#tr-oth")!.setAttribute("opacity", op);
      s.querySelectorAll<SVGGElement>("#tr-jumps g").forEach((g) => g.setAttribute("opacity", +g.dataset.x! <= xNow + 0.5 ? "1" : "0"));
      const hd = s.querySelector("#tr-head")!; const right = xNow > st.W * 0.7;
      hd.setAttribute("x", String(right ? xNow - 8 : xNow + 8)); hd.setAttribute("y", String(st.Y(q.p[0]) - 10)); hd.setAttribute("text-anchor", right ? "end" : "start");
      const v = pct(q.p[0]); if (num.current && num.current.dataset.v !== v) { num.current.dataset.v = v; num.current.innerHTML = `${v}<small>%</small>`; }
      if (when.current) when.current.textContent = q.played === 0 ? "pretemporada · antes del primer partido"
        : `después de ${q.played} partidos (fecha ${Math.round(q.played / mpr)}) · ${dateShort(q.date, { day: "numeric", month: "long" })}${i === P.length - 1 ? " · hoy" : ""}`;
      if (hint.current) hint.current.style.opacity = t > 0.04 ? "0" : "1";
    };

    build(); update();
    const onScroll = () => update();
    const onResize = () => { const cw = chart.current!.clientWidth; if (st && Math.abs(cw - st.cw) > 40) build(); update(); };
    addEventListener("scroll", onScroll, { passive: true }); addEventListener("resize", onResize);
    return () => { removeEventListener("scroll", onScroll); removeEventListener("resize", onResize); };
  }, [T]);

  return (
    <section className="story" id="trayectoria" ref={sec}>
      <div className="story-pin"><div className="wrap story-grid">
        <div className="story-text">
          <h2>{T.teams[0].name}, {T.syear}</h2>
          <p className="story-sub">Probabilidad de salir campeón</p>
          <div className="story-num tnum" ref={num}>{pct(T.points[0].p[0])}<small>%</small></div>
          <div className="story-when" ref={when}>pretemporada · antes del primer partido</div>
          <p className="hint" ref={hint}>Baja para avanzar la temporada ↓</p>
        </div>
        <div className="story-chart" ref={chart}><svg ref={svg} role="img" aria-label="Probabilidad de título a lo largo de la temporada" /></div>
      </div></div>
    </section>
  );
}
