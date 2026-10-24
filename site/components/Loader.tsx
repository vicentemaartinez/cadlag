"use client";
import { useEffect, useRef, useState } from "react";
import Glyph from "./Glyph";
import { fmtInt } from "@/lib/format";

// Persiana de entrada con contadores. Una vez por sesión (el script del <head> marca html.seen).
export default function Loader({ seasons, matches, leagues }: { seasons: number; matches: number; leagues: number }) {
  const [gone, setGone] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const v = [useRef<HTMLDivElement>(null), useRef<HTMLDivElement>(null), useRef<HTMLDivElement>(null), useRef<HTMLDivElement>(null)];
  const bar = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const seen = document.documentElement.classList.contains("seen") || matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (seen) { setGone(true); return; }
    try { sessionStorage.setItem("cadlag-seen", "1"); } catch {}
    const tot = [seasons, matches, matches * 2.65, leagues]; const t0 = performance.now(), dur = 1500; let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - t, 3);
      v.forEach((r, i) => { if (r.current) r.current.textContent = fmtInt(tot[i] * e); });
      if (bar.current) bar.current.style.transform = `scaleX(${e})`;
      if (t < 1) raf = requestAnimationFrame(step);
      else setTimeout(() => { root.current?.classList.add("up"); setTimeout(() => setGone(true), 1000); }, 240);
    };
    raf = requestAnimationFrame(step); return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (gone) return null;
  const k = ["temporadas simuladas", "partidos sorteados", "goles", "ligas"];
  return (
    <div className="loader" ref={root} aria-hidden="true">
      <div className="lt"><Glyph />càdlàg</div>
      <div>
        <div className="grid">{k.map((x, i) => <div key={x}><div className="k">{x}</div><div className="v" ref={v[i]}>0</div></div>)}</div>
        <div className="bar"><span ref={bar} /></div>
      </div>
    </div>
  );
}
