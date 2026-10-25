"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import type { IndexLeague } from "@/lib/types";
import { COUNTRY, NAMES, leagueHref } from "@/lib/leagues";
import { pct, rounds } from "@/lib/format";
import { cssVar, heat, onTheme, reloadHeat } from "@/lib/heat";

// Carrusel 3D (coverflow) como selector de ligas: arrastre, flechas, teclado y trackpad hacia los lados.
export default function Flow({ leagues }: { leagues: IndexLeague[] }) {
  const router = useRouter();
  const host = useRef<HTMLDivElement>(null);
  const st = useRef({ pos: 0, target: 0, raf: 0 });
  const start = Math.max(0, leagues.findIndex((l) => l.comp === "chi.1"));
  const [active, setActive] = useState(start);
  const reduced = useRef(false);

  useEffect(() => {
    const el = host.current!; const cards = [...el.querySelectorAll<HTMLElement>(".card")]; const s = st.current;
    reduced.current = matchMedia("(prefers-reduced-motion: reduce)").matches;
    s.pos = s.target = start;
    const n = leagues.length;
    const spacing = () => (cards[0] ? cards[0].offsetWidth * (innerWidth < 620 ? 0.52 : 0.66) : 200);
    const layout = () => {
      const sp = spacing(); const act = Math.round(s.pos);
      cards.forEach((c, i) => {
        const d = i - s.pos, ad = Math.abs(d), cd = Math.max(-1, Math.min(1, d));
        const x = d * sp + cd * sp * 0.18, z = -Math.min(ad, 3) * 150, rot = -cd * 42, sc = 1 - Math.min(ad, 3) * 0.03;
        c.style.transform = `translate(-50%,-50%) translateX(${x}px) translateZ(${z}px) rotateY(${rot}deg) scale(${sc})`;
        c.style.zIndex = String(100 - Math.round(ad * 10)); c.style.opacity = String(Math.max(0, 1 - Math.max(0, ad - 2.2) * 0.9));
        c.classList.toggle("on", i === act); c.setAttribute("aria-selected", String(i === act));
      });
      setActive(act);
    };
    const go = (i: number) => {
      s.target = Math.max(0, Math.min(n - 1, i)); cancelAnimationFrame(s.raf);
      if (reduced.current) { s.pos = s.target; layout(); return; }
      const step = () => { s.pos += (s.target - s.pos) * 0.14; if (Math.abs(s.target - s.pos) < 0.001) s.pos = s.target; layout(); if (s.pos !== s.target) s.raf = requestAnimationFrame(step); };
      s.raf = requestAnimationFrame(step);
    };
    const paint = () => {
      const paper = cssVar("--paper"), paper2 = cssVar("--paper-2");
      cards.forEach((c, li) => {
        const cv = c.querySelector("canvas")!; const T = leagues[li].teams, nn = T.length;
        const w = cv.clientWidth || 260, dpr = Math.min(2, devicePixelRatio || 1); cv.width = w * dpr; cv.height = w * dpr;
        const x = cv.getContext("2d")!; x.fillStyle = paper; x.fillRect(0, 0, cv.width, cv.height);
        const g = dpr, cell = (cv.width - g * (nn - 1)) / nn;
        T.forEach((t, r) => t.p.forEach((p, k) => { x.fillStyle = p < 0.0005 ? paper2 : heat(p); x.fillRect(k * (cell + g), r * (cell + g), cell, cell); }));
      });
    };
    (el as HTMLDivElement & { go?: (i: number) => void }).go = go;

    let down: { x: number; p: number; moved: boolean; vx: number; lx: number; lt: number } | null = null;
    const pd = (e: PointerEvent) => { down = { x: e.clientX, p: s.pos, moved: false, vx: 0, lx: e.clientX, lt: performance.now() }; el.setPointerCapture(e.pointerId); el.classList.add("drag"); };
    const pm = (e: PointerEvent) => {
      if (!down) return; const dx = e.clientX - down.x; if (Math.abs(dx) > 6) down.moved = true;
      const now = performance.now(); down.vx = (e.clientX - down.lx) / Math.max(1, now - down.lt); down.lx = e.clientX; down.lt = now;
      cancelAnimationFrame(s.raf); s.pos = s.target = Math.max(-0.4, Math.min(n - 0.6, down.p - dx / spacing())); layout();
    };
    const pu = (e: PointerEvent) => {
      if (!down) return; el.classList.remove("drag");
      if (down.moved) go(Math.round(s.pos - (down.vx * 180) / spacing()));
      else {
        const card = (document.elementFromPoint(e.clientX, e.clientY) as HTMLElement | null)?.closest<HTMLElement>(".card");
        if (card) { const i = +card.dataset.i!; if (i === Math.round(s.pos)) router.push(leagueHref(leagues[i].comp)); else go(i); }
      }
      down = null;
    };
    const pc = () => { down = null; el.classList.remove("drag"); go(Math.round(s.pos)); };
    let wt: ReturnType<typeof setTimeout> | undefined;
    const wheel = (e: WheelEvent) => {
      const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.shiftKey ? e.deltaY : 0; if (!dx) return;
      e.preventDefault(); cancelAnimationFrame(s.raf);
      s.pos = s.target = Math.max(-0.3, Math.min(n - 0.7, s.pos + dx / spacing())); layout();
      clearTimeout(wt); wt = setTimeout(() => go(Math.round(s.pos)), 120);
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") { go(Math.round(s.target) + 1); e.preventDefault(); }
      if (e.key === "ArrowLeft") { go(Math.round(s.target) - 1); e.preventDefault(); }
      if (e.key === "Enter") router.push(leagueHref(leagues[Math.round(s.target)].comp));
    };
    const click = (e: MouseEvent) => e.preventDefault();
    el.addEventListener("pointerdown", pd); el.addEventListener("pointermove", pm); el.addEventListener("pointerup", pu); el.addEventListener("pointercancel", pc);
    el.addEventListener("wheel", wheel, { passive: false }); el.addEventListener("keydown", key); el.addEventListener("click", click);
    const resize = () => { layout(); paint(); };
    addEventListener("resize", resize);
    reloadHeat(); layout(); paint(); document.fonts?.ready.then(resize);
    const off = onTheme(paint);
    return () => {
      off(); removeEventListener("resize", resize); cancelAnimationFrame(s.raf);
      el.removeEventListener("pointerdown", pd); el.removeEventListener("pointermove", pm); el.removeEventListener("pointerup", pu); el.removeEventListener("pointercancel", pc);
      el.removeEventListener("wheel", wheel); el.removeEventListener("keydown", key); el.removeEventListener("click", click);
    };
  }, [leagues, router, start]);

  const go = (i: number) => (host.current as (HTMLDivElement & { go?: (i: number) => void }) | null)?.go?.(i);
  return (
    <section className="wrap" id="ligas" style={{ paddingTop: 0 }}>
      <div className="flow" ref={host} tabIndex={0} role="listbox" aria-label="Ligas">
        {leagues.map((l, i) => {
          const v = (t: (typeof l.teams)[number]) => t.asc ?? t.title ?? t.champion ?? t.p[0]; const up = l.teams[0].asc != null, tor = l.teams[0].title != null, cup = !!l.cup;
          let f = l.teams[0]; for (const t of l.teams) if (v(t) > v(f)) f = t;
          const [r, R] = rounds(l.matches_played, l.matches_total, l.teams.length);
          return (
            <Link key={l.comp} className="card" href={leagueHref(l.comp)} data-i={i} role="option" aria-selected={i === active} aria-label={NAMES[l.comp]} draggable={false}>
              <div className="top"><span>{COUNTRY[l.comp]} · {l.season_label}</span><span className="r"><span className="when tnum">fecha {r} de {R}</span><span className="open">Abrir →</span></span></div>
              <h3>{NAMES[l.comp]}</h3>
              <canvas aria-hidden="true" />
              <div className="fav"><div className="nm">{f.name}<small>{up ? "favorito al ascenso" : tor ? "favorito al torneo" : cup ? "favorito a ganarla" : "favorito al título"}</small></div><div className="p tnum">{pct(v(f), 0)}<small style={{ fontSize: ".5em" }}> %</small></div></div>
            </Link>
          );
        })}
      </div>
      <div className="flow-ctl">
        <div className="flow-idx">
          {leagues.map((l, i) => <button key={l.comp} type="button" aria-current={i === active} onClick={() => go(i)}>{NAMES[l.comp]}</button>)}
        </div>
        <div className="flow-arrows">
          <button type="button" aria-label="Liga anterior" onClick={() => go(active - 1)}>←</button>
          <button type="button" aria-label="Liga siguiente" onClick={() => go(active + 1)}>→</button>
        </div>
      </div>
    </section>
  );
}
