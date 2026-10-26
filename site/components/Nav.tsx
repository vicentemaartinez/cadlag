"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import Glyph from "./Glyph";
import { CUPS, EUROPE, LEAGUES, NAMES, SOUTH, leagueHref } from "@/lib/leagues";
import { fmt, pct } from "@/lib/format";
import { THEME_EVENT, reloadHeat } from "@/lib/heat";

export type NavSummary = { comp: string; fav: string; p: number }[];
type Panel = "ligas" | "copas" | "modelo" | "mundo" | "proyecto";

export default function Nav({ summary, topTeam, lambda2 }: { summary: NavSummary; topTeam: string; lambda2: number }) {
  const [open, setOpen] = useState<Panel | null>(null);
  const [sheet, setSheet] = useState(false);
  const [h, setH] = useState(0);
  const inner = useRef<HTMLDivElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const path = usePathname();
  const fav = (c: string) => summary.find((s) => s.comp === c)!;
  const has = (c: string) => summary.some((s) => s.comp === c);

  const show = useCallback((p: Panel) => { if (timer.current) clearTimeout(timer.current); setOpen(p); }, []);
  const hide = useCallback((delay = 140) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setOpen(null), delay);
  }, []);
  useEffect(() => { setH(open && inner.current ? inner.current.offsetHeight : 0); }, [open]);
  useEffect(() => { setOpen(null); setSheet(false); }, [path]);
  useEffect(() => { document.body.style.overflow = sheet ? "hidden" : ""; }, [sheet]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpen(null); setSheet(false); } };
    addEventListener("keydown", k); return () => removeEventListener("keydown", k);
  }, []);

  const toggleTheme = () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    const next = dark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("cadlag-theme", next); } catch {}
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", next === "dark" ? "#2A2622" : "#ECE7DD");
    reloadHeat(); window.dispatchEvent(new Event(THEME_EVENT));
  };

  const L = (c: string, i: number) => (
    <Link key={c} className="mg-link mg" style={{ ["--i" as string]: i }} href={leagueHref(c)}>
      <b>{NAMES[c]}</b><span>{fav(c).fav} {pct(fav(c).p, 0)} %</span>
    </Link>
  );
  const A = (href: string, i: number, title: string, sub: string, ext = false) => ext
    ? <a key={href + title} className="mg-link mg" style={{ ["--i" as string]: i }} href={href} target="_blank" rel="noopener"><b>{title}</b><span>{sub}</span></a>
    : <Link key={href + title} className="mg-link mg" style={{ ["--i" as string]: i }} href={href}><b>{title}</b><span>{sub}</span></Link>;
  const col = (label: string, i: number, kids: React.ReactNode) => (
    <div className="mg-col"><span className="label mg" style={{ ["--i" as string]: i }}>{label}</span>{kids}</div>
  );
  const panels: Record<Panel, React.ReactNode> = {
    ligas: <>{col("Europa", 0, EUROPE.filter(has).map((c, i) => L(c, i + 1)))}{col("Sudamérica", 1, SOUTH.filter(has).map((c, i) => L(c, i + 2)))}</>,
    copas: <>{col("Europa", 0, CUPS.filter(has).map((c, i) => L(c, i + 1)))}</>,
    modelo: <>
      {col("Cómo funciona", 0, [A("/#metodo", 1, "De los goles a la tabla", "4 pasos"), A("/#trayectoria", 2, "Una temporada", "Chile 2026")])}
      {col("Qué tan bien predice", 1, [A("/#resultados", 2, "Resultados", "contra Bet365"), A("/#resultados", 3, "Lo que no estaba", "4 hipótesis")])}
      {col("Probar", 2, [A(leagueHref("chi.1"), 3, "¿Y si…?", "laboratorio")])}
    </>,
    mundo: <>
      {col("Comparación", 0, [A("/mundo/", 1, "Fuerza de cada liga", "14 ligas"), A("/mundo/", 2, "Ranking global", `${topTeam} primero`)])}
      {col("Estructura", 1, [A("/mundo/", 2, "El puente", `λ₂ = ${fmt(lambda2, 3)}`)])}
    </>,
    proyecto: <>
      {col("Sobre", 0, [A("/#nombre", 1, "Por qué càdlàg", "el nombre"), A("/", 2, "Portada", "inicio")])}
      {col("Fuentes", 1, [A("https://www.football-data.co.uk/", 2, "football-data.co.uk", "cuotas", true), A("https://www.espn.com/soccer/", 3, "ESPN", "resultados", true)])}
    </>,
  };

  let i = 0; const it = () => ({ className: "it", style: { ["--i" as string]: i++ } });
  return (
    <>
      <header className="nav" id="nav" onPointerLeave={(e) => e.pointerType === "mouse" && hide()} onPointerEnter={() => timer.current && clearTimeout(timer.current)}>
        <div className="wrap nav-row">
          <Link className="brand" href="/" aria-label="càdlàg, inicio"><Glyph />càdlàg</Link>
          <nav className="menu" aria-label="Principal">
            {(["ligas", "copas", "modelo", "mundo", "proyecto"] as Panel[]).map((p) => (
              <button key={p} className="mi" type="button" aria-expanded={open === p}
                onPointerEnter={(e) => e.pointerType === "mouse" && show(p)}
                onClick={() => (open === p ? hide(0) : show(p))}>{p[0].toUpperCase() + p.slice(1)}</button>
            ))}
          </nav>
          <div className="nav-end">
            <button className="icon-btn" type="button" onClick={toggleTheme} aria-label="Cambiar entre tema claro y oscuro">
              <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" strokeWidth="1.5" /><path d="M10 3a7 7 0 0 1 0 14z" fill="currentColor" /></svg>
            </button>
            <button className="icon-btn burger" type="button" aria-label={sheet ? "Cerrar menú" : "Abrir menú"} aria-expanded={sheet} onClick={() => setSheet(!sheet)}>
              <span><span /><span /></span>
            </button>
          </div>
        </div>
        <div className="mega" style={{ height: h }} onClick={(e) => (e.target as HTMLElement).closest("a") && hide(0)}>
          <div className="wrap mega-in" ref={inner} key={open ?? "none"}>{open && panels[open]}</div>
        </div>
      </header>
      <div className={"veil" + (open ? " on" : "")} onClick={() => hide(0)} />
      <div className={"sheet" + (sheet ? " on" : "")} onClick={(e) => (e.target as HTMLElement).closest("a") && setSheet(false)}>
        <div className="wrap">
          <div className="grp"><span className="label">Ligas</span>
            {LEAGUES.filter((c) => has(c) && !CUPS.includes(c)).map((c) => <Link key={c} {...it()} href={leagueHref(c)}>{NAMES[c]}<span>{pct(fav(c).p, 0)} %</span></Link>)}
          </div>
          {CUPS.some(has) && <div className="grp"><span className="label">Copas</span>
            {CUPS.filter(has).map((c) => <Link key={c} {...it()} href={leagueHref(c)}>{NAMES[c]}<span>{pct(fav(c).p, 0)} %</span></Link>)}
          </div>}
          <div className="grp"><span className="label">Modelo</span>
            <Link {...it()} href="/#metodo">Cómo funciona</Link><Link {...it()} href="/#trayectoria">Una temporada</Link><Link {...it()} href="/#resultados">Resultados</Link>
          </div>
          <div className="grp"><span className="label">Más</span>
            <Link {...it()} href="/mundo/">Mundo</Link><Link {...it()} href="/#nombre">Por qué càdlàg</Link>
          </div>
        </div>
      </div>
    </>
  );
}
