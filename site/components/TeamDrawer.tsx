"use client";
import { useEffect, useRef, useState } from "react";
import type { League } from "@/lib/types";
import { LNAME, NAMES } from "@/lib/leagues";
import { fmt } from "@/lib/format";
import { heat } from "@/lib/heat";
import type { Extras } from "@/lib/sims";

type Need = "title" | "promotion" | "playoffs" | "survival" | "top8" | "top24" | "champion";
const STAGES = ["Fase liga", "Top 24", "Octavos", "Cuartos", "Semifinal", "Final", "Campeón"];

// Cajón de equipo: posición final, fuerza estimada y qué necesita para el título o para salvarse.
export default function TeamDrawer({ D, t, M, ex, onClose }: { D: League; t: number | null; M: Float64Array; ex: Extras; onClose: () => void }) {
  const [need, setNeed] = useState<Need>("title");
  const hasAsc = !!D.sims.asc, split = D.sims.split ?? null, cup = !!D.sims.cup;
  // objetivos según la liga: ascenso (Championship), playoffs y campeón del torneo (Argentina)
  const tabs: [Need, string][] = cup ? [["top8", "Top 8"], ["top24", "Top 24"], ["champion", "Campeón"]] : hasAsc ? [["promotion", "Ascenso"], ["title", "Título"], ["survival", "Salvarse"]]
    : split ? [["playoffs", "Playoffs"], ["title", "Campeón"], ["survival", "Salvarse"]] : [["title", "Título"], ["survival", "Salvarse"]];
  const close = useRef<HTMLButtonElement>(null);
  const open = t !== null; const n = D.order.length;
  useEffect(() => { if (open) { setNeed(tabs[0][0]); close.current?.focus({ preventScroll: true }); } }, [open, t, hasAsc, split]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { const k = (e: KeyboardEvent) => e.key === "Escape" && onClose(); addEventListener("keydown", k); return () => removeEventListener("keydown", k); }, [onClose]);

  const T = open ? D.teams.find((x) => x.id === D.order[t!])! : null;
  return (
    <>
      <div className={"scrim" + (open ? " on" : "")} onClick={onClose} />
      <aside className={"drawer" + (open ? " on" : "")} aria-hidden={!open} aria-labelledby="d-name">
        {T && (() => {
          const probs = [...Array(n).keys()].map((k) => M[t! * n + k]); const pmax = Math.max(...probs);
          const nd = T.needs || {}; const R = T.rating; const rank = D.teams.indexOf(T) + 1;
          let left = 3 * (T.remaining ?? Math.max(0, 2 * (n - 1) - T.played)); // puntos que quedan en juego
          const chips: React.ReactNode[] = [];
          if (nd.title_clinched) chips.push(<span key="c" className="chip good">Campeón asegurado</span>);
          if (nd.title_eliminated) chips.push(<span key="e" className="chip">Sin opción matemática al título</span>);
          if (nd.relegated) chips.push(<span key="r" className="chip bad">Descendido</span>);
          else if (nd.safe && rank > n / 2) chips.push(<span key="s" className="chip good">Salvado matemáticamente</span>);
          if (nd.best_position) chips.push(<span key="b" className="chip">Mejor posible: {nd.best_position}.º · peor: {nd.worst_position}.º</span>);
          const iv = (lab: string, m: number, s: number) => {
            const lo = -0.6, hi = 0.6, x = (v: number) => ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100;
            return (
              <div className="iv-row"><span>{lab}</span>
                <div className="iv-track"><span className="z" style={{ left: `${x(0)}%` }} /><span className="ci" style={{ left: `${x(m - 1.645 * s)}%`, width: `${x(m + 1.645 * s) - x(m - 1.645 * s)}%` }} /><span className="pt" style={{ left: `${x(m)}%` }} /></div>
                <span className="tnum" style={{ fontSize: 12.5, textAlign: "right" }}>{m >= 0 ? "+" : "−"}{fmt(Math.abs(m), 2)}</span>
              </div>
            );
          };
          const c = nd[need]?.curve || [];
          if (c.length) left = c[c.length - 1][0];   // en Argentina, título y playoffs cuentan solo el torneo en curso
          const n50 = nd[need]?.points_for_50 ?? null, n90 = nd[need]?.points_for_90 ?? null;
          const goal = need === "top8" ? "termina entre los 8 primeros" : need === "top24" ? "sigue en carrera (top 24)" : need === "champion" ? "gana la copa" : need === "title" ? "sale campeón" : need === "promotion" ? "asciende" : need === "playoffs" ? "clasifica a los playoffs" : "se salva";
          // objetivos que se definen en una llave (campeón del torneo argentino o de una copa): ni ganando
          // todo se llega al 50 %, así que la curva se lee contra su propio máximo
          const ko = (split && need === "title") || need === "champion";
          const pmaxC = c.length ? Math.max(...c.map((r) => r[1])) : 0;
          const now = ko ? (need === "title" ? ex.title?.[t!] : ex.champion?.[t!]) ?? 0 : 0;
          const half = ko ? c.find((r) => r[1] >= 0.5 * pmaxC)?.[0] ?? null : null;
          const sentence = () => {
            if (!c.length) return null;
            if (ko) {
              if (pmaxC < 0.0005) return "Ya no llega a la llave del título, ni ganando todo lo que le queda.";
              const where = cup ? "la copa" : `el ${split?.tournament ?? "torneo"}`;
              const first = `Ganando los ${left} puntos que le quedan, gana ${where} en el ${fmt(pmaxC * 100, 1)} % de las simulaciones (hoy, ${fmt(now * 100, 1)} %).`;
              const second = half != null && half > 0 ? ` Con ${half} puntos ya tiene la mitad de esa chance.` : "";
              return first + second + " La fase regular decide si entra a la llave, el cruce y la localía; el título se juega en las eliminatorias.";
            }
            if (n50 == null) return `Aunque gane todo lo que le queda, ${goal} en menos de la mitad de las simulaciones.`;
            const first = n50 === 0 ? `Sin sumar más puntos ya ${goal} en la mitad de las simulaciones` : `Si suma ${n50} de los ${left} puntos que le quedan, ${goal} en la mitad de las simulaciones`;
            const second = n90 == null ? "." : n90 === n50 ? ", y en nueve de cada diez." : `; con ${n90}, en nueve de cada diez.`;
            return first + second;
          };
          let chart: React.ReactNode = null;
          if (c.length) {
            const W = 440, H = 160, pad = [14, 10, 22, 42]; const xs = c.map((r) => r[0]); const x0 = Math.min(...xs), x1 = Math.max(...xs);
            // en los objetivos de llave el eje llega al máximo de la curva (redondeado a 5 puntos porcentuales)
            const top = ko ? Math.max(0.05, Math.ceil((pmaxC * 1.05) * 20) / 20) : 1;
            const X = (v: number) => pad[3] + ((v - x0) / Math.max(1, x1 - x0)) * (W - pad[1] - pad[3]), Y = (v: number) => pad[0] + (1 - v / top) * (H - pad[0] - pad[2]);
            const line = c.map((r, i) => `${i ? "L" : "M"}${X(r[0]).toFixed(1)},${Y(r[1]).toFixed(1)}`).join("");
            const at = (v: number | null) => (v == null ? null : c.find((r) => r[0] >= v) ?? null);
            const xt = [x0, Math.round((x0 + x1) / 2), x1];
            chart = (
              <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Probabilidad de lograr el objetivo según los puntos que sume" style={{ fontFamily: "var(--f-sans)" }}>
                {[0, top / 2, top].map((v) => <g key={v}><line x1={pad[3]} x2={W - pad[1]} y1={Y(v)} y2={Y(v)} stroke="var(--rule)" /><text x={pad[3] - 6} y={Y(v) + 3} textAnchor="end" fontSize="10.5" style={{ fill: "var(--ink-2)" }}>{fmt(v * 100, v * 100 % 1 ? 1 : 0)} %</text></g>)}
                {xt.map((v) => <text key={v} x={X(v)} y={H - 6} textAnchor="middle" fontSize="10.5" style={{ fill: "var(--ink-2)" }}>+{v}</text>)}
                <path d={line} fill="none" stroke="var(--beni)" strokeWidth="2" />
                {(ko ? [at(half)] : [at(n50), at(n90)]).map((r, i) => r && <circle key={i} cx={X(r[0])} cy={Y(r[1])} r="4" fill={i ? "var(--beni)" : "var(--paper)"} stroke="var(--beni)" strokeWidth="1.6" />)}
              </svg>
            );
          }
          return (
            <>
              <button className="icon-btn x" type="button" ref={close} onClick={onClose} aria-label="Cerrar">
                <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
              </button>
              <span className="label">{NAMES[D.meta.comp]} · {T.points} pts en {T.played} partidos{cup && ex.champion ? ` · ${fmt(ex.champion[t!] * 100, 1)} % de ganarla` : ""}{T.adj ? ` (descuento de ${-T.adj})` : ""}{hasAsc && ex.asc ? ` · ${fmt(ex.asc[t!] * 100, 1)} % de ascender` : ""}{split && ex.title ? ` · zona ${T.zone} · ${fmt(ex.title[t!] * 100, 1)} % de ganar el ${split.tournament}` : ""}</span>
              <h2 id="d-name" style={{ marginTop: 10 }}>{T.name}</h2>
              <div className="chips">{chips}</div>
              <div className="dsec"><span className="label">{cup ? "Puesto en la fase liga" : "Distribución de la posición final"}</span>
                <div className="dist">{probs.map((p, k) => <div key={k} title={`${k + 1}.º: ${fmt(p * 100, 1)} %`} style={{ height: `${(p / pmax) * 100}%`, animationDelay: `${k * 18}ms`, background: heat(p) }} />)}</div>
                <div className="dist-ax">{probs.map((_, k) => <span key={k}>{k + 1}</span>)}</div>
              </div>
              {cup && T.stage_probs && (() => {
                const reach = STAGES.map((_, k) => T.stage_probs!.slice(k).reduce((a, b) => a + b, 0));
                return (
                  <div className="dsec"><span className="label">Hasta dónde llega</span>
                    {STAGES.slice(1).map((lab, j) => { const v = reach[j + 1]; return (
                      <div className="iv-row" key={lab}><span>{lab}</span>
                        <div className="iv-track"><span className="ci" style={{ left: 0, width: `${v * 100}%`, background: heat(v) }} /></div>
                        <span className="tnum" style={{ fontSize: 12.5, textAlign: "right" }}>{fmt(v * 100, 1)} %</span>
                      </div>); })}
                  </div>
                );
              })()}
              <div className="dsec"><span className="label">Fuerza estimada{cup ? " (escala común: 0 es un rival promedio de la Premier)" : ""}</span>
                {iv("Ataque", R.attack, R.attack_sd)}{iv("Defensa", R.defense, R.defense_sd)}
                <p className="note" style={{ marginTop: 10 }}>
                  {cup
                    ? <>Contra un rival promedio de la Premier en cancha neutral: {fmt(R.xgf_neutral, 2)} goles a favor y {fmt(R.xga_neutral, 2)} en contra. {R.league ? `Su fuerza sale de la ${LNAME[R.league] ?? R.league} más el nivel de esa liga en Europa.` : `Su liga no está modelada: la fuerza sale solo de sus partidos europeos (${R.n_uefa ?? 0} en las últimas tres temporadas)${T.few_data ? ", así que es muy incierta" : ""}.`}</>
                    : R.sot_for != null
                    ? <>Contra un rival promedio en cancha neutral: {fmt(R.sot_for, 1)} tiros al arco con {fmt((R.conv_for ?? 0) * 100, 0)} % de conversión, {fmt(R.xgf_neutral, 2)} goles a favor; recibe {fmt(R.sot_against ?? 0, 1)} tiros al arco y le convierten el {fmt((R.conv_against ?? 0) * 100, 0)} %, {fmt(R.xga_neutral, 2)} en contra.</>
                    : <>Contra un rival promedio en cancha neutral: {fmt(R.xgf_neutral, 2)} goles a favor y {fmt(R.xga_neutral, 2)} en contra.</>}
                </p>
              </div>
              <div className="dsec"><span className="label">¿Qué necesita?</span>
                <div className="need-ctl seg" role="group" aria-label="Objetivo">
                  {tabs.map(([k, l]) => <button key={k} type="button" aria-pressed={need === k} onClick={() => setNeed(k)}>{l}</button>)}
                </div>
                {chart}
                <p className="need-txt">{sentence()}</p>
              </div>
            </>
          );
        })()}
      </aside>
    </>
  );
}
