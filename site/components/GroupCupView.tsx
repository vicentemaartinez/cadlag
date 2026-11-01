"use client";
import { useEffect, useMemo, useState } from "react";
import { LNAME, NAMES } from "@/lib/leagues";
import { dateShort, fmt, fmtInt, initials, pct } from "@/lib/format";
import { groupScenario, loadGroupSims, type GroupSim, type GroupSimsMeta } from "@/lib/groupsims";
import { heat, onTheme } from "@/lib/heat";
import Roll from "./Roll";

// Copas con fase de grupos (Libertadores, Sudamericana): llaves abiertas, grupos y hasta dónde llega cada equipo.
export type GroupCup = {
  comp: string; name: string; season_label: string; kind: "groups"; n_sims: number;
  matches_played: number; matches_total: number; last_result: string | null;
  stages: string[]; reach: string[];
  suspense: { champion_entropy_bits: number; champion_max_bits: number };
  groups: { name: string; teams: string[] }[];
  ties: { round: string; home: string; away: string; home_name: string; away_name: string; legs: (number[] | null)[]; done: boolean; p_home: number }[];
  teams: {
    id: string; name: string; group: string | null; group_probs: number[] | null; entry: number;
    stage_probs: number[]; zones: Record<string, number>; played: number; won: number; drawn: number; lost: number;
    gf: number; ga: number; points: number;
    rating: { net: number; net_sd: number; league: string | null };
  }[];
  sims?: GroupSimsMeta;   // mientras quedan partidos de grupo: simulaciones para el laboratorio
};

// etapa (como en cadlag.cup_conmebol.REACH) desde la que cuenta cada columna
const REACH_AT: Record<string, Record<string, number>> = {
  "conmebol.libertadores": { r16: 1, qf: 2, sf: 3, final: 4, champion: 5 },
  "conmebol.sudamericana": { playoff: 1, r16: 2, qf: 3, sf: 4, final: 5, champion: 6 },
};

const REACH_LABEL: Record<string, string> = { playoff: "Playoff", r16: "8vos", qf: "4tos", sf: "Semi", final: "Final", champion: "Camp." };
const REACH_TITLE: Record<string, string> = {
  playoff: "Juega el playoff (segundos de la Sudamericana contra terceros de la Libertadores)", r16: "Llega a octavos",
  qf: "Llega a cuartos", sf: "Llega a semifinales", final: "Juega la final", champion: "Gana la copa",
};

function Cell({ v }: { v: number }) {
  return <td className="num tnum asc-c" style={{ background: v >= 0.0005 ? heat(v) : undefined, color: v > 0.33 ? "var(--heat-text)" : undefined }}>{v >= 0.0005 ? (v >= 0.9995 ? "100" : fmt(v * 100, 1)) : ""}</td>;
}

type Team = GroupCup["teams"][number];

export default function GroupCupView({ D }: { D: GroupCup }) {
  const [, setTick] = useState(0);
  useEffect(() => onTheme(() => setTick((x) => x + 1)), []);
  const byId = new Map(D.teams.map((t) => [t.id, t]));
  const champ = (t: Team) => t.zones.champion ?? 0;

  // --- laboratorio (solo mientras quedan partidos de grupo)
  const L = D.sims && D.sims.K > 0 ? D.sims : null;
  const [sim, setSim] = useState<GroupSim | null>(null);
  const [fixed, setFixed] = useState<Record<number, number>>({});
  useEffect(() => { setSim(null); setFixed({}); if (L) loadGroupSims(L, D.comp).then(setSim).catch(() => setSim(null)); }, [L, D.comp]);
  const nf = Object.keys(fixed).length;
  const base = useMemo(() => (sim ? groupScenario(sim, {}) : null), [sim]);
  const cur = useMemo(() => (sim && nf ? groupScenario(sim, fixed) : null), [sim, fixed, nf]);
  const ix = useMemo(() => new Map((L?.order ?? []).map((id, i) => [id, i])), [L]);
  const at = REACH_AT[D.comp] ?? {};
  const gprob = (t: Team) => (cur ? [0, 1, 2, 3].map((k) => cur.G[ix.get(t.id)! * 4 + k]) : t.group_probs ?? [0, 0, 0, 0]);
  const zone = (t: Team, k: string) => (cur && ix.has(t.id) ? cur.reach[ix.get(t.id)! * 8 + at[k] + 1] : t.zones[k] ?? 0);
  const dzone = (t: Team, k: string) => (cur && base && ix.has(t.id) ? cur.reach[ix.get(t.id)! * 8 + at[k] + 1] - base.reach[ix.get(t.id)! * 8 + at[k] + 1] : 0);
  const pick = (k: number, o: number) => {
    if (!sim) return;
    const next = { ...fixed }; if (next[k] === o) delete next[k]; else next[k] = o;
    if (groupScenario(sim, next).cnt === 0) return;   // combinación imposible en el modelo
    setFixed(next);
  };
  const favAll = () => { if (!L || !sim) return; const f: Record<number, number> = {}; L.fixtures.forEach((x, k) => (f[k] = x.p[0] >= x.p[2] ? 0 : 2)); if (groupScenario(sim, f).cnt) setFixed(f); };
  const nm = (id: string) => byId.get(id)?.name ?? id;
  const sorted = [...D.teams].sort((a, b) => champ(b) - champ(a) || (b.zones[D.reach[0]] ?? 0) - (a.zones[D.reach[0]] ?? 0));
  const fav = sorted[0];
  const open = D.ties.filter((t) => !t.done);
  const H = D.suspense.champion_entropy_bits, Hm = D.suspense.champion_max_bits;
  const groupsDone = D.matches_played >= D.matches_total;
  return (
    <>
      <header className="lhero wrap">
        <div className="lhero-grid">
          <div className="lead">
            <span className="label">{NAMES[D.comp]} · Sudamérica · {D.season_label}</span>
            <h1>{fav.name} <span className="soft">{champ(fav) >= 0.5 ? `va camino a ganar la ${D.name.replace("Copa ", "")}` : `es el favorito a la ${D.name.replace("Copa ", "")}`}</span></h1>
            <div className="big">
              <div className="pct tnum"><Roll value={pct(champ(fav))} /><small>%</small></div>
              <p className="cap">{champ(fav) >= 0.5 ? "de probabilidad de ganarla." : "de ganarla; las llaves reparten mucho."}</p>
            </div>
          </div>
          <aside className="side">
            <div className="plate"><div className="k">Incertidumbre del campeón</div><div className="v tnum">{fmt(H, 2)} <i>de {fmt(Hm, 0)} bits</i></div><div className="meter"><span style={{ transform: `scaleX(${H / Hm})` }} /></div><div className="s">0 es campeón decidido.</div></div>
            <div className="plate"><div className="k">{open.length ? `Llaves abiertas · ${open[0].round}` : "Fase de grupos"}</div><div className="v tnum">{open.length || D.matches_total - D.matches_played} <i>{open.length ? (open.length === 1 ? "llave" : "llaves") : "partidos por jugar"}</i></div><div className="s tnum">{D.matches_played} de {D.matches_total} partidos de grupos jugados</div></div>
          </aside>
        </div>
      </header>

      {D.ties.length > 0 && (
        <section className="wrap" style={{ paddingTop: 8 }}>
          <div className="sh"><div className="t"><h2>Llaves</h2></div><p>Probabilidad de pasar de cada lado, con lo ya jugado. Las llaves a dos partidos van directo a penales; la final es a partido único, con alargue.</p></div>
          <div className="ties">
            {[...D.ties].reverse().filter((t) => !t.done || D.ties.every((x) => x.done)).map((t, i) => (
              <div className={"tie" + (t.done ? " done" : "")} key={i}>
                <div className="tie-r">{t.round}</div>
                {[[t.home_name, t.p_home], [t.away_name, 1 - t.p_home]].map(([nm, p], j) => (
                  <div className="tie-t" key={j}>
                    <span className="nm">{nm as string}</span>
                    <span className="bar"><span style={{ transform: `scaleX(${p as number})`, background: heat(p as number) }} /></span>
                    <span className="tnum pp">{t.done ? ((p as number) > 0.5 ? "pasó" : "") : `${fmt((p as number) * 100, 1)} %`}</span>
                  </div>
                ))}
                <div className="tie-l tnum">{t.legs.filter(Boolean).map((l) => `${l![0]}-${l![1]}`).join(" · ") || "por jugar"}</div>
              </div>
            ))}
          </div>
          {D.ties.some((t) => t.done) && D.ties.some((t) => !t.done) && (
            <details className="ties-done"><summary>Llaves ya jugadas ({D.ties.filter((t) => t.done).length})</summary>
              <ul>{[...D.ties].reverse().filter((t) => t.done).map((t, i) => {
                const w = t.p_home > 0.5 ? t.home_name : t.away_name;
                return <li key={i}><span className="r">{t.round}</span> {t.home_name} – {t.away_name} <span className="tnum">{t.legs.filter(Boolean).map((l) => `${l![0]}-${l![1]}`).join(" · ")}</span> · pasó <b>{w}</b></li>;
              })}</ul>
            </details>
          )}
        </section>
      )}

      <section className="wrap">
        <div className="sh"><div className="t"><h2>Hasta dónde llega</h2></div><p>{cur ? `Con los ${nf} resultado${nf > 1 ? "s" : ""} fijado${nf > 1 ? "s" : ""} en el laboratorio; junto al nombre, el cambio en la primera columna.` : `En % sobre ${D.n_sims.toLocaleString("es-CL")} ediciones simuladas${groupsDone ? "" : ", desde los grupos"}.`} La fuerza es la de la capa sudamericana: la de su liga más el nivel de esa liga en las copas.</p></div>
        <div className="mx-scroll">
          <table className="mx reach" style={{ minWidth: 470 + D.reach.length * 56 }}>
            <colgroup><col style={{ width: 26 }} /><col style={{ width: 230 }} /><col style={{ width: 56 }} /><col style={{ width: 150 }} />{D.reach.map((k) => <col key={k} />)}</colgroup>
            <thead><tr className="cols"><th /><th style={{ textAlign: "left", paddingLeft: 4 }}>Equipo</th><th>Grupo</th><th title="Liga con la que se ancla su fuerza">Liga</th>
              {D.reach.map((k) => <th key={k} className="asc-h" title={REACH_TITLE[k]}>{REACH_LABEL[k]}</th>)}</tr></thead>
            <tbody>
              {sorted.filter((t) => (t.zones[D.reach[0]] ?? 0) >= 0.0005).map((t, i) => (
                <tr key={t.id}>
                  <td className="pos tnum">{i + 1}</td>
                  <td className="team"><span className="team-btn"><span className="badge">{initials(t.name)}</span><span className="nm">{t.name}</span>{(() => {
                    const dv = dzone(t, D.reach[0]);
                    return Math.abs(dv) >= 0.005 ? <span className={"gdelta " + (dv > 0 ? "up" : "down")}>{dv > 0 ? "+" : "−"}{fmt(Math.abs(dv) * 100, 1)}</span> : null;
                  })()}</span></td>
                  <td className="num" style={{ textAlign: "center" }}>{t.group ?? "—"}</td>
                  <td className="num lg">{t.rating.league ? LNAME[t.rating.league] ?? t.rating.league : "—"}</td>
                  {D.reach.map((k) => <Cell key={k} v={zone(t, k)} />)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="wrap" id="grupos">
        <div className="sh"><div className="t"><h2>Grupos</h2></div><p>Puesto final en cada grupo{groupsDone ? " (ya definido)" : ""}. {D.comp.endsWith("libertadores") ? "Los dos primeros van a octavos y el tercero, al playoff de la Sudamericana." : "El primero va a octavos y el segundo, al playoff contra un tercero de la Libertadores."}{L ? " Fija resultados en el laboratorio y los grupos y la tabla de arriba se recalculan." : ""}</p></div>
        <div className={L ? "board" : undefined}>
          <div className="groups">
            {D.groups.map((g) => {
              // orden: puesto esperado base (con el grupo terminado, el puesto real con los desempates del reglamento)
              const ep = (t: Team) => (t.group_probs ?? []).reduce((a, p, k) => a + p * (k + 1), 0);
              const rows = g.teams.map((id) => byId.get(id)!).sort((a, b) => ep(a) - ep(b) || b.points - a.points);
              const hot = cur && L ? L.fixtures.some((f, k) => fixed[k] !== undefined && g.teams.includes(f.home)) : false;
              return (
                <div className={"grp-card" + (hot ? " hot" : "")} key={g.name}>
                  <h3>Grupo {g.name}</h3>
                  <table className="mx">
                    <colgroup><col style={{ width: "42%" }} /><col style={{ width: "9%" }} /><col style={{ width: "9%" }} /><col /><col /><col /><col /></colgroup>
                    <thead><tr className="cols"><th style={{ textAlign: "left" }}>Equipo</th><th>PJ</th><th>Pts</th>{[1, 2, 3, 4].map((k) => <th key={k} className="p">{k}</th>)}</tr></thead>
                    <tbody>
                      {rows.map((t) => (
                        <tr key={t.id}>
                          <td className="team"><span className="nm">{t.name}</span></td>
                          <td className="num tnum">{t.played}</td><td className="num pts tnum">{t.points}</td>
                          {gprob(t).map((p, k) => <Cell key={k} v={p} />)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })}
          </div>

          {L && (
            <aside className="lab" aria-labelledby="lab-title">
              <div className="lab-head">
                <h3 id="lab-title">¿Y si…?</h3>
                <p>Fija resultados de la próxima fecha.</p>
                <div className="counter"><span>Muestra efectiva</span><span className="tnum"><b>{fmtInt(cur ? cur.ess : L.S)}</b> de {fmtInt(L.S)}</span></div>
                <div className="sample-bar"><span style={{ transform: `scaleX(${Math.min(1, (cur ? cur.ess : L.S) / L.S)})`, backgroundColor: cur && cur.ess < 400 ? "var(--beni)" : undefined }} /></div>
                <div className={"warn" + (cur && cur.ess < 400 ? " on" : "")}>Muestra efectiva bajo 400: resultados ruidosos.</div>
              </div>
              <ul className="fx-list">
                {L.fixtures.map((f, k) => {
                  const here = fixed[k] !== undefined;
                  const pr = cur ? [cur.probs[3 * k], cur.probs[3 * k + 1], cur.probs[3 * k + 2]] : f.p;
                  const hn = nm(f.home), an = nm(f.away);
                  const grp = byId.get(f.home)?.group;
                  return (
                    <li className="fx" key={k}>
                      <div className="fx-top"><div className="fx-teams"><span>{hn}</span><span style={{ color: "var(--ink-3)" }}>–</span><span>{an}</span></div><span className="when">{grp ? `Gr. ${grp} · ` : ""}{dateShort(f.date, { weekday: "short", day: "numeric", month: "short" })}</span></div>
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
          )}
        </div>
      </section>
    </>
  );
}
