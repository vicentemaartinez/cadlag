"use client";
import { useEffect, useState } from "react";
import { LNAME, NAMES } from "@/lib/leagues";
import { fmt, initials, pct } from "@/lib/format";
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
};

const REACH_LABEL: Record<string, string> = { playoff: "Playoff", r16: "8vos", qf: "4tos", sf: "Semi", final: "Final", champion: "Camp." };
const REACH_TITLE: Record<string, string> = {
  playoff: "Juega el playoff (segundos de la Sudamericana contra terceros de la Libertadores)", r16: "Llega a octavos",
  qf: "Llega a cuartos", sf: "Llega a semifinales", final: "Juega la final", champion: "Gana la copa",
};

function Cell({ v }: { v: number }) {
  return <td className="num tnum asc-c" style={{ background: v >= 0.0005 ? heat(v) : undefined, color: v > 0.33 ? "var(--heat-text)" : undefined }}>{v >= 0.0005 ? (v >= 0.9995 ? "100" : fmt(v * 100, 1)) : ""}</td>;
}

export default function GroupCupView({ D }: { D: GroupCup }) {
  const [, setTick] = useState(0);
  useEffect(() => onTheme(() => setTick((x) => x + 1)), []);
  const byId = new Map(D.teams.map((t) => [t.id, t]));
  const champ = (t: GroupCup["teams"][number]) => t.zones.champion ?? 0;
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
        <div className="sh"><div className="t"><h2>Hasta dónde llega</h2></div><p>En % sobre {D.n_sims.toLocaleString("es-CL")} ediciones simuladas{groupsDone ? "" : ", desde los grupos"}. La fuerza es la de la capa sudamericana: la de su liga más el nivel de esa liga en las copas.</p></div>
        <div className="mx-scroll">
          <table className="mx reach" style={{ minWidth: 470 + D.reach.length * 56 }}>
            <colgroup><col style={{ width: 26 }} /><col style={{ width: 230 }} /><col style={{ width: 56 }} /><col style={{ width: 150 }} />{D.reach.map((k) => <col key={k} />)}</colgroup>
            <thead><tr className="cols"><th /><th style={{ textAlign: "left", paddingLeft: 4 }}>Equipo</th><th>Grupo</th><th title="Liga con la que se ancla su fuerza">Liga</th>
              {D.reach.map((k) => <th key={k} className="asc-h" title={REACH_TITLE[k]}>{REACH_LABEL[k]}</th>)}</tr></thead>
            <tbody>
              {sorted.filter((t) => (t.zones[D.reach[0]] ?? 0) >= 0.0005).map((t, i) => (
                <tr key={t.id}>
                  <td className="pos tnum">{i + 1}</td>
                  <td className="team"><span className="team-btn"><span className="badge">{initials(t.name)}</span><span className="nm">{t.name}</span></span></td>
                  <td className="num" style={{ textAlign: "center" }}>{t.group ?? "—"}</td>
                  <td className="num lg">{t.rating.league ? LNAME[t.rating.league] ?? t.rating.league : "—"}</td>
                  {D.reach.map((k) => <Cell key={k} v={t.zones[k] ?? 0} />)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="wrap">
        <div className="sh"><div className="t"><h2>Grupos</h2></div><p>Puesto final en cada grupo{groupsDone ? " (ya definido)" : ""}. {D.comp.endsWith("libertadores") ? "Los dos primeros van a octavos y el tercero, al playoff de la Sudamericana." : "El primero va a octavos y el segundo, al playoff contra un tercero de la Libertadores."}</p></div>
        <div className="groups">
          {D.groups.map((g) => {
            // orden: puesto esperado (con el grupo terminado, el puesto real con los desempates del reglamento)
            const ep = (t: GroupCup["teams"][number]) => (t.group_probs ?? []).reduce((a, p, k) => a + p * (k + 1), 0);
            const rows = g.teams.map((id) => byId.get(id)!).sort((a, b) => ep(a) - ep(b) || b.points - a.points);
            return (
              <div className="grp-card" key={g.name}>
                <h3>Grupo {g.name}</h3>
                <table className="mx">
                  <colgroup><col style={{ width: "42%" }} /><col style={{ width: "9%" }} /><col style={{ width: "9%" }} /><col /><col /><col /><col /></colgroup>
                  <thead><tr className="cols"><th style={{ textAlign: "left" }}>Equipo</th><th>PJ</th><th>Pts</th>{[1, 2, 3, 4].map((k) => <th key={k} className="p">{k}</th>)}</tr></thead>
                  <tbody>
                    {rows.map((t) => (
                      <tr key={t.id}>
                        <td className="team"><span className="nm">{t.name}</span></td>
                        <td className="num tnum">{t.played}</td><td className="num pts tnum">{t.points}</td>
                        {(t.group_probs ?? [0, 0, 0, 0]).map((p, k) => <Cell key={k} v={p} />)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      </section>
    </>
  );
}
