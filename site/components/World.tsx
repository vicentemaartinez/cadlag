import type { Global } from "@/lib/types";
import { LNAME } from "@/lib/leagues";
import { fmt } from "@/lib/format";

// Comparación entre ligas (render estático): intervalos por liga, grafo con el corte de Fiedler, ranking global.
export default function World({ G }: { G: Global }) {
  const F = G.fiedler; const col = (l: string) => (F[l] > 0 ? "var(--kaki)" : "var(--ai)");
  const L = G.leagues.slice().sort((a, b) => b.net - a.net); const W = 640, rh = 26, H = L.length * rh + 40, x0 = -1.6, x1 = 0.4, pl = 128, pr = 20;
  const X = (v: number) => pl + ((v - x0) / (x1 - x0)) * (W - pl - pr);
  const GW = 460, GH = 360; const nodes = Object.keys(F);
  const P: Record<string, [number, number]> = { "eng.1": [200, 110], "esp.1": [235, 190], "ita.1": [170, 250], "ger.1": [120, 175], "fra.1": [140, 95], "eng.2": [225, 40], "esp.2": [255, 275], "ita.2": [150, 322], "ger.2": [45, 200], "fra.2": [70, 55], "bra.1": [372, 150], "arg.1": [392, 245], "chi.1": [432, 190], "bra.2": [415, 85] };
  const pos: Record<string, [number, number]> = {}; nodes.forEach((k, i) => (pos[k] = P[k] || [60 + i * 25, 340]));
  const wmax = Math.max(...G.graph.map((e) => e.w));
  const R = G.ranking.slice(0, 24); const rx0 = -0.4, rx1 = 1.4; const RX = (v: number) => ((Math.max(rx0, Math.min(rx1, v)) - rx0) / (rx1 - rx0)) * 100;

  return (
    <>
      <header className="lhero wrap">
        <div className="lhero-grid">
          <div className="lead">
            <span className="label">Grafo de copas internacionales y ascensos</span>
            <h1>Comparación entre ligas</h1>
            <div className="big"><div className="pct tnum">{fmt(G.eigen[1], 3)}</div><p className="cap">de conectividad: Europa y Sudamérica se tocan por muy pocos partidos.</p></div>
          </div>
          <aside className="side">
            <div className="plate"><div className="k">λ₂ (Fiedler)</div><div className="v tnum">{fmt(G.eigen[1], 3)}</div><div className="s">contra λ₃ = {fmt(G.eigen[2], 2)}: un cuello de botella.</div></div>
            <div className="plate"><div className="k">Partidos de copa usados</div><div className="v tnum">2.208</div><div className="s">más 611 ascensos y descensos</div></div>
            <div className="plate"><div className="k">Referencia</div><div className="v">Premier League</div><div className="s">fuerza 0 por definición</div></div>
          </aside>
        </div>
      </header>
      <section className="wrap" style={{ paddingTop: 8 }}>
        <div className="world">
          <div className="panel"><h3>Fuerza neta de cada liga</h3><p className="pn">Frente a la Premier League (0), con intervalo de 90 %.</p>
            <div style={{ overflowX: "auto" }}>
              <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 520, fontFamily: "var(--f-sans)" }} role="img" aria-label="Fuerza neta de cada liga con intervalos">
                {[-1.5, -1, -0.5, 0].map((v) => <g key={v}><line x1={X(v)} x2={X(v)} y1={10} y2={H - 26} stroke="var(--rule)" /><text x={X(v)} y={H - 10} textAnchor="middle" fontSize="10.5" style={{ fill: "var(--ink-2)" }}>{fmt(v, 1)}</text></g>)}
                {L.map((l, i) => { const y = 22 + i * rh; const lo = l.net - 1.645 * l.net_sd, hi = l.net + 1.645 * l.net_sd;
                  return <g key={l.league}><text x={pl - 10} y={y + 4} textAnchor="end" fontSize="12.5">{LNAME[l.league] || l.league}</text>
                    <line x1={X(lo)} x2={X(hi)} y1={y} y2={y} stroke={col(l.league)} strokeWidth="5" strokeOpacity=".35" strokeLinecap="round" />
                    <circle cx={X(l.net)} cy={y} r="5" fill={col(l.league)} /></g>; })}
              </svg>
            </div>
          </div>
          <div className="panel"><h3>El grafo</h3><p className="pn">Ligas unidas por partidos de copa y equipos que suben o bajan.</p>
            <svg viewBox={`0 0 ${GW} ${GH}`} width="100%" role="img" aria-label="Grafo de ligas" style={{ fontFamily: "var(--f-sans)" }}>
              {G.graph.filter((e) => pos[e.a] && pos[e.b]).map((e, i) => { const cut = F[e.a] > 0 !== F[e.b] > 0;
                return <line key={i} x1={pos[e.a][0]} y1={pos[e.a][1]} x2={pos[e.b][0]} y2={pos[e.b][1]} stroke={cut ? "var(--beni)" : "var(--ink-3)"} strokeOpacity={cut ? 0.9 : 0.35} strokeWidth={0.6 + (3 * Math.log1p(e.w)) / Math.log1p(wmax)} />; })}
              {nodes.map((k) => <g key={k}><circle cx={pos[k][0]} cy={pos[k][1]} r="7" fill={col(k)} stroke="var(--paper)" strokeWidth="2" /><text x={pos[k][0]} y={pos[k][1] - 11} textAnchor="middle" fontSize="11">{LNAME[k] || k}</text></g>)}
              <text x={GW / 2} y={GH - 6} textAnchor="middle" fontSize="11" style={{ fill: "var(--beni)" }}>en rojo, las aristas que cruzan el corte: el puente</text>
            </svg>
          </div>
        </div>
        <div className="panel" style={{ marginTop: 40 }}><h3>Ranking global</h3><p className="pn">Fuerza del equipo más la de su liga, con intervalo de 90 %.</p>
          <div style={{ overflowX: "auto" }}>
            <table className="rank"><tbody>
              {R.map((r, i) => (
                <tr key={r.name + r.league}>
                  <td className="tnum" style={{ width: 34, color: "var(--ink-3)" }}>{i + 1}</td>
                  <td>{r.name}<div className="lg">{LNAME[r.league] || r.league}</div></td>
                  <td className="iv"><div className="iv-track">
                    <span className="ci" style={{ left: `${RX(r.global_net - 1.645 * r.sd)}%`, width: `${RX(r.global_net + 1.645 * r.sd) - RX(r.global_net - 1.645 * r.sd)}%`, background: `color-mix(in oklab,${col(r.league)} 35%,transparent)` }} />
                    <span className="pt" style={{ left: `${RX(r.global_net)}%`, background: col(r.league) }} />
                  </div></td>
                  <td className="tnum" style={{ textAlign: "right", width: 70 }}>{fmt(r.global_net, 2)}</td>
                </tr>
              ))}
            </tbody></table>
          </div>
        </div>
      </section>
    </>
  );
}
