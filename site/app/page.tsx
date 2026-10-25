import Flow from "@/components/Flow";
import Story from "@/components/Story";
import Loader from "@/components/Loader";
import { getIndex, getTrajectory } from "@/lib/data";
import { fmt } from "@/lib/format";

export default function Home() {
  const idx = getIndex(); const T = getTrajectory("chi.1");
  const last = idx.leagues.map((l) => l.last_result).filter(Boolean).sort().pop();
  const upd = last ? new Date(last).toLocaleDateString("es-CL", { day: "numeric", month: "long" }) : "—";
  const seasons = idx.leagues.reduce((a, l) => a + l.S, 0);
  const left = idx.leagues.reduce((a, l) => a + l.S * (l.matches_total - l.matches_played), 0);
  const e = idx.backtest["eng.1"];
  const m = (e?.["Dos latentes"] ?? e?.["Modelo (predictiva posterior)"])?.rps ?? 0, b = e?.["Bet365 (Shin)"].rps ?? 0, h = e?.["Frecuencias históricas"].rps ?? 0;

  return (
    <>
      <Loader seasons={seasons} matches={left} leagues={idx.leagues.length} />
      <header className="hero wrap">
        <h1 className="hero-word">càdlàg</h1>
        <div className="hero-row">
          <p className="hero-lead">Dónde termina cada equipo, simulando {(idx.leagues[0]?.S ?? 20000).toLocaleString("es-CL")} veces lo que queda de cada liga.</p>
          <p className="hero-meta">Nueve ligas{idx.leagues.some((l) => l.comp === "uefa.champions") ? " y la Champions" : ""} · actualizado el {upd}</p>
        </div>
      </header>

      <Flow leagues={idx.leagues} />

      {T && <Story T={T} />}

      <section className="wrap" id="metodo">
        <div className="sh"><div className="t"><h2>Cómo funciona</h2></div></div>
        <div className="steps">
          <div className="step"><div className="n">1</div><h3>Datos</h3><p>Resultados, minutos de gol, tarjetas y tiros de ESPN para 21 competiciones, con identificadores de equipo que cruzan ligas y copas.</p><div className="fig"><span><b>106.150</b>partidos</span><span><b>2010–2027</b>temporadas</span></div></div>
          <div className="step"><div className="n">2</div><h3>Fuerzas que caminan</h3><p>Cada equipo tiene cuatro fuerzas: crear y evitar tiros al arco, convertirlos y que no se los conviertan. Siguen un paseo aleatorio y un filtro de Laplace las actualiza partido a partido.</p><div className="fig"><span><b>4</b>fuerzas por equipo</span><span><b>16</b>hiperparámetros</span></div></div>
          <div className="step"><div className="n">3</div><h3>Temporadas simuladas</h3><p>Cada simulación sortea las fuerzas una vez, las deja evolucionar hasta el final y aplica los desempates reales de la liga.</p><div className="fig"><span><b>20.000</b>por liga</span><span><b>85 %</b>cobertura del 90 %</span></div></div>
          <div className="step"><div className="n">4</div><h3>Evaluación</h3><p>Pronósticos fuera de muestra desde 2016, contra el mercado sin margen (método de Shin) y contra las frecuencias históricas.</p><div className="fig"><span><b>{fmt(m, 4)}</b>RPS del modelo</span><span><b>{fmt(b, 4)}</b>RPS de Bet365</span></div></div>
        </div>
      </section>

      <section className="wrap" id="resultados">
        <div className="sh"><div className="t"><h2>Resultados</h2></div></div>
        <div className="results">
          <div className="res-col">
            <h3>Lo que funciona</h3>
            <ul className="rows">
              <li><b className="tnum">{fmt(m - b, 4)}</b><span>de RPS detrás de Bet365 en la Premier desde 2016. Las frecuencias históricas quedan a {fmt(h - b, 4)}.</span></li>
              <li><b className="tnum">+0,004</b><span>de log-verosimilitud por partido en las nueve ligas al separar crear tiros al arco de convertirlos.</span></li>
              <li><b className="tnum">×2,03</b><span>de varianza perdida si se fijan las fuerzas en vez de sortearlas.</span></li>
              <li><b className="tnum">+0,018</b><span>de log-verosimilitud por partido UEFA al anclar cada equipo a la fuerza que muestra en su liga.</span></li>
              <li><b className="tnum">0,017</b><span>de conectividad entre ligas: Europa y Sudamérica casi no se tocan.</span></li>
            </ul>
          </div>
          <div className="res-col">
            <h3>Lo que no estaba</h3>
            <ul className="rows neg">
              <li><b>Ciclos</b><span>de piedra, papel y tijera: HodgeRank en 108 temporadas, p = 0,81.</span></li>
              <li><b>Estilos</b><span>de juego latentes: el primer valor singular no supera al nulo.</span></li>
              <li><b>Fichajes</b><span>y gasto neto: no mejoran el pronóstico fuera de muestra.</span></li>
              <li><b>Contagio</b><span>de goles: tras un gol, ambos equipos bajan el ritmo.</span></li>
            </ul>
          </div>
        </div>
      </section>

      <section className="wrap" id="nombre">
        <div className="name-sec">
          <div className="name-fig">
            <svg viewBox="0 0 420 220" aria-hidden="true">
              <line x1="10" y1="170" x2="190" y2="170" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              <circle cx="200" cy="170" r="9" fill="var(--paper)" stroke="currentColor" strokeWidth="3" />
              <line x1="200" y1="160" x2="200" y2="62" stroke="currentColor" strokeWidth="1.2" strokeDasharray="3 5" opacity=".6" />
              <circle cx="200" cy="50" r="9" fill="currentColor" />
              <line x1="200" y1="50" x2="410" y2="50" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              <text x="10" y="205" fontSize="14" style={{ fill: "var(--ink-2)" }}>antes del gol</text>
              <text x="410" y="28" fontSize="14" textAnchor="end" style={{ fill: "var(--ink-2)" }}>después del gol</text>
            </svg>
          </div>
          <div className="name-txt">
            <h2>Por qué càdlàg</h2>
            <p className="fr">continue à droite, limite à gauche</p>
            <p>Continua por la derecha, con límite por la izquierda. Así son las probabilidades de este proyecto: quietas entre partidos y con un salto en cada resultado.</p>
          </div>
        </div>
      </section>
    </>
  );
}
