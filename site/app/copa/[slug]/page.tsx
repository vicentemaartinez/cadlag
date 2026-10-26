import type { Metadata } from "next";
import LeagueTabs from "@/components/LeagueTabs";
import LeagueView from "@/components/LeagueView";
import { availableLeagues, getLeague } from "@/lib/data";
import { COMP_OF, NAMES, SLUG, isCup } from "@/lib/leagues";

export const dynamicParams = false;
// Si ninguna copa tiene datos publicados todavía, se genera igual la página de la Champions (con un aviso),
// porque la exportación estática no admite una ruta dinámica sin páginas.
export function generateStaticParams() {
  const cups = availableLeagues().filter(isCup);
  return (cups.length ? cups : ["uefa.champions"]).map((c) => ({ slug: SLUG[c] }));
}
const has = (comp: string) => availableLeagues().includes(comp);

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const comp = COMP_OF[(await params).slug];
  if (!has(comp)) return { title: NAMES[comp] };
  const D = getLeague(comp);
  const v = (t: (typeof D.teams)[number]) => (D.sims.cup ? t.zones.champion ?? 0 : t.position_probs[0]);
  let b = D.teams[0]; for (const t of D.teams) if (v(t) > v(b)) b = t;
  return { title: NAMES[comp], description: `${NAMES[comp]} ${D.meta.season_label}: ${b.name} sale campeón en el ${Math.round(v(b) * 100)} % de ${D.sims.S.toLocaleString("es-CL")} ${D.sims.cup ? "ediciones" : "temporadas"} simuladas.` };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const comp = COMP_OF[(await params).slug];
  if (!has(comp)) return (<section className="wrap" style={{ padding: "120px 0" }}><h1>{NAMES[comp]}</h1><p>El pronóstico aparece con la próxima actualización de los datos.</p></section>);
  const D = getLeague(comp);
  return (<><LeagueTabs current={comp} leagues={availableLeagues().filter(isCup)} /><LeagueView key={comp} D={D} /></>);
}
