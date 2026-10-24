import type { Metadata } from "next";
import LeagueTabs from "@/components/LeagueTabs";
import LeagueView from "@/components/LeagueView";
import { availableLeagues, getLeague } from "@/lib/data";
import { COMP_OF, NAMES, SLUG } from "@/lib/leagues";

export const dynamicParams = false;
export function generateStaticParams() { return availableLeagues().map((c) => ({ slug: SLUG[c] })); }

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const comp = COMP_OF[(await params).slug]; const D = getLeague(comp);
  let b = D.teams[0]; for (const t of D.teams) if (t.position_probs[0] > b.position_probs[0]) b = t;
  return { title: NAMES[comp], description: `${NAMES[comp]} ${D.meta.season_label}: ${b.name} sale campeón en el ${Math.round(b.position_probs[0] * 100)} % de ${D.sims.S.toLocaleString("es-CL")} temporadas simuladas.` };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const comp = COMP_OF[(await params).slug]; const D = getLeague(comp);
  return (<><LeagueTabs current={comp} leagues={availableLeagues()} /><LeagueView key={comp} D={D} /></>);
}
