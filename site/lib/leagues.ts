export const LEAGUES = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "eng.2", "chi.1", "bra.1", "arg.1", "uefa.champions"] as const;
export type Comp = (typeof LEAGUES)[number];

export const NAMES: Record<string, string> = {
  "eng.1": "Premier League", "esp.1": "LaLiga", "ita.1": "Serie A", "ger.1": "Bundesliga",
  "fra.1": "Ligue 1", "eng.2": "Championship", "chi.1": "Liga de Primera", "bra.1": "Brasileirão", "arg.1": "Liga Profesional",
  "uefa.champions": "Champions League",
};
export const ABBR: Record<string, string> = {
  "eng.1": "ENG", "esp.1": "ESP", "ita.1": "ITA", "ger.1": "ALE", "fra.1": "FRA", "eng.2": "CHA", "chi.1": "CHI", "bra.1": "BRA", "arg.1": "ARG", "uefa.champions": "UCL",
};
export const COUNTRY: Record<string, string> = {
  "eng.1": "Inglaterra", "esp.1": "España", "ita.1": "Italia", "ger.1": "Alemania", "fra.1": "Francia", "eng.2": "Inglaterra", "chi.1": "Chile", "bra.1": "Brasil", "arg.1": "Argentina", "uefa.champions": "Europa",
};
export const LNAME: Record<string, string> = {
  ...NAMES, "chi.1": "Chile", "arg.1": "Argentina", "esp.2": "LaLiga 2", "ita.2": "Serie B",
  "ger.2": "2. Bundesliga", "fra.2": "Ligue 2", "bra.2": "Série B",
  // ligas que solo alimentan la capa europea de las copas
  "por.1": "Primeira Liga", "ned.1": "Eredivisie", "tur.1": "Süper Lig", "bel.1": "Pro League belga",
  "gre.1": "Superliga griega", "aut.1": "Bundesliga austriaca", "nor.1": "Eliteserien", "sco.1": "Premiership escocesa",
  "den.1": "Superliga danesa", "sui.1": "Super League suiza", "swe.1": "Allsvenskan",
};
export const EUROPE = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "eng.2"];
export const SOUTH = ["chi.1", "bra.1", "arg.1"];
export const CUPS = ["uefa.champions"];
export const isCup = (comp: string) => CUPS.includes(comp);

export const SLUG: Record<string, string> = {
  "eng.1": "premier-league", "esp.1": "laliga", "ita.1": "serie-a", "ger.1": "bundesliga", "fra.1": "ligue-1", "eng.2": "championship", "chi.1": "chile", "bra.1": "brasileirao", "arg.1": "argentina", "uefa.champions": "champions",
};
export const COMP_OF: Record<string, string> = Object.fromEntries(Object.entries(SLUG).map(([c, s]) => [s, c]));
export const leagueHref = (comp: string) => (CUPS.includes(comp) ? `/copa/${SLUG[comp]}/` : `/liga/${SLUG[comp]}/`);

export const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
