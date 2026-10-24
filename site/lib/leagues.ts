export const LEAGUES = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "eng.2", "chi.1", "bra.1", "arg.1"] as const;
export type Comp = (typeof LEAGUES)[number];

export const NAMES: Record<string, string> = {
  "eng.1": "Premier League", "esp.1": "LaLiga", "ita.1": "Serie A", "ger.1": "Bundesliga",
  "fra.1": "Ligue 1", "eng.2": "Championship", "chi.1": "Liga de Primera", "bra.1": "Brasileirão", "arg.1": "Liga Profesional",
};
export const ABBR: Record<string, string> = {
  "eng.1": "ENG", "esp.1": "ESP", "ita.1": "ITA", "ger.1": "ALE", "fra.1": "FRA", "eng.2": "CHA", "chi.1": "CHI", "bra.1": "BRA", "arg.1": "ARG",
};
export const COUNTRY: Record<string, string> = {
  "eng.1": "Inglaterra", "esp.1": "España", "ita.1": "Italia", "ger.1": "Alemania", "fra.1": "Francia", "eng.2": "Inglaterra", "chi.1": "Chile", "bra.1": "Brasil", "arg.1": "Argentina",
};
export const LNAME: Record<string, string> = {
  ...NAMES, "chi.1": "Chile", "arg.1": "Argentina", "esp.2": "LaLiga 2", "ita.2": "Serie B",
  "ger.2": "2. Bundesliga", "fra.2": "Ligue 2", "bra.2": "Série B",
};
export const EUROPE = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "eng.2"];
export const SOUTH = ["chi.1", "bra.1", "arg.1"];

export const SLUG: Record<string, string> = {
  "eng.1": "premier-league", "esp.1": "laliga", "ita.1": "serie-a", "ger.1": "bundesliga", "fra.1": "ligue-1", "eng.2": "championship", "chi.1": "chile", "bra.1": "brasileirao", "arg.1": "argentina",
};
export const COMP_OF: Record<string, string> = Object.fromEntries(Object.entries(SLUG).map(([c, s]) => [s, c]));
export const leagueHref = (comp: string) => `/liga/${SLUG[comp]}/`;

export const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
