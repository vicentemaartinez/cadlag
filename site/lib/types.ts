import type { SplitMeta } from "./sims";

export type Zone = { key: string; label: string; positions: number[] };

export type NeedCurve = { curve: [number, number, number][]; points_for_50: number | null; points_for_90: number | null };

export type Team = {
  id: string; name: string; played: number; won: number; drawn: number; lost: number; gf: number; ga: number;
  points: number; adj?: number; remaining?: number; zone?: string; tpts?: number; tpj?: number; xpts: number; pts_q: number[]; position_probs: number[]; expected_position: number;
  zones: Record<string, number>;
  rating: {
    attack: number; defense: number; attack_sd: number; defense_sd: number; net: number; net_sd: number; xgf_neutral: number; xga_neutral: number;
    // solo con el modelo de dos latentes (tiros al arco y conversión frente a un rival promedio)
    sot_for?: number; conv_for?: number; sot_against?: number; conv_against?: number;
    league?: string | null;   // copas: liga doméstica del equipo (null si no la modelamos)
  };
  stage_probs?: number[];     // copas: P(termina en cada etapa), de la fase liga al título
  needs: {
    title?: NeedCurve; survival?: NeedCurve; promotion?: NeedCurve; playoffs?: NeedCurve; best_position?: number; worst_position?: number;
    top8?: NeedCurve; top24?: NeedCurve; champion?: NeedCurve;
    title_clinched?: boolean; title_eliminated?: boolean; relegated?: boolean; safe?: boolean;
  };
};

export type Decisive = {
  home: string; away: string; home_name: string; away_name: string; date: string;
  p_home: number; p_draw: number; p_away: number; mi_title: number; mi_relegation: number; mi_promotion?: number;
};

export type Fixture = { home: string; away: string; date: string; p: [number, number, number] };

export type League = {
  meta: {
    comp: string; name: string; country: string; season_label: string; matches_played: number; matches_total: number;
    zones: Zone[]; suspense: { champion_entropy_bits: number; champion_max_bits: number }; last_result: string | null;
  };
  order: string[];
  teams: Team[];
  decisive: Decisive[];
  sims: { S: number; K: number; n: number; layout?: string; asc?: boolean; split?: SplitMeta; cup?: boolean; fixtures: Fixture[] };
  model?: { home_advantage: number; goal_level: number; offsets?: Record<string, { attack: number; defense: number }> };
};

export type IndexLeague = {
  comp: string; season_label: string; matches_played: number; matches_total: number; last_result: string | null; S: number; cup?: boolean;
  teams: { id: string; name: string; p: number[]; asc?: number; title?: number; champion?: number }[];
};

export type Index = {
  leagues: IndexLeague[];
  backtest: Record<string, Record<string, { rps: number; log: number; n: number }>>;
};

export type Global = {
  leagues: { league: string; net: number; net_sd: number }[];
  ranking: { name: string; league: string; global_net: number; sd: number }[];
  fiedler: Record<string, number>;
  graph: { a: string; b: string; w: number }[];
  eigen: number[];
};

export type Trajectory = {
  comp: string; syear: number; S: number; total: number;
  teams: { id: string; name: string }[];
  points: { date: string; played: number; p: number[] }[];
};
