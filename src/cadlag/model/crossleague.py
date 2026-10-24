r"""Comparación entre ligas: offsets de liga estimados en el grafo de partidos internacionales.

Planteamiento
-------------
Cada filtro de liga entrega fuerzas *relativas a su liga* (gauge: sum alpha = sum delta = 0).
Para comparar equipos de ligas distintas se agrega, por liga ℓ y temporada s, un offset de
ataque A_{ℓ,s} y de defensa D_{ℓ,s}. Un partido de copa entre i ∈ ℓ (local) y j ∈ ℓ' observa

    log lam = mu_c + eta (1 - neutral) + (alpha_i + A_ℓ) - (delta_j + D_ℓ')
    log nu  = mu_c +                    (alpha_j + A_ℓ') - (delta_i + D_ℓ)

con alpha, delta tomados del filtro de cada liga en la fecha del partido y mu_c el nivel de
goles de la copa c. Los offsets siguen un paseo aleatorio entre temporadas,
theta_{ℓ,s} = theta_{ℓ,s-1} + N(0, tau^2 I), y la Premier League es la referencia (offset 0).

Además, los equipos que **cambian de división** (ascensos y descensos, misma id de ESPN) son
aristas gaussianas: su fuerza neta (alpha+delta) al final de la temporada s en ℓ y al final de
s+1 en ℓ' difieren en r_{ℓ',s+1} - r_{ℓ,s} más ruido (cambio real del plantel + incertidumbre
de los filtros). Así las segundas divisiones quedan conectadas con las primeras.

Inferencia: aproximación de Laplace (Newton sobre el modo, Hessiano como precisión).

Lectura eléctrica
-----------------
En la aproximación gaussiana la precisión posterior de las fuerzas netas r = A + D tiene la
forma H = L_w + (cadena temporal) + (anclaje de la referencia), donde L_w es el laplaciano del
grafo de ligas con conductancia w_{ℓℓ'} = información de Fisher acumulada en sus partidos. Por
lo tanto

    Var(r_ℓ - r_ℓ') = (e_ℓ - e_ℓ')^T H^{-1} (e_ℓ - e_ℓ') = R_eff(ℓ, ℓ'),

la resistencia efectiva entre ℓ y ℓ' en la red eléctrica con esas conductancias. Agregar
partidos (o transferencias, o priors de mercado) equivale a agregar conductores en paralelo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..leagues import CUP_COMPETITIONS

REFERENCE = "eng.1"


def league_season(comp: str, date: pd.Timestamp) -> int:
    south = comp.split(".")[0] in ("chi", "bra", "arg")
    if south:
        return date.year
    return date.year if date.month >= 7 else date.year - 1


# ----------------------------------------------------------------------------------------
# Estados de los filtros por equipo y fecha
# ----------------------------------------------------------------------------------------
@dataclass
class TeamStates:
    """Trayectoria filtrada de cada equipo en cada liga."""
    table: pd.DataFrame  # columnas: comp, team, date, a, d, va, vd, mu

    def at(self, team: str, date: pd.Timestamp, max_days: int = 400):
        t = self.table
        g = self._by_team.get(team)
        if g is None:
            return None
        i = np.searchsorted(g["date"].to_numpy(), date.to_datetime64(), side="left") - 1
        if i < 0:
            return None
        row = g.iloc[i]
        if (date - row["date"]).days > max_days:
            return None
        return row

    def __post_init__(self):
        t = self.table.copy()
        t["date"] = t["date"].dt.tz_convert(None)
        self.table = t.sort_values("date")
        self._by_team = {k: g.reset_index(drop=True) for k, g in self.table.groupby("team")}


def collect_states(models: dict) -> TeamStates:
    """``models``: {comp: DynamicDC}. Extrae (alpha, delta, var) de cada equipo tras cada partido."""
    rows = []
    for comp, mod in models.items():
        ld = mod.ld
        _, _, o = mod.run(states=True)
        m, pdg = o["m"], o["Pdiag"]
        N = ld.N
        d = ld.matches
        for side in ("hi", "ai"):
            idx = d[side].to_numpy()
            rows.append(pd.DataFrame(dict(
                comp=comp, team=[ld.team_ids[i] for i in idx], date=d["date"].to_numpy(),
                syear=d["syear"].to_numpy(),
                a=m[np.arange(len(d)), idx], d=m[np.arange(len(d)), N + idx],
                va=pdg[np.arange(len(d)), idx], vd=pdg[np.arange(len(d)), N + idx],
                mu=m[:, 2 * N])))
    t = pd.concat(rows, ignore_index=True)
    t["date"] = pd.to_datetime(t["date"], utc=True)
    return TeamStates(t)


# ----------------------------------------------------------------------------------------
# Datos del grafo
# ----------------------------------------------------------------------------------------
def cup_edges(matches: pd.DataFrame, states: TeamStates) -> pd.DataFrame:
    d = matches[matches["comp"].isin(CUP_COMPETITIONS) & matches["finished"]].copy()
    d["day"] = d["date"].dt.tz_convert(None)
    out = []
    for r in d.itertuples():
        sh, sa = states.at(r.home, r.day), states.at(r.away, r.day)
        if sh is None or sa is None:
            continue
        out.append(dict(id=r.id, cup=r.comp, date=r.day, neutral=bool(r.neutral), hg=r.hg, ag=r.ag,
                        lh=sh["comp"], la=sa["comp"],
                        sh=league_season(sh["comp"], r.day), sa=league_season(sa["comp"], r.day),
                        ah=sh["a"], dh=sh["d"], aa=sa["a"], da=sa["d"],
                        vh=sh["va"] + sh["vd"], va=sa["va"] + sa["vd"]))
    return pd.DataFrame(out)


def mover_edges(states: TeamStates) -> pd.DataFrame:
    """Equipos que pasan de una liga a otra entre temporadas consecutivas (ascensos/descensos)."""
    t = states.table
    last = t.sort_values("date").groupby(["team", "comp", "syear"]).tail(1)
    last = last.assign(ls=[league_season(c, d) for c, d in zip(last["comp"], last["date"])])
    out = []
    for team, g in last.groupby("team"):
        g = g.sort_values("date")
        rows = g.to_dict("records")
        for a, b in zip(rows[:-1], rows[1:]):
            if a["comp"] == b["comp"]:
                continue
            gap = (b["date"] - a["date"]).days
            if gap > 500:
                continue
            out.append(dict(team=team, l1=a["comp"], s1=a["ls"], l2=b["comp"], s2=b["ls"],
                            net1=a["a"] + a["d"], net2=b["a"] + b["d"],
                            v=a["va"] + a["vd"] + b["va"] + b["vd"]))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------------------
@dataclass
class CrossLeagueFit:
    keys: list            # [(liga, temporada)]
    theta: np.ndarray     # (K, 2) offsets (A, D)
    cov: np.ndarray       # (2K, 2K) covarianza posterior (Laplace)
    cups: dict            # nivel de goles por copa y localía
    edges: pd.DataFrame
    movers: pd.DataFrame
    tau: float
    log_evidence: float
    theta_full: np.ndarray | None = None
    kix: dict | None = None
    cix: dict | None = None
    kappa: float = 1.0
    beta: float = 0.0

    def loglik(self, edges: pd.DataFrame) -> np.ndarray:
        """Log-verosimilitud Poisson de partidos de copa (para validación cruzada)."""
        th, K, P = self.theta_full, len(self.keys), len(self.theta_full)
        out = []
        from scipy.special import gammaln
        for r in edges.itertuples():
            kh, ka, c = self.kix.get((r.lh, r.sh)), self.kix.get((r.la, r.sa)), self.cix.get(r.cup)
            if kh is None or ka is None or c is None:
                out.append(np.nan); continue
            lh = np.exp(th[2 * kh] - th[2 * ka + 1] + th[2 * K + c] + (0 if r.neutral else th[P - 1]) + r.ah - r.da)
            la = np.exp(th[2 * ka] - th[2 * kh + 1] + th[2 * K + c] + r.aa - r.dh)
            out.append(r.hg * np.log(lh) - lh - gammaln(r.hg + 1) + r.ag * np.log(la) - la - gammaln(r.ag + 1))
        return np.array(out)

    def net(self) -> pd.DataFrame:
        K = len(self.keys)
        r = self.theta.sum(1)
        v = np.array([self.cov[2 * k, 2 * k] + self.cov[2 * k + 1, 2 * k + 1] + 2 * self.cov[2 * k, 2 * k + 1]
                      for k in range(K)])
        return pd.DataFrame(dict(league=[k[0] for k in self.keys], season=[k[1] for k in self.keys],
                                 attack=self.theta[:, 0], defense=self.theta[:, 1], net=r, net_sd=np.sqrt(v)))

    def resistance(self, season_of: dict) -> pd.DataFrame:
        """Matriz de Var(r_ℓ - r_ℓ') (= resistencia efectiva) para las temporadas indicadas."""
        idx = {k: i for i, k in enumerate(self.keys)}
        L = list(season_of)
        R = np.full((len(L), len(L)), np.nan)
        for a, la in enumerate(L):
            for b, lb in enumerate(L):
                ka, kb = (la, season_of[la]), (lb, season_of[lb])
                if ka not in idx or kb not in idx:
                    continue
                u = np.zeros(2 * len(self.keys))
                u[2 * idx[ka]:2 * idx[ka] + 2] += 1
                u[2 * idx[kb]:2 * idx[kb] + 2] -= 1
                R[a, b] = u @ self.cov @ u
        return pd.DataFrame(R, index=L, columns=L)


def transfer_pairs(tr: pd.DataFrame) -> pd.DataFrame:
    """Agrega traspasos definitivos entre ligas del grafo: por par ordenado de (liga, temporada)
    cuenta cuántos jugadores fueron de A a B y de B a A."""
    t = tr[tr["from_comp"].notna() & tr["to_comp"].notna() & (tr["from_comp"] != tr["to_comp"])]
    rows = {}
    for r in t.itertuples():
        a, b = (r.from_comp, int(r.from_season)), (r.to_comp, int(r.to_season))
        key = (a, b) if a < b else (b, a)
        cur = rows.setdefault(key, [0, 0])
        cur[0 if a < b else 1] += 1          # [A→B, B→A] con A < B
    return pd.DataFrame([dict(la=k[0][0], sa=k[0][1], lb=k[1][0], sb=k[1][1], n_ab=v[0], n_ba=v[1]) for k, v in rows.items()])


def fit(edges: pd.DataFrame, movers: pd.DataFrame, tau: float = 0.08, mover_sd: float = 0.12,
        prior_sd: float = 1.0, iters: int = 30, transfers: pd.DataFrame | None = None,
        beta: float = 0.0, kappa: float = 1.0) -> CrossLeagueFit:
    """``transfers`` (de ``transfer_pairs``) agrega comparaciones logísticas sobre las mismas fuerzas
    netas: P(un jugador va de A a B | traspaso entre A y B) = σ(β (r_B − r_A)), con la verosimilitud
    dividida por κ (cuasi-verosimilitud: los traspasos no son independientes ni puramente deportivos)."""
    # índices de (liga, temporada)
    keys = set(zip(edges["lh"], edges["sh"])) | set(zip(edges["la"], edges["sa"]))
    if len(movers):
        keys |= set(zip(movers["l1"], movers["s1"])) | set(zip(movers["l2"], movers["s2"]))
    # completar las cadenas temporales de cada liga
    by_league = {}
    for l, s in keys:
        by_league.setdefault(l, set()).add(s)
    keys = []
    for l, ss in sorted(by_league.items()):
        keys += [(l, s) for s in range(min(ss), max(ss) + 1)]
    kix = {k: i for i, k in enumerate(keys)}
    K = len(keys)
    cups = sorted(edges["cup"].unique())
    cix = {c: i for i, c in enumerate(cups)}
    P = 2 * K + len(cups) + 1  # offsets (A, D) + nivel por copa + localía

    # ---- prior: paseo aleatorio por liga, referencia anclada, prior débil en el inicio ----
    Q = np.zeros((P, P))
    for l in by_league:
        ks = [kix[k] for k in keys if k[0] == l]
        for a, b in zip(ks[:-1], ks[1:]):
            for c in range(2):
                i, j = 2 * a + c, 2 * b + c
                Q[i, i] += 1 / tau ** 2; Q[j, j] += 1 / tau ** 2
                Q[i, j] -= 1 / tau ** 2; Q[j, i] -= 1 / tau ** 2
        for c in range(2):
            w = 1e8 if l == REFERENCE else 1 / prior_sd ** 2
            Q[2 * ks[0] + c, 2 * ks[0] + c] += w
            if l == REFERENCE:  # toda la cadena de la referencia fija en 0
                for k in ks:
                    Q[2 * k + c, 2 * k + c] += 1e8
    Q[2 * K:, 2 * K:] += np.eye(len(cups) + 1) / 10.0 ** 2

    # ---- aristas de movimiento entre divisiones: gaussianas sobre la fuerza neta ----------
    Gm = np.zeros((P, P))
    gm = np.zeros(P)
    for r in movers.itertuples():
        k1, k2 = kix.get((r.l1, r.s1)), kix.get((r.l2, r.s2))
        if k1 is None or k2 is None:
            continue
        u = np.zeros(P)
        u[2 * k1:2 * k1 + 2] += 1
        u[2 * k2:2 * k2 + 2] -= 1
        # net1 + r1 ≈ net2 + r2   =>   u·theta ≈ net2 - net1
        w = 1.0 / (r.v + mover_sd ** 2)
        Gm += w * np.outer(u, u)
        gm += w * u * (r.net2 - r.net1)

    # ---- traspasos: comparaciones logísticas sobre las fuerzas netas -------------------
    Ut, nab, ntot = np.zeros((0, P)), np.zeros(0), np.zeros(0)
    if transfers is not None and beta > 0 and len(transfers):
        tt = [r for r in transfers.itertuples() if (r.la, r.sa) in kix and (r.lb, r.sb) in kix]
        Ut = np.zeros((len(tt), P))
        for e, r in enumerate(tt):
            ka_, kb_ = kix[(r.la, r.sa)], kix[(r.lb, r.sb)]
            Ut[e, 2 * kb_:2 * kb_ + 2] += 1; Ut[e, 2 * ka_:2 * ka_ + 2] -= 1   # r_B − r_A
        nab = np.array([r.n_ab for r in tt], float); ntot = np.array([r.n_ab + r.n_ba for r in tt], float)

    # ---- partidos de copa: Poisson con offsets ----------------------------------------
    E = edges
    nE = len(E)
    Xh = np.zeros((nE, P)); Xa = np.zeros((nE, P))
    for e, r in enumerate(E.itertuples()):
        kh, ka = kix[(r.lh, r.sh)], kix[(r.la, r.sa)]
        c = 2 * K + cix[r.cup]
        Xh[e, 2 * kh] += 1; Xh[e, 2 * ka + 1] -= 1; Xh[e, c] += 1; Xh[e, P - 1] += 0 if r.neutral else 1
        Xa[e, 2 * ka] += 1; Xa[e, 2 * kh + 1] -= 1; Xa[e, c] += 1
    off_h = (E["ah"] - E["da"]).to_numpy()
    off_a = (E["aa"] - E["dh"]).to_numpy()
    yh, ya = E["hg"].to_numpy(float), E["ag"].to_numpy(float)

    theta = np.zeros(P)
    theta[2 * K:2 * K + len(cups)] = np.log(1.3)
    theta[P - 1] = 0.2
    for _ in range(iters):
        lh = np.exp(Xh @ theta + off_h)
        la = np.exp(Xa @ theta + off_a)
        g = Xh.T @ (yh - lh) + Xa.T @ (ya - la) - Q @ theta - Gm @ theta + gm
        H = Xh.T @ (Xh * lh[:, None]) + Xa.T @ (Xa * la[:, None]) + Q + Gm
        if len(ntot):
            pt = 1 / (1 + np.exp(-beta * (Ut @ theta)))
            g = g + (beta / kappa) * Ut.T @ (nab - ntot * pt)
            H = H + (beta ** 2 / kappa) * Ut.T @ (Ut * (ntot * pt * (1 - pt))[:, None])
        step = np.linalg.solve(H, g)
        theta += step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.inv(H)
    # evidencia de Laplace (para elegir tau): log p(y) ≈ l(θ*) - ½θ*ᵀQθ* - ½ log|H| + ½ log|Q|
    from scipy.special import gammaln
    ll = (yh * np.log(lh) - lh - gammaln(yh + 1)).sum() + (ya * np.log(la) - la - gammaln(ya + 1)).sum()
    if len(ntot):
        pt = 1 / (1 + np.exp(-beta * (Ut @ theta)))
        ll += ((nab * np.log(pt) + (ntot - nab) * np.log1p(-pt)).sum()) / kappa
    quad = -0.5 * theta @ (Q + Gm) @ theta + gm @ theta
    log_ev = ll + quad - 0.5 * np.linalg.slogdet(H)[1] + 0.5 * np.linalg.slogdet(Q + Gm)[1]
    return CrossLeagueFit(keys, theta[:2 * K].reshape(K, 2), cov[:2 * K, :2 * K],
                          dict(levels=dict(zip(cups, np.exp(theta[2 * K:2 * K + len(cups)]))),
                               home=float(np.exp(theta[P - 1]))),
                          E, movers, tau, float(log_ev), theta.copy(), kix, cix, kappa, beta)


def league_graph(edges: pd.DataFrame, movers: pd.DataFrame, weight="fisher") -> pd.DataFrame:
    """Conductancias agregadas entre ligas (todas las temporadas): número de partidos y movimientos."""
    w = {}
    for r in edges.itertuples():
        if r.lh == r.la:
            continue
        k = tuple(sorted((r.lh, r.la)))
        w[k] = w.get(k, 0) + 1
    for r in movers.itertuples():
        k = tuple(sorted((r.l1, r.l2)))
        w[k] = w.get(k, 0) + 1
    return pd.DataFrame([dict(a=a, b=b, w=v) for (a, b), v in w.items()])


def fiedler(graph: pd.DataFrame):
    """Valor y vector de Fiedler del laplaciano del grafo de ligas (conectividad algebraica)."""
    nodes = sorted(set(graph["a"]) | set(graph["b"]))
    ix = {n: i for i, n in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    for r in graph.itertuples():
        i, j = ix[r.a], ix[r.b]
        L[i, j] -= r.w; L[j, i] -= r.w; L[i, i] += r.w; L[j, j] += r.w
    # laplaciano normalizado para que ligas con muchos partidos no dominen
    d = np.sqrt(np.diag(L))
    Ln = L / d[:, None] / d[None, :]
    vals, vecs = np.linalg.eigh(Ln)
    return nodes, vals, vecs[:, 1] / d
