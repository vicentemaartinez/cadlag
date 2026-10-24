r"""Extensión E6: modelo en vivo (proceso de conteo con estados) y prueba de "contagio".

Reloj del modelo
----------------
ESPN registra los goles del tiempo añadido en 45:00 y 90:00 exactos. Por eso se usa un reloj
tau con cuatro fases: primer tiempo [0, 45), añadido del primero [45, 46), segundo tiempo
[46, 91) y añadido del segundo [91, 92). Cada fase añadida dura una "unidad": su coeficiente
mide directamente cuántos goles se esperan en el descuento, sin necesitar su duración real.

Intensidad (log-lineal) del equipo s en el instante tau
-------------------------------------------------------
    log lambda_s(tau) = z_s - log 90                       (goles esperados antes del partido, del filtro)
                      + b_{k(tau)}                         (perfil temporal por tramos de 15')
                      + c^T score_state_s(tau)             (ganando por 1, por 2+, perdiendo por 1, por 2+)
                      + r_own * rojas_propias + r_opp * rojas_rival
                      + h_self * 1{gol propio en los últimos 10'} + h_opp * 1{gol recibido en los últimos 10'}

Con estados constantes entre eventos, la verosimilitud del proceso es la de un GLM de Poisson
sobre segmentos con exposición = duración (Lindsey; "piecewise exponential"). Los términos h_*
son la versión multiplicativa de un kernel de Hawkes con ventana rectangular.

Hawkes aditivo
--------------
Condicionando en la intensidad ajustada lambda_0 (que ya incluye el estado del marcador), se
prueba la excitación aditiva  lambda(tau) = lambda_0(tau) + sum_{k: tau_k < tau} a beta e^{-beta (tau - tau_k)}
con goles propios (auto-excitación) y del rival (excitación cruzada) por máxima verosimilitud;
a = 0 es el nulo de "no contagio" (prueba de razón de verosimilitud en la frontera, mezcla chi^2).

Probabilidades en vivo
----------------------
Dado el estado en tau (marcador, rojas), la distribución del marcador final resuelve las
ecuaciones de Kolmogorov hacia adelante dp/dtau = p Q(tau) en la grilla de marcadores; como Q
es constante por tramos se integra con exponenciales de matriz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm

BINS = [(0, 15), (15, 30), (30, 45), (45, 46), (46, 61), (61, 76), (76, 91), (91, 92)]
BIN_NAMES = ["0-15", "15-30", "30-45", "45+", "45-60", "60-75", "75-90", "90+"]
FEATS = ["lead1", "lead2", "trail1", "trail2", "red_own", "red_opp", "after_own_goal", "after_conceded"]
WINDOW = 10.0


def tau_of(t_sec: float) -> float:
    if t_sec < 2700:
        return t_sec / 60.0
    if t_sec == 2700:
        return 45.5
    if t_sec < 5400:
        return 46.0 + (t_sec - 2700) / 60.0
    return 91.5


def _bin(tau):
    for k, (a, b) in enumerate(BINS):
        if a <= tau < b:
            return k
    return len(BINS) - 1


def build_segments(matches: pd.DataFrame, ev: pd.DataFrame, use_contagion=True) -> pd.DataFrame:
    """Divide cada partido en segmentos de estado constante, por equipo.

    ``matches`` necesita columnas id, zh, za (log goles esperados pre-partido) y hg, ag.
    Devuelve filas (match, side, bin, exposición, goles, features).
    """
    ev = ev[ev["type"].isin(["G", "P", "O", "R"]) & ev["side"].isin(["h", "a"]) & ev["t"].notna()]
    by = {k: g for k, g in ev.groupby("id")}
    rows = []
    for r in matches.itertuples():
        g = by.get(r.id)
        evs = [] if g is None else sorted((tau_of(t), typ, side) for t, typ, side in zip(g["t"], g["type"], g["side"]))
        # cortes: bordes de tramo + eventos + fin de ventanas de contagio
        cuts = {a for a, _ in BINS} | {92.0}
        last_goal = {"h": [], "a": []}
        for tau, typ, side in evs:
            cuts.add(tau)
            if use_contagion and typ in "GPO":
                cuts.add(min(tau + WINDOW, 92.0))
        cuts = sorted(c for c in cuts if 0 <= c <= 92)
        score = {"h": 0, "a": 0}
        reds = {"h": 0, "a": 0}
        goals_t = {"h": [], "a": []}
        ei = 0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            # aplicar eventos que ocurren exactamente en a
            while ei < len(evs) and evs[ei][0] <= a + 1e-9:
                tau, typ, side = evs[ei]
                if typ in "GPO":
                    score[side] += 1
                    goals_t[side].append(tau)
                else:
                    reds[side] += 1
                ei += 1
            # goles dentro de (a, b]: se cuentan en este segmento (los cortes están en los eventos,
            # así que cada gol cae justo en el borde derecho de su segmento)
            n_goals = {"h": 0, "a": 0}
            j = ei
            while j < len(evs) and evs[j][0] <= b + 1e-9:
                if evs[j][1] in "GPO" and abs(evs[j][0] - b) < 1e-9:
                    n_goals[evs[j][2]] += 1
                j += 1
            k = _bin(a)
            for side, opp, z in (("h", "a", r.zh), ("a", "h", r.za)):
                d = score[side] - score[opp]
                rows.append((r.id, side, k, b - a, n_goals[side], z,
                             d == 1, d >= 2, d == -1, d <= -2, min(reds[side], 2), min(reds[opp], 2),
                             any(a - t < WINDOW for t in goals_t[side] if t <= a),
                             any(a - t < WINDOW for t in goals_t[opp] if t <= a)))
    cols = ["id", "side", "bin", "expo", "goals", "z"] + FEATS
    return pd.DataFrame(rows, columns=cols)


def fit_glm(seg: pd.DataFrame, feats=FEATS, iters=50, ridge=1e-6):
    """GLM de Poisson con offset z - log 90 + log(exposición). Devuelve coeficientes y errores estándar."""
    K = len(BINS)
    X = np.zeros((len(seg), K + len(feats)))
    X[np.arange(len(seg)), seg["bin"].to_numpy()] = 1.0
    for j, f in enumerate(feats):
        X[:, K + j] = seg[f].to_numpy(float)
    off = seg["z"].to_numpy() - np.log(90.0) + np.log(seg["expo"].to_numpy())
    y = seg["goals"].to_numpy(float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(X @ beta + off)
        g = X.T @ (y - mu) - ridge * beta
        H = X.T @ (X * mu[:, None]) + ridge * np.eye(len(beta))
        st = np.linalg.solve(H, g)
        beta += st
        if np.abs(st).max() < 1e-10:
            break
    cov = np.linalg.inv(H)
    names = BIN_NAMES + list(feats)
    ll = float((y * np.log(mu) - mu).sum())
    return pd.DataFrame(dict(coef=beta, se=np.sqrt(np.diag(cov)), mult=np.exp(beta)), index=names), ll


# ----------------------------------------------------------------------------------------
# Hawkes aditivo condicionado a la intensidad de base
# ----------------------------------------------------------------------------------------
def hawkes_test(seg: pd.DataFrame, coef: pd.DataFrame, ev: pd.DataFrame, betas=(0.1, 0.2, 0.5, 1.0)):
    """Estima a_self y a_cross (goles extra esperados por gol, kernel exponencial) con lambda_0 fija."""
    from scipy.optimize import minimize

    K = len(BINS)
    feats = [f for f in coef.index[K:] if not f.startswith("after_")]
    b = coef["coef"].to_numpy()
    eta = b[seg["bin"].to_numpy()] + sum(coef.loc[f, "coef"] * seg[f].to_numpy(float) for f in feats)
    lam0 = np.exp(seg["z"].to_numpy() - np.log(90.0) + eta)  # intensidad de base por unidad de tau
    ev = ev[ev["type"].isin(["G", "P", "O"]) & ev["side"].isin(["h", "a"]) & ev["t"].notna()]
    goals = {k: [(tau_of(t), s) for t, s in zip(g["t"], g["side"])] for k, g in ev.groupby("id")}
    # reconstruir, por segmento, inicio/fin en tau y los goles previos
    seg = seg.assign(lam0=lam0)
    seg["end"] = seg.groupby(["id", "side"])["expo"].cumsum()
    seg["start"] = seg["end"] - seg["expo"]
    ids = seg["id"].to_numpy()
    sides = seg["side"].to_numpy()
    st, en = seg["start"].to_numpy(), seg["end"].to_numpy()
    y = seg["goals"].to_numpy()
    results = []
    for beta in betas:
        # integrales del kernel (por unidad de a) en cada segmento y valor en el borde derecho
        Iself = np.zeros(len(seg)); Icross = np.zeros(len(seg))
        Eself = np.zeros(len(seg)); Ecross = np.zeros(len(seg))
        for i in range(len(seg)):
            for (tk, sk) in goals.get(ids[i], ()):
                if tk >= en[i] - 1e-9:
                    continue
                lo = max(st[i], tk)
                integ = np.exp(-beta * (lo - tk)) - np.exp(-beta * (en[i] - tk))
                val = beta * np.exp(-beta * (en[i] - tk))
                if sk == sides[i]:
                    Iself[i] += integ; Eself[i] += val
                else:
                    Icross[i] += integ; Ecross[i] += val
        L0 = seg["lam0"].to_numpy() * seg["expo"].to_numpy()

        def nll(p):
            a1, a2 = np.exp(p)
            rate_end = seg["lam0"].to_numpy() + a1 * Eself + a2 * Ecross
            comp = L0 + a1 * Iself + a2 * Icross
            return -(np.sum(y * np.log(rate_end)) - comp.sum())

        r = minimize(nll, np.log([1e-3, 1e-3]), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
        r0 = nll(np.log([1e-12, 1e-12]))
        results.append(dict(beta=beta, a_self=float(np.exp(r.x[0])), a_cross=float(np.exp(r.x[1])),
                            lr=float(2 * (r0 - r.fun))))
    return pd.DataFrame(results)


# ----------------------------------------------------------------------------------------
# Probabilidades en vivo (Kolmogorov hacia adelante)
# ----------------------------------------------------------------------------------------
def live_distribution(zh: float, za: float, coef: pd.DataFrame, tau0: float, score=(0, 0), reds=(0, 0),
                      maxg: int = 10) -> np.ndarray:
    """Distribución del marcador final dado el estado en tau0 (sin contagio ni rojas futuras)."""
    K = len(BINS)
    c = coef["coef"]
    G = maxg + 1
    nS = G * G
    p = np.zeros(nS)
    p[min(score[0], maxg) * G + min(score[1], maxg)] = 1.0

    def Q(k):
        M = np.zeros((nS, nS))
        for h in range(G):
            for a in range(G):
                s = h * G + a
                for side, gl in (("h", h - a), ("a", a - h)):
                    z = zh if side == "h" else za
                    own_r, opp_r = (reds if side == "h" else reds[::-1])
                    eta = (c.iloc[k] + c["lead1"] * (gl == 1) + c["lead2"] * (gl >= 2) + c["trail1"] * (gl == -1)
                           + c["trail2"] * (gl <= -2) + c["red_own"] * min(own_r, 2) + c["red_opp"] * min(opp_r, 2))
                    rate = np.exp(z - np.log(90.0) + eta)
                    if side == "h" and h < maxg:
                        M[s, (h + 1) * G + a] += rate; M[s, s] -= rate
                    if side == "a" and a < maxg:
                        M[s, h * G + a + 1] += rate; M[s, s] -= rate
        return M

    for k, (a, b) in enumerate(BINS):
        lo = max(a, tau0)
        if lo >= b:
            continue
        p = p @ expm(Q(k) * (b - lo))
    return p.reshape(G, G)
