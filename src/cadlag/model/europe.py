r"""Capa europea: fuerzas de los clubes de toda Europa en una escala común, para las copas UEFA.

Planteamiento
-------------
Los equipos de las ligas que modelamos (Premier, LaLiga, Serie A, Bundesliga, Ligue 1) ya tienen una
fuerza bien medida por su filtro doméstico, pero relativa a su liga. Los demás (Porto, PSV,
Galatasaray, Bodø/Glimt, ...) solo se ven en las copas. Cada partido UEFA (local i, visita j) observa

    log E[goles de i] = mu + eta (1 - neutral) + att_i - def_j,      goles ~ NegBin(·, r)

donde, según el equipo:

  * **liga modelada ℓ**: att_i = gamma · att_i^dom + A_ℓ  y  def_i = gamma · def_i^dom + D_ℓ.
    La fuerza doméstica sale del modelo oficial de dos latentes, linealizado en goles:
    log E[goles] ≈ c + a_i − b_j + log σ(φ) + κ (f_i − g_j) con κ = 1 − σ(φ), o sea
    att^dom = a + κ f y def^dom = b + κ g, con la incertidumbre del filtro doméstico en esa fecha.
    (A_ℓ, D_ℓ) son los offsets de la liga; la Premier es la referencia (offset 0). Además cada
    equipo tiene un residuo europeo (u_att, u_def) ~ N(0, vu) que aprende de sus partidos UEFA lo
    que la liga no muestra (rinde más o menos en Europa que en casa); revierte a 0 entre temporadas.
  * **resto de Europa**: (att_i, def_i) es un estado latente propio. Su prior al aparecer por
    primera vez es N(beta_c, v0) con beta_c según la copa que juega (Champions, Europa League,
    Conference o las previas de cualquiera): clasificar a la Champions ya dice algo de la fuerza.
    Dentro de la temporada sigue un paseo aleatorio (varianza q por día); entre temporadas revierte hacia la media de la copa
    que juega ese año, m ← beta_c + phi (m − beta_c), con varianza phi² P + s². (Se probaron una
    media propia para los equipos que ya estaban al empezar los datos y una media común de
    reversión: ajustan mejor dentro de muestra y peor fuera, 2,989 frente a 2,974 nats por partido.)

Los parámetros globales g = (mu, eta, A_ℓ, D_ℓ) tienen un paseo aleatorio lento y, entre
temporadas, los offsets reciben varianza tau².

Filtro factorizado
------------------
Con cientos de equipos, la covarianza completa no cabe en el gradiente. Se usa un filtro de
densidad supuesta factorizado (ADF): la distribución se guarda como un bloque global (g, con su
covarianza completa) y un bloque 2×2 por equipo, independientes entre sí. En cada partido se
juntan los bloques que intervienen, se hace la actualización de Laplace exacta en ese estado local
(como en el filtro doméstico) y se devuelven las marginales a cada bloque. Se pierde la correlación
que un partido crea entre los dos equipos y los offsets; a cambio, cada paso cuesta O(1).

Los hiperparámetros se ajustan maximizando la log-verosimilitud predictiva un paso adelante
(prequential), con gradientes exactos por autodiferenciación a través de todo el filtro.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..data import store
from ..leagues import EUROPE_EXTRA
from .filter_shots2 import _laplace, _nb

jax.config.update("jax_enable_x64", True)

UEFA = ["uefa.champions", "uefa.europa", "uefa.europa.conf"]
BIG5 = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"]       # la Premier es la referencia
DOMESTIC = BIG5 + EUROPE_EXTRA                               # ligas con modelo doméstico
OFFSET_LEAGUES = DOMESTIC[1:]
G = 2 + 2 * len(OFFSET_LEAGUES)                              # (mu, eta, A_ℓ..., D_ℓ...)
CACHE = store.ROOT / "outputs" / "europe"

HYPER_NAMES = ["mu0", "eta0", "log_qg", "log_tau", "log_voff0", "log_r", "log_qt", "logit_phi", "log_st",
               "log_v0", "beta_ucl", "beta_uel", "beta_uecl", "gamma", "log_vdom",
               "log_vu", "logit_phiu", "log_su",     # residuo europeo de los equipos con liga
               "beta_qual",                          # prior de quien aparece en las previas
               "off_extra", "log_voff_extra"]        # prior de los offsets de las ligas fuera del top 5
DEFAULT_HYPER = np.array([0.15, 0.25, np.log(1e-6), np.log(0.06), np.log(0.1 ** 2), np.log(20.0), np.log(2e-5),
                          2.0, np.log(0.08), np.log(0.15), -0.15, -0.35, -0.50, 1.0, np.log(1e-3),
                          np.log(0.01), 1.0, np.log(0.05), -0.9, -0.35, np.log(0.2 ** 2)])


def unpack(h):
    return dict(mu0=h[0], eta0=h[1], qg=jnp.exp(h[2]), tau=jnp.exp(h[3]), voff0=jnp.exp(h[4]), r=jnp.exp(h[5]),
                qt=jnp.exp(h[6]), phi=jax.nn.sigmoid(h[7]), st=jnp.exp(h[8]), v0=jnp.exp(h[9]),
                beta=jnp.stack([h[10], h[11], h[12], h[18]]), gamma=h[13], vdom=jnp.exp(h[14]),
                vu=jnp.exp(h[15]), phiu=jax.nn.sigmoid(h[16]), su=jnp.exp(h[17]), off_extra=h[19], voff_extra=jnp.exp(h[20]))


def pad(h):
    """Hiperparámetros guardados sin el residuo europeo (15 entradas): se completan con el valor por defecto."""
    h = np.asarray(h, float)
    return np.r_[h, DEFAULT_HYPER[len(h):]] if len(h) < len(DEFAULT_HYPER) else h


QUAL_SLUGS = {"play-off-round", "playoff-round", "playoffs", "preliminary-round", "qualifying-round"}


def is_qualifying(slug: pd.Series) -> pd.Series:
    """Rondas previas (hasta 2019 ESPN las trae; desde 2020 ya no)."""
    s = slug.astype(str)
    return s.isin(QUAL_SLUGS) | s.str.contains("qualifying")


def season_of(date: pd.Timestamp) -> int:
    return date.year if date.month >= 7 else date.year - 1


# ----------------------------------------------------------------------------------------
# Fuerzas domésticas (modelo oficial) en la fecha de cada partido
# ----------------------------------------------------------------------------------------
def domestic_states(matches: pd.DataFrame, which: str = "prod", comps=DOMESTIC) -> pd.DataFrame:
    """Estado filtrado (att, def) de cada equipo tras cada partido de su liga.

    ``which="prod"`` usa los hiperparámetros de producción (``models/``); ``"backtest"`` los
    ajustados con temporadas anteriores a 2016 (``outputs/backtest``), para validar sin mirar el
    futuro. Se guarda en ``outputs/europe/domestic_{which}.parquet``."""
    from ..data.events import load_stats
    from .filter import prepare
    from .filter_shots import with_shots
    from .filter_shots2 import TwoLatentShots, pad
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"domestic_{which}.parquet"
    stats = load_stats()
    rows = []
    for comp in comps:
        ld = with_shots(prepare(matches, comp), stats)
        hf = (store.ROOT / "models" / f"{comp}_shots2_h.npy") if which == "prod" else \
             (store.ROOT / "outputs" / "backtest" / f"{comp}_shots2_h_2016.npy")
        if not hf.exists() or not (matches["comp"] == comp).any():   # liga sin modelo o sin datos todavía
            print(f"[europa] {comp}: sin modelo o sin datos, sus equipos quedan con estado propio", flush=True)
            continue
        mod = TwoLatentShots(ld, pad(np.load(hf)))
        _, _, o = mod.run()
        kap = 1 - 1 / (1 + np.exp(-o["phi"]))      # κ = 1 − σ(φ) en esa fecha
        d = ld.matches
        for side, mk, Pk in (("hi", "hm", "hP"), ("ai", "am", "aP")):
            mm, PP = o[mk], o[Pk]
            k = kap
            att = mm[:, 0] + k * mm[:, 2]; dfn = mm[:, 1] + k * mm[:, 3]
            va = PP[:, 0, 0] + 2 * k * PP[:, 0, 2] + k ** 2 * PP[:, 2, 2]
            vd = PP[:, 1, 1] + 2 * k * PP[:, 1, 3] + k ** 2 * PP[:, 3, 3]
            cad = PP[:, 0, 1] + k * PP[:, 0, 3] + k * PP[:, 2, 1] + k ** 2 * PP[:, 2, 3]
            rows.append(pd.DataFrame(dict(comp=comp, team=[ld.team_ids[i] for i in d[side]], date=d["date"].to_numpy(),
                                          att=att, dfn=dfn, va=va, vd=vd, cad=cad)))
        print(f"[europa] estados domésticos {comp}: {len(d)} partidos", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t["date"] = pd.to_datetime(t["date"], utc=True)
    t = t.sort_values("date").reset_index(drop=True)
    t.to_parquet(f)
    return t


def load_domestic(matches: pd.DataFrame | None = None, which: str = "prod", refresh: bool = False) -> pd.DataFrame:
    f = CACHE / f"domestic_{which}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    if matches is None:
        matches, _ = store.load()
    return domestic_states(matches, which)


# ----------------------------------------------------------------------------------------
# Datos para el filtro
# ----------------------------------------------------------------------------------------
@dataclass
class EuroData:
    matches: pd.DataFrame        # partidos UEFA en orden (con columnas de cada lado)
    teams: list                  # ids ESPN de los equipos con estado latente propio
    arr: dict                    # arreglos para el scan
    season0: int = 0             # temporada que corresponde a si = 0


def _lookup(dom: pd.DataFrame, team, date, max_days):
    """Último estado doméstico del equipo antes de ``date`` (si no tiene más de ``max_days``)."""
    g = dom.get(team)
    if g is None:
        return None
    dd = g["day"].to_numpy()
    t = date.tz_convert(None).to_datetime64()
    i = np.searchsorted(dd, t, side="left") - 1
    if i < 0:
        return None
    gap = (t - dd[i]) / np.timedelta64(1, "D")
    if gap > max_days:
        return None
    r = g.iloc[i]
    return r, gap


def build(matches: pd.DataFrame, dom: pd.DataFrame, max_days: int = 120, upto=None,
          extra: pd.DataFrame | None = None) -> EuroData:
    """Arma los arreglos del filtro con los partidos UEFA terminados (y, si se da ``extra``, partidos
    por jugar al final, con peso 0, para obtener su predictiva)."""
    d = matches[matches["comp"].isin(UEFA) & matches["finished"]].copy()
    if upto is not None:
        d = d[d["date"] <= upto]
    d["w"] = 1.0
    if extra is not None and len(extra):
        e = extra.copy(); e["w"] = 0.0
        e["hg"] = e["hg"].fillna(0); e["ag"] = e["ag"].fillna(0)
        d = pd.concat([d, e], ignore_index=True)
    d = d.sort_values("date", kind="stable").reset_index(drop=True)
    d["neutral"] = d["neutral"].fillna(False).astype(bool) | (d["slug"].astype(str) == "final")   # ESPN no marca la final
    # el marcador de ESPN incluye el alargue: esos partidos no informan la verosimilitud de 90'
    et = d["extra_time"].fillna(False).astype(bool).to_numpy() if "extra_time" in d else np.zeros(len(d), bool)
    d.loc[et, "w"] = 0.0
    dom = dom.copy()
    dom["day"] = pd.to_datetime(dom["date"], utc=True).dt.tz_convert(None)
    by = {k: g.reset_index(drop=True) for k, g in dom.groupby("team")}
    lidx = {c: i for i, c in enumerate(DOMESTIC)}
    tix: dict = {}
    cols = {k: [] for k in ("core", "L", "att", "dfn", "va", "vd", "cad", "ti")}
    side_cols = {s: {k: [] for k in cols} for s in ("h", "a")}
    for r in d.itertuples():
        for s, team in (("h", r.home), ("a", r.away)):
            got = _lookup(by, team, r.date, max_days)
            c = side_cols[s]
            if got is not None:
                row, gap = got
                c["core"].append(1.0); c["L"].append(lidx[row["comp"]])
                c["att"].append(row["att"]); c["dfn"].append(row["dfn"])
                c["va"].append(row["va"]); c["vd"].append(row["vd"]); c["cad"].append(row["cad"])
            else:
                c["core"].append(0.0); c["L"].append(0)
                for k in ("att", "dfn", "va", "vd", "cad"):
                    c[k].append(0.0)
            if team not in tix:
                tix[team] = len(tix)
            c["ti"].append(tix[team])     # todos tienen bloque: estado propio o residuo europeo
    seasons = d["date"].map(season_of).to_numpy()
    season0 = int(seasons.min())
    si = seasons - season0                      # años desde la primera temporada (cuenta los saltos)
    day = ((d["date"] - d["date"].iloc[0]).dt.total_seconds() / 86400).to_numpy()
    comp = d["comp"].map({c: i for i, c in enumerate(UEFA)}).to_numpy().copy()
    comp[is_qualifying(d["slug"]).to_numpy()] = 3            # previas: prior propio (beta_qual)
    arr = dict(x=d["hg"].to_numpy(float), y=d["ag"].to_numpy(float), w=d["w"].to_numpy(float),
               neu=d["neutral"].fillna(False).astype(float).to_numpy(), day=day, si=si,
               new=np.r_[True, si[1:] != si[:-1]], comp=comp)
    for s in ("h", "a"):
        for k, v in side_cols[s].items():
            arr[f"{k}_{s}"] = np.asarray(v, int if k in ("L", "ti") else float)
    for s in ("h", "a"):
        d[f"core_{s}"] = arr[f"core_{s}"]
    d["season"] = seasons
    return EuroData(d, list(tix), arr, season0)


# ----------------------------------------------------------------------------------------
# Filtro
# ----------------------------------------------------------------------------------------
def _ll(z, x, y, r):
    return _nb(x, z[0], r) + _nb(y, z[1], r)


_g = jax.grad(_ll)
_h = jax.hessian(_ll)
NOFF = len(OFFSET_LEAGUES)
EXTRA_MASK = np.array([float(c not in BIG5) for c in OFFSET_LEAGUES])


def _offset_rows(L, sign_att):
    """Fila (G,) que suma A_ℓ (o D_ℓ) del equipo de la liga L (0 = Premier, sin offset)."""
    v = jnp.zeros(G)
    j = jnp.maximum(L - 1, 0)
    on = (L > 0).astype(float)
    return v.at[2 + j].add(on) if sign_att else v.at[2 + NOFF + j].add(on)


def make_filter(nT: int, chunk: int = 128):
    def run(h, arr):
        hp = unpack(h)
        beta = hp["beta"]
        ex = jnp.r_[EXTRA_MASK, EXTRA_MASK]        # offsets (A, D) de ligas fuera del top 5
        mg0 = jnp.r_[hp["mu0"], hp["eta0"], ex * hp["off_extra"]]
        Pg0 = jnp.diag(jnp.r_[jnp.full(2, 0.05 ** 2), (1 - ex) * hp["voff0"] + ex * hp["voff_extra"]])
        isoff = jnp.r_[0.0, 0.0, jnp.ones(2 * NOFF)]
        TM0 = jnp.zeros((nT, 2)); TP0 = jnp.tile(jnp.eye(2), (nT, 1, 1)); Tl = jnp.zeros(nT); Ts = -jnp.ones(nT, int)
        Tc = -jnp.ones(nT)
        qt, gam = hp["qt"], hp["gamma"]

        def team_prior(t, m, P, last, ls, lc, day, si, comp, core):
            """Bloque del equipo t antes del partido. Con liga (core=1) es el residuo europeo u: prior
            N(0, vu), revierte a 0 con phiu. Sin liga es su fuerza: prior N(beta_c, v0), revierte a beta_c."""
            mt, Pt = m[t], P[t]
            fresh = (ls[t] < 0) | (lc[t] != core)
            k = jnp.maximum(si - ls[t], 0).astype(float)
            phi = jnp.where(core > 0, hp["phiu"], hp["phi"]); st = jnp.where(core > 0, hp["su"], hp["st"])
            # los que aparecen en las dos primeras temporadas de los datos no son debutantes: prior propio
            b = jnp.where(core > 0, 0.0, beta[comp]); b0 = b
            v0 = jnp.where(core > 0, hp["vu"], hp["v0"])
            ph = phi ** k
            m_rev = b + ph * (mt - b)
            P_rev = ph ** 2 * Pt + jnp.eye(2) * st ** 2 * (1 - ph ** 2) / (1 - phi ** 2 + 1e-9)
            mt = jnp.where(fresh, jnp.full(2, b0), jnp.where(k > 0, m_rev, mt))
            Pt = jnp.where(fresh, jnp.eye(2) * v0, jnp.where(k > 0, P_rev, Pt))
            dt = jnp.where(fresh | (k > 0), 0.0, day - last[t])
            return mt, Pt + jnp.eye(2) * qt * dt

        def step(carry, inp):
            mg, Pg, TM, TP, Tl, Ts, Tc, lastday = carry
            # ---- dinámica global
            Pg = Pg + jnp.diag(jnp.full(G, hp["qg"])) * (inp["day"] - lastday)
            Pg = Pg + jnp.where(inp["new"], 1.0, 0.0) * jnp.diag(isoff * hp["tau"] ** 2)
            th, ta = inp["ti_h"], inp["ti_a"]
            ch, ca = inp["core_h"], inp["core_a"]
            mh, Ph = team_prior(th, TM, TP, Tl, Ts, Tc, inp["day"], inp["si"], inp["comp"], ch)
            ma, Pa = team_prior(ta, TM, TP, Tl, Ts, Tc, inp["day"], inp["si"], inp["comp"], ca)
            # ---- estado local x = (g, att_h, def_h, att_a, def_a)
            x = jnp.r_[mg, mh, ma]
            P = jnp.zeros((G + 4, G + 4)).at[:G, :G].set(Pg).at[G:G + 2, G:G + 2].set(Ph).at[G + 2:, G + 2:].set(Pa)
            rh = jnp.zeros(G).at[0].set(1).at[1].set(1 - inp["neu"]) + ch * _offset_rows(inp["L_h"], True) - ca * _offset_rows(inp["L_a"], False)
            ra = jnp.zeros(G).at[0].set(1) + ca * _offset_rows(inp["L_a"], True) - ch * _offset_rows(inp["L_h"], False)
            A = jnp.stack([jnp.r_[rh, 1.0, 0.0, 0.0, -1.0], jnp.r_[ra, 0.0, -1.0, 1.0, 0.0]])
            c = jnp.stack([gam * (ch * inp["att_h"] - ca * inp["dfn_a"]), gam * (ca * inp["att_a"] - ch * inp["dfn_h"])])
            vdom = hp["vdom"]
            Sd = gam ** 2 * jnp.array([[ch * inp["va_h"] + ca * inp["vd_a"], -(ch * inp["cad_h"] + ca * inp["cad_a"])],
                                       [-(ch * inp["cad_h"] + ca * inp["cad_a"]), ca * inp["va_a"] + ch * inp["vd_h"]]])
            Sd = Sd + jnp.eye(2) * vdom * jnp.array([ch + ca, ch + ca])
            zbar = A @ x + c
            S = A @ P @ A.T + Sd + 1e-9 * jnp.eye(2)
            X, Y, r = inp["x"], inp["y"], hp["r"]
            z, Sig, logp = _laplace(zbar, S, lambda z: _g(z, X, Y, r), lambda z: _h(z, X, Y, r), lambda z: _ll(z, X, Y, r))
            upd = inp["w"] > 0
            K = P @ A.T @ jnp.linalg.inv(S)
            x1 = x + K @ (z - zbar); P1 = P + K @ (Sig - S) @ K.T; P1 = 0.5 * (P1 + P1.T)
            x1 = jnp.where(upd, x1, x); P1 = jnp.where(upd, P1, P)
            mg, Pg = x1[:G], P1[:G, :G]
            # ---- devolver las marginales a cada bloque
            def put(TM, TP, Tl, Ts, Tc, t, core, mt, Pt):
                return (TM.at[t].set(mt), TP.at[t].set(Pt), Tl.at[t].set(inp["day"]), Ts.at[t].set(inp["si"]),
                        Tc.at[t].set(core))
            TM, TP, Tl, Ts, Tc = put(TM, TP, Tl, Ts, Tc, th, ch, x1[G:G + 2], P1[G:G + 2, G:G + 2])
            TM, TP, Tl, Ts, Tc = put(TM, TP, Tl, Ts, Tc, ta, ca, x1[G + 2:], P1[G + 2:, G + 2:])
            return (mg, Pg, TM, TP, Tl, Ts, Tc, inp["day"]), dict(logp=logp, zbar=zbar, S=S)

        init = (mg0, Pg0, TM0, TP0, Tl, Ts, Tc, jnp.asarray(0.0))
        n = arr["x"].shape[0]
        nb = -(-n // chunk); pad = nb * chunk - n
        arrp = {k: jnp.concatenate([v, jnp.repeat(v[-1:], pad, 0)]) if pad else v for k, v in arr.items()}
        if pad:   # relleno sin efecto: peso 0, mismo día
            arrp["w"] = arrp["w"].at[n:].set(0.0); arrp["new"] = arrp["new"].at[n:].set(False)
        arrc = {k: v.reshape((nb, chunk) + v.shape[1:]) for k, v in arrp.items()}

        @jax.checkpoint
        def outer(carry, blk):
            return jax.lax.scan(step, carry, blk)
        final, outs = jax.lax.scan(outer, init, arrc)
        outs = {k: v.reshape((nb * chunk,) + v.shape[2:])[:n] for k, v in outs.items()}
        return final, outs
    return run


class EuroFilter:
    def __init__(self, ed: EuroData, h=None):
        self.ed = ed
        self.h = pad(DEFAULT_HYPER if h is None else h)
        self.arr = {k: jnp.asarray(v) for k, v in ed.arr.items()}
        self._run = make_filter(max(len(ed.teams), 1))
        self._jrun = jax.jit(lambda h: self._run(h, self.arr))

    def fit(self, mask=None, maxiter=200, l2=1e-3, verbose=True):
        from scipy.optimize import minimize
        w = jnp.asarray(self.ed.arr["w"] if mask is None else self.ed.arr["w"] * mask)
        h0 = jnp.asarray(DEFAULT_HYPER)

        def f(h):
            _, o = self._run(h, self.arr)
            return -(o["logp"] * w).sum() + l2 * jnp.sum((h - h0) ** 2) * 100.0
        vg = jax.jit(jax.value_and_grad(f))

        def fun(h):
            v, g = vg(jnp.asarray(h)); return float(v), np.nan_to_num(np.asarray(g, float))
        res = minimize(fun, self.h, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
        self.h = res.x
        if verbose:
            print(f"[europa] -logp/partido = {res.fun / float(w.sum()):.4f} ({res.nit} iter)", flush=True)
        return res

    def run(self):
        final, o = self._jrun(jnp.asarray(self.h))
        final = tuple(np.asarray(v) for v in final)
        return final, {k: np.asarray(v) for k, v in o.items()}

    def params(self):
        p = unpack(jnp.asarray(self.h))
        return {k: (np.asarray(v) if np.ndim(v) else float(v)) for k, v in p.items()}


# ----------------------------------------------------------------------------------------
# Fuerzas de un conjunto de equipos en una fecha (para simular una copa)
# ----------------------------------------------------------------------------------------
@dataclass
class Strengths:
    """(att, def) en la escala común: m, P ya incluyen gamma · doméstico + residuo para los equipos
    con liga; a esos se les suma además el offset de su liga, que se sortea junto con (mu, eta)."""
    teams: list
    core: np.ndarray            # (n,) bool
    L: np.ndarray               # (n,) liga (índice en DOMESTIC; 0 = Premier o sin liga)
    m: np.ndarray               # (n, 2)
    P: np.ndarray               # (n, 2, 2)
    mg: np.ndarray; Pg: np.ndarray
    r: float

    def _off(self, g):
        """Offsets (S, n) de ataque y defensa a partir de los parámetros globales g (S, G)."""
        S = g.shape[0]; n = len(self.teams)
        oa, od = np.zeros((S, n)), np.zeros((S, n))
        for i in np.where(self.core & (self.L > 0))[0]:
            oa[:, i] = g[:, 2 + self.L[i] - 1]; od[:, i] = g[:, 2 + NOFF + self.L[i] - 1]
        return oa, od

    def draw(self, S, rng):
        """F = dict(att, dfn (S, n), mu, eta (S,))."""
        n = len(self.teams)
        g = rng.multivariate_normal(self.mg, self.Pg, size=S, method="cholesky")
        Lc = np.linalg.cholesky(self.P + 1e-12 * np.eye(2))
        x = self.m[None] + np.einsum("nij,snj->sni", Lc, rng.standard_normal((S, n, 2)))
        oa, od = self._off(g)
        return dict(att=x[..., 0] + oa, dfn=x[..., 1] + od, mu=g[:, 0], eta=g[:, 1])

    def table(self, names=None):
        oa, od = self._off(self.mg[None])
        att, dfn = self.m[:, 0] + oa[0], self.m[:, 1] + od[0]
        sd = np.sqrt(self.P[:, 0, 0] + self.P[:, 1, 1] + 2 * self.P[:, 0, 1])
        return pd.DataFrame(dict(team=self.teams, name=[(names or {}).get(t, t) for t in self.teams],
                                 core=self.core, att=att, dfn=dfn, net=att + dfn, sd_net=sd))


def _revert(mt, Pt, b, phi, st, k):
    ph = phi ** k
    return b + ph * (mt - b), ph ** 2 * Pt + np.eye(2) * st ** 2 * (1 - ph ** 2) / (1 - phi ** 2)


def strengths_at(ef: "EuroFilter", final, dom: pd.DataFrame, teams: list, date: pd.Timestamp, comp: str,
                 max_days: int = 120) -> Strengths:
    """Estado en ``date`` (después de los partidos UEFA ya filtrados en ``final``)."""
    hp = ef.params()
    mg, Pg, TM, TP, Tl, Ts, Tc, lastday = final
    ed = ef.ed
    day = (date - ed.matches["date"].iloc[0]).total_seconds() / 86400
    si = season_of(date) - ed.season0
    ci = UEFA.index(comp)
    Pg = Pg + np.eye(G) * hp["qg"] * max(day - float(lastday), 0)
    if si > int(ed.arr["si"].max()):
        Pg = Pg + np.diag(np.r_[0.0, 0.0, np.ones(2 * NOFF)]) * hp["tau"] ** 2
    dom = dom.copy(); dom["day"] = pd.to_datetime(dom["date"], utc=True).dt.tz_convert(None)
    by = {k: g.reset_index(drop=True) for k, g in dom.groupby("team")}
    tix = {t: i for i, t in enumerate(ed.teams)}
    lidx = {c: i for i, c in enumerate(DOMESTIC)}
    n = len(teams)
    core = np.zeros(n, bool); L = np.zeros(n, int); m = np.zeros((n, 2)); P = np.zeros((n, 2, 2))
    beta = np.asarray(hp["beta"]); gam = hp["gamma"]
    for k, t in enumerate(teams):
        got = _lookup(by, t, date, max_days)
        c = got is not None
        # bloque propio del equipo (residuo si tiene liga, fuerza si no)
        b, v0, phi, st = (0.0, hp["vu"], hp["phiu"], hp["su"]) if c else (beta[ci], hp["v0"], hp["phi"], hp["st"])
        j = tix.get(t)
        if j is None or Ts[j] < 0 or Tc[j] != float(c):
            um, uP = np.full(2, b), np.eye(2) * v0
        else:
            um, uP = TM[j], TP[j]
            kk = max(si - int(Ts[j]), 0)
            if kk > 0:
                um, uP = _revert(um, uP, b, phi, st, kk)
            else:
                uP = uP + np.eye(2) * hp["qt"] * max(day - Tl[j], 0)
        if c:
            row, gap = got
            core[k] = True; L[k] = lidx[row["comp"]]
            m[k] = gam * np.array([row["att"], row["dfn"]]) + um
            P[k] = gam ** 2 * np.array([[row["va"], row["cad"]], [row["cad"], row["vd"]]]) + uP + np.eye(2) * hp["vdom"] / 2
        else:
            m[k], P[k] = um, uP
    return Strengths(list(teams), core, L, m, P, np.asarray(mg), np.asarray(Pg), float(hp["r"]))
