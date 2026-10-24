"""Motor de pronóstico por liga: envuelve el modelo de goles en producción.

El modelo oficial es el de dos latentes (``shots2``, ``model.filter_shots2``): fuerzas para crear
y evitar tiros al arco y para convertirlos y evitar que los conviertan. El Dixon–Coles dinámico de
un latente (``dc``, ``model.filter``) queda como modelo de referencia y de respaldo, con la misma
interfaz.

``models/model_selection.json`` (escrito por ``scripts/select_models.py``) puede devolver una liga
al núcleo, pero solo si los backtests fuera de muestra muestran que el de dos latentes pierde
(dos errores estándar en contra, en partidos o en temporadas). Sin archivo, todas usan ``shots2``.
"""
from __future__ import annotations

import json

import numpy as np

from ..data import store
from .filter import DynamicDC, prepare

MODELS = store.ROOT / "models"
SELECTION = MODELS / "model_selection.json"


def kind_for(comp: str) -> str:
    try:
        return json.loads(SELECTION.read_text()).get(comp, {}).get("kind", "shots2")
    except FileNotFoundError:
        return "shots2"


def _sig(x):
    return 1 / (1 + np.exp(-x))


class Engine:
    def __init__(self, matches, comp: str, kind: str | None = None, refit: bool = False, verbose: bool = True):
        self.comp = comp
        self.kind = kind or kind_for(comp)
        ld = prepare(matches, comp)
        MODELS.mkdir(exist_ok=True)
        if self.kind == "shots2":
            from ..data.events import load_stats
            from .filter_shots import with_shots
            from .filter_shots2 import TwoLatentShots
            ld = with_shots(ld, load_stats())
            f = MODELS / f"{comp}_shots2_h.npy"
            mod = TwoLatentShots(ld, np.load(f) if f.exists() and not refit else None)
            if refit or not f.exists():
                mod.fit(verbose=verbose); np.save(f, mod.h)
        else:
            f = MODELS / f"{comp}_h.npy"
            mod = DynamicDC(ld, np.load(f) if f.exists() and not refit else None)
            if refit or not f.exists():
                mod.fit(verbose=verbose); np.save(f, mod.h)
        self.ld, self.mod = ld, mod

    # ---- interfaz común ---------------------------------------------------------------
    @property
    def N(self):
        return self.ld.N

    def params(self) -> dict:
        p = self.mod.params()
        p.setdefault("kind", "dc")
        return p

    def run(self):
        mT, PT, _ = self.mod.run()
        return mT, PT

    def run_checkpoints(self, idx):
        return self.mod.run_checkpoints(idx)

    def preseason(self, m, P, syear):
        if self.kind == "shots2":
            from .filter_shots2 import season_transition_np
            return season_transition_np(m, P, self.params(), self.ld, syear)
        from ..eval.backtest import preseason_transition
        return preseason_transition(m, P, self.params(), self.ld, syear)

    def levels(self, m) -> dict:
        N = self.N
        if self.kind == "shots2":
            return dict(home_advantage=float(np.exp(m[4 * N + 1])), goal_level=float(np.exp(m[4 * N]) * _sig(m[4 * N + 2])))
        return dict(home_advantage=float(np.exp(m[2 * N + 1])), goal_level=float(np.exp(m[2 * N])))

    def ratings(self, m, P, teams) -> list[dict]:
        """Ataque y defensa en log-goles frente al promedio (comparables entre modelos)."""
        N = self.N
        tix = {t: i for i, t in enumerate(self.ld.team_ids)}
        out = []
        if self.kind != "shots2":
            mu = m[2 * N]
            for t in teams:
                i = tix[t]; a, d = m[i], m[N + i]
                va, vd, cad = P[i, i], P[N + i, N + i], P[i, N + i]
                out.append(dict(attack=float(a), defense=float(d), attack_sd=float(np.sqrt(va)), defense_sd=float(np.sqrt(vd)),
                                net=float(a + d), net_sd=float(np.sqrt(va + vd + 2 * cad)),
                                xgf_neutral=float(np.exp(mu + a)), xga_neutral=float(np.exp(mu - d))))
            return out
        c, phi = m[4 * N], m[4 * N + 2]
        for t in teams:
            i = tix[t]; blk = [i, N + i, 2 * N + i, 3 * N + i]
            a, b, f, g = m[blk]
            Pb = P[np.ix_(blk, blk)]
            att = a + np.log(_sig(phi + f)) - np.log(_sig(phi))
            dfn = b - np.log(_sig(phi - g)) + np.log(_sig(phi))
            ua = np.array([1, 0, 1 - _sig(phi + f), 0]); ud = np.array([0, 1, 0, 1 - _sig(phi - g)])
            out.append(dict(attack=float(att), defense=float(dfn), attack_sd=float(np.sqrt(ua @ Pb @ ua)),
                            defense_sd=float(np.sqrt(ud @ Pb @ ud)), net=float(att + dfn),
                            net_sd=float(np.sqrt((ua + ud) @ Pb @ (ua + ud))),
                            xgf_neutral=float(np.exp(c + a) * _sig(phi + f)), xga_neutral=float(np.exp(c - b) * _sig(phi - g)),
                            creation=float(a), prevention=float(b), finishing=float(f), keeping=float(g),
                            finishing_sd=float(np.sqrt(Pb[2, 2])),
                            sot_for=float(np.exp(c + a)), conv_for=float(_sig(phi + f)),
                            sot_against=float(np.exp(c - b)), conv_against=float(_sig(phi - g))))
        return out

    def upcoming(self, m, P, st, names, horizon_days=10) -> list[dict]:
        from ..pipeline import upcoming as upcoming_dc
        if self.kind != "shots2":
            return upcoming_dc(m, P, self.params(), self.ld, st, names, horizon_days)
        from scipy.stats import nbinom

        from .filter_shots2 import outcome_probs_mc
        N = self.N; hp = self.params()
        tix = {t: i for i, t in enumerate(self.ld.team_ids)}
        todo = st.todo[st.todo["days_ahead"] <= horizon_days]
        if len(todo) == 0:
            todo = st.todo.head(10)
        q = np.r_[np.full(2 * N, hp["qs"] ** 2), np.full(2 * N, hp["qf"] ** 2), np.full(3, hp["qlev"] ** 2)]
        rows, Z, SS = [], [], []
        for _, r in todo.iterrows():
            h, a = tix[st.teams[r.hi]], tix[st.teams[r.ai]]
            A = np.zeros((4, 4 * N + 3))
            A[0, [h, N + a, 4 * N, 4 * N + 1]] = [1, -1, 1, 1 - float(r.neutral)]
            A[1, [a, N + h, 4 * N]] = [1, -1, 1]
            A[2, [2 * N + h, 3 * N + a, 4 * N + 2]] = [1, -1, 1]
            A[3, [2 * N + a, 3 * N + h, 4 * N + 2]] = [1, -1, 1]
            Z.append(A @ m); SS.append(A @ (P + np.diag(q * r.days_ahead)) @ A.T)
            rows.append(dict(home=st.teams[r.hi], away=st.teams[r.ai], date=str(r.date)))
        if not rows:
            return []
        Z, SS = np.array(Z), np.array(SS)
        pr = outcome_probs_mc(Z, SS, hp["r"], n_draws=256)
        g = np.arange(11)
        for k, row in enumerate(rows):
            lh = np.exp(Z[k, 0]) * _sig(Z[k, 2]); la = np.exp(Z[k, 1]) * _sig(Z[k, 3])
            ph = nbinom.pmf(g, hp["r"], hp["r"] / (hp["r"] + lh)); pa = nbinom.pmf(g, hp["r"], hp["r"] / (hp["r"] + la))
            J = np.outer(ph, pa); best = np.unravel_index(np.argmax(J), J.shape)
            row.update(p_home=float(pr[k, 0]), p_draw=float(pr[k, 1]), p_away=float(pr[k, 2]), xg_home=float(lh), xg_away=float(la),
                       likely_score=[int(best[0]), int(best[1])], p_likely=float(J[best]),
                       home_name=names.get(row["home"]), away_name=names.get(row["away"]))
        return rows
