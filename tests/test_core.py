import numpy as np
import pandas as pd
import pytest

from cadlag.eval import odds, scores
from cadlag.leagues import CRIT_CODE, LEAGUES
from cadlag.model.dixon_coles import outcome_probs, predictive_grid, score_grid
from cadlag.sim import tiebreak
from cadlag.sim.season import sample_dc


# ---------------------------------------------------------------- Dixon–Coles
@pytest.mark.parametrize("lam,mu,rho", [(1.5, 1.1, -0.1), (0.4, 2.7, 0.05), (2.0, 0.3, -0.2)])
def test_dc_conserva_masa(lam, mu, rho):
    g = score_grid(lam, mu, rho, maxg=40)
    assert abs(g.sum() - 1.0) < 1e-12


def test_predictiva_con_S_cero_es_plugin():
    z = np.array([[0.3, 0.1]])
    g0 = predictive_grid(z, np.zeros((1, 2, 2)), -0.05)[0]
    g1 = score_grid(np.exp(0.3), np.exp(0.1), -0.05)
    assert np.allclose(g0, g1, atol=1e-12)


def test_predictiva_mas_dispersa_que_plugin():
    """La mezcla sobre la incertidumbre de z aumenta la varianza de la diferencia de goles."""
    z = np.array([[0.4, 0.0]])
    S = np.array([[[0.05, 0.0], [0.0, 0.05]]])
    gp = predictive_grid(z, S, 0.0)[0]
    g0 = predictive_grid(z, np.zeros_like(S), 0.0)[0]
    k = np.arange(gp.shape[0])
    D = k[:, None] - k[None, :]
    var = lambda g: (g * D ** 2).sum() - (g * D).sum() ** 2
    assert var(gp) > var(g0)


def test_muestreo_dc_coincide_con_pmf():
    rng = np.random.default_rng(1)
    lam = np.full(400000, 1.2)
    nu = np.full(400000, 0.9)
    x, y = sample_dc(lam, nu, -0.12, rng)
    g = score_grid(1.2, 0.9, -0.12)
    for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
        emp = np.mean((x == a) & (y == b))
        assert abs(emp - g[a, b]) < 4 * np.sqrt(g[a, b] / len(x))


# ---------------------------------------------------------------- puntuación
def test_rps_y_brier_cero_si_acierta_con_certeza():
    p = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    o = np.array([0, 1])
    assert scores.rps(p, o).max() == 0 and scores.brier(p, o).max() == 0


def test_rps_es_propia_en_promedio():
    q = np.array([0.5, 0.3, 0.2])
    grid = np.linspace(0.01, 0.98, 60)
    best, bestv = None, np.inf
    for a in grid:
        for b in grid:
            if a + b >= 1:
                continue
            p = np.array([[a, b, 1 - a - b]])
            v = sum(q[o] * scores.rps(p, np.array([o]))[0] for o in range(3))
            if v < bestv:
                best, bestv = (a, b), v
    assert abs(best[0] - 0.5) < 0.03 and abs(best[1] - 0.3) < 0.03


# ---------------------------------------------------------------- de-vig
def test_shin_suma_uno_y_z_positivo():
    o = np.array([[1.80, 3.60, 4.50]])
    p, z = odds.shin(o, return_z=True)
    assert abs(p.sum() - 1) < 1e-9 and 0 < z[0] < 0.1
    # Shin corrige más a los "longshots" que el método multiplicativo
    pm = odds.multiplicative(o)
    assert p[0, 2] < pm[0, 2] and p[0, 0] > pm[0, 0]


# ---------------------------------------------------------------- desempates
def _rank(league, pts, gd, gf, w=None, agf=None, h2h=None):
    n = len(pts)
    z = np.zeros(n)
    hp = np.zeros((n, n)) if h2h is None else h2h
    crit = np.array([CRIT_CODE[c] for c in LEAGUES[league].tiebreak], np.int64)
    order = tiebreak.rank_one(np.array(pts, float), np.array(gd, float), np.array(gf, float),
                              z if w is None else np.array(w, float), z if agf is None else np.array(agf, float),
                              hp, np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)),
                              np.linspace(0.9, 0.1, n), crit, LEAGUES[league].h2h_reapply)
    return list(order)


def test_premier_diferencia_de_gol_antes_que_h2h():
    h2h = np.zeros((2, 2)); h2h[1, 0] = 6  # el equipo 1 le ganó los dos partidos al 0
    assert _rank("eng.1", [70, 70], [30, 20], [60, 60], h2h=h2h) == [0, 1]


def test_laliga_h2h_antes_que_diferencia_de_gol():
    h2h = np.zeros((2, 2)); h2h[1, 0] = 6
    assert _rank("esp.1", [70, 70], [30, 20], [60, 60], h2h=h2h) == [1, 0]


def test_laliga_minitabla_de_tres_y_recalculo():
    # A, B, C empatados en puntos. Minitabla: A 6, B 3, C 3 -> A primero; B y C se
    # vuelven a desempatar entre ellos: B le ganó a C dos veces.
    h = np.zeros((3, 3))
    h[0, 1] = 3; h[0, 2] = 3          # A: 6 (ganó a B una vez y a C una vez)
    h[1, 0] = 0; h[1, 2] = 6          # B: le ganó dos veces a C
    h[2, 0] = 3; h[2, 1] = 0          # C: le ganó una vez a A
    # A=6, B=6, C=3 en la minitabla de tres -> A y B empatan -> recalculo A vs B: A 3, B 0
    assert _rank("esp.1", [50, 50, 50], [0, 0, 0], [0, 0, 0], h2h=h) == [0, 1, 2]


def test_brasil_victorias_primero():
    assert _rank("bra.1", [60, 60], [10, 30], [50, 50], w=[18, 16]) == [0, 1]


# ---------------------------------------------------------------- Hodge
def test_hodge_completo_sin_armonico_y_ortogonal():
    from cadlag.model.hodge import build_flows, decompose
    rng = np.random.default_rng(0)
    n = 8
    h, a = np.array([(i, j) for i in range(n) for j in range(n) if i != j]).T
    gd = rng.poisson(1.4, len(h)) - rng.poisson(1.1, len(h))
    e, Y, W = build_flows(h, a, gd.astype(float))
    d = decompose(n, e, Y, W)
    assert d["share_harmonic"] < 1e-8
    assert abs(d["share_grad"] + d["share_curl"] + d["share_harmonic"] - 1) < 1e-8


def test_hodge_ciclo_sin_triangulos_es_armonico():
    from cadlag.model.hodge import decompose
    # ciclo 0->1->2->3->0 de 4 nodos sin cuerdas: flujo circular puro es armónico
    e = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    Y = np.array([1.0, 1.0, 1.0, -1.0])
    d = decompose(4, e, Y, np.ones(4))
    assert d["share_harmonic"] > 0.999
