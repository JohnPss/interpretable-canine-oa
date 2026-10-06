"""Testes do módulo comum (python -m pytest test_modelagem.py -q). Só dado sintético."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

import modelagem as mc


def _sintetico(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t_ev = rng.exponential(1 / (0.2 * np.exp(0.7 * x)))
    t_c = rng.uniform(0.2, 5, size=n)
    tempo = np.minimum(t_ev, t_c)
    evento = (t_ev <= t_c).astype(int)
    return x, tempo, evento


def test_km_igual_lifelines():
    _, t, e = _sintetico()
    km = KaplanMeierFitter().fit(t, e)
    for h in mc.HORIZONTES:
        assert abs(mc.km_risco(t, e, h) - (1 - km.predict(h))) < 1e-10


def test_risco_de_cox_igual_lifelines():
    x, t, e = _sintetico()
    d = pd.DataFrame({"x": x, "tempo": t, "evento": e})
    cph = CoxPHFitter().fit(d, "tempo", "evento")
    lp = cph.predict_log_partial_hazard(d[["x"]]).to_numpy()
    r_nosso = mc.risco_de_cox(lp, t, e, lp)
    r_lifelines = 1 - cph.predict_survival_function(d[["x"]], times=list(mc.HORIZONTES)).to_numpy().T
    assert np.abs(r_nosso - r_lifelines).max() < 1e-3


def test_nll_parcial_igual_lifelines():
    x, t, e = _sintetico()
    t = np.round(t, 1)                                   # força empates
    d = pd.DataFrame({"x": x, "tempo": t, "evento": e})
    cph = CoxPHFitter(baseline_estimation_method="breslow").fit(d, "tempo", "evento")
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    sk = CoxPHSurvivalAnalysis(ties="breslow").fit(d[["x"]], Surv.from_arrays(e.astype(bool), t))
    lp = sk.coef_[0] * x
    # log-verossimilhança de Breslow direta, O(n^2), como referência
    ref = -sum(lp[i] - np.log(np.exp(lp[t >= t[i]]).sum()) for i in np.flatnonzero(e == 1)) / e.sum()
    assert abs(mc.nll_parcial(lp, t, e) - ref) < 1e-9


def test_c_index_igual_lifelines():
    x, t, e = _sintetico()
    m = mc._metricas_conjunto(mc.Surv.from_arrays(e.astype(bool), t), t, e, x,
                              np.tile(x[:, None], (1, len(mc.HORIZONTES))))
    assert abs(m[("c_index", None)] - concordance_index(t, -x, e)) < 1e-6


def test_folds_agrupados_e_estratificados():
    rng = np.random.default_rng(1)
    n = 5000
    grupo = rng.integers(0, 4000, n)                 # tutores com vários cães
    estrato = rng.integers(0, 8, n)
    f = mc._atribuir_folds(estrato, grupo, 5, 42)
    for g in np.unique(grupo):
        assert len(np.unique(f[grupo == g])) == 1    # tutor nunca dividido
    props = pd.crosstab(f, estrato, normalize="columns")
    assert (props.max() - props.min()).max() < 0.03


def test_faixas():
    assert list(mc.faixa_etaria([0, 6.99, 7, 9.99, 10, 12.99, 13, 20])) == \
        ["< 7 anos"] * 2 + ["7–10 anos"] * 2 + ["10–13 anos"] * 2 + ["13+ anos"] * 2


def test_calibracao_por_grupo():
    _, t, e = _sintetico()
    n = len(t)
    rng = np.random.default_rng(3)
    idade = rng.uniform(0, 16, n)
    risco = np.sort(rng.uniform(0, 1, (n, len(mc.HORIZONTES))), axis=1)
    dados = pd.DataFrame({"dog_id": np.arange(n), "particao": "teste", "tempo": t, "evento": e, "idade": idade})
    pred = pd.DataFrame(risco, columns=mc.COLS_RISCO).assign(dog_id=np.arange(n))
    cal = mc.calibracao_por_grupo(pred, dados=dados)
    g = cal[cal.estrato == "geral"]
    assert len(g) == 10 and g.n.sum() == n and g.n.max() - g.n.min() <= 1
    assert (np.diff(g.previsto) > 0).all()                       # grupos em ordem de risco
    i2 = mc.HORIZONTES.index(mc.H_CALIB)
    k1 = np.argsort(risco[:, i2], kind="stable")[: g.n.iloc[0]]  # 1º decil = menores riscos
    assert np.isclose(g.observado.iloc[0], mc.km_risco(t[k1], e[k1], mc.H_CALIB))
    assert cal[cal.estrato != "geral"].groupby("estrato").size().eq(5).all()


if __name__ == "__main__":  # roda sem pytest: python test_modelagem.py
    import traceback
    falhas = 0
    for nome, f in sorted((n, g) for n, g in globals().items() if n.startswith("test_")):
        try:
            f()
            print("OK   ", nome)
        except Exception:
            falhas += 1
            print("FALHA", nome)
            traceback.print_exc()
    raise SystemExit(falhas)
