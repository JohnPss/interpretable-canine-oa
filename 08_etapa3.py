"""Etapa III — resultados, confiabilidade, vieses e interpretabilidade (um script, seções cacheadas).

Seções (rodar todas ou --secoes a,b,...):
  cv         CV 5-fold agrupada por tutor e estratificada, repetida 3x (sementes 142-144) no TREINO;
             média ± dp entre as 15 partições; C-index de ajuste (in-sample) para o gap treino/validação.
  teste      Ajuste de todos os modelos no treino inteiro, avaliação ÚNICA no teste com IC 95% por bootstrap
             de tutores (mesmas réplicas para todos os modelos → diferenças pareadas); inclinação de calibração.
  sementes   XGBoost (hiperparâmetros fixos) com 10 sementes → média ± dp no teste.
  clf2a      Leitura binária em 2 anos com pesos IPCW: matriz de confusão, sensibilidade, especificidade, VPP, F1,
             acurácia; limiar de Youden escolhido nas predições fora-do-fold do TREINO; baseline classe majoritária;
             curvas ROC e PR ponderadas; calibração por decil.
  vieses     Seleção (incluídos × excluídos sem seguimento: SMD e IPW), censura (simétrica, modelo de censura),
             datas estimadas, landmark de 1 ano, validação temporal, renda/escolaridade, versão da dieta,
             subgrupos, razões de risco do Cox em especificações aninhadas (confusão / causalidade reversa).
  vazamento  Rótulos permutados (3 sementes) e C-index univariado de cada atributo no teste.
  shap       TreeSHAP no teste: importância por atributo e grupo, estabilidade entre os 5 modelos de fold,
             concordância com as contribuições do Cox, interações a priori, modelo sem saúde.
  tex        Tabelas LaTeX e cópia das figuras para ../Artigo/artigo/.

Reprodutibilidade: semente global modelagem.SEED = 42; split/folds de data/processed/split.csv e folds.csv
(gerados por modelagem.py); hiperparâmetros de resultados/xgb_cox_hiperparametros.json (07). Versões em
requirements.txt. Uso: python 08_etapa3.py [--secoes cv,teste,...] [--n-boot 1000] [--jobs 6]
Saídas: resultados/etapa3/*.csv|json, figuras/etapa3/*.pdf, ../Artigo/artigo/{tabelas,figuras}/.
"""
import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

import features_config as fc
import modelagem as mc

warnings.filterwarnings("ignore")
fc.CATEGORICAS += ["renda", "escolaridade"]          # só nas sensibilidades de confusão socioeconômica

OUT = mc.RES / "etapa3"
FIG = mc.ROOT / "figuras" / "etapa3"
ART = mc.ROOT.parent / "Artigo" / "artigo"
HP = json.loads((mc.RES / "xgb_cox_hiperparametros.json").read_text())["modelos"]
SINAIS = ["peso_mudanca_12m", "apetite", "subpeso_ever", "sobrepeso_ever"]
T0 = time.time()


def virgula(*axes):
    """Separador decimal brasileiro nos eixos."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
    for ax in axes:
        ax.xaxis.set_major_formatter(f)
        ax.yaxis.set_major_formatter(f)


def br(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
# Modelos
# ------------------------------------------------------------------------------------------------
def _sem(cols, tirar):
    return [c for c in cols if c not in tirar]


XGB_P = ("xgb", HP["xgb_cox"]["params"], HP["xgb_cox"]["n_estimators"])
XGB_S = ("xgb", HP["xgb_cox_sem_saude"]["params"], HP["xgb_cox_sem_saude"]["n_estimators"])
MODELOS = {   # nome: (cols, (tipo, params/penalizador, n_arvores))
    "cox_idade": (["idade"], ("cox", 0.0, None)),
    "cox_idade_porte": (["idade", "porte"], ("cox", 0.0, None)),
    "cox_idade_porte_nsis": (["idade", "porte", "n_sistemas_doenca"], ("cox", 0.0, None)),
    "cox_estaveis": (fc.ESTAVEIS, ("cox", 0.01, None)),
    "cox_sem_saude_sinais": (_sem(fc.SEM_SAUDE, SINAIS), ("cox", 0.01, None)),
    "cox_sem_saude": (fc.SEM_SAUDE, ("cox", 0.01, None)),
    "cox_linear": (fc.MODELO_PRINCIPAL, ("cox", 0.01, None)),
    "xgb_estaveis": (fc.ESTAVEIS, XGB_S),
    "xgb_sem_saude_sinais": (_sem(fc.SEM_SAUDE, SINAIS), XGB_S),
    "xgb_sem_saude": (fc.SEM_SAUDE, XGB_S),
    "xgb_cox": (fc.MODELO_PRINCIPAL, XGB_P),
}
ROTULO = {"cox_idade": "Só idade", "cox_idade_porte": "Idade + porte",
          "cox_idade_porte_nsis": "Idade + porte + nº sistemas", "cox_estaveis": "Cox, só estáveis",
          "cox_sem_saude_sinais": "Cox, sem saúde e sinais", "cox_sem_saude": "Cox, sem saúde",
          "cox_linear": "Cox linear", "xgb_estaveis": "XGB, só estáveis",
          "xgb_sem_saude_sinais": "XGB, sem saúde e sinais", "xgb_sem_saude": "XGB, sem saúde",
          "xgb_cox": "XGBoost Cox"}


def _dm(X, y=None):
    nomes = [c.replace("<", "lt").replace("[", "(").replace("]", ")") for c in X.columns]
    return xgb.DMatrix(X, label=y, feature_names=nomes)


class Ajuste:
    """Modelo ajustado num conjunto de treino; escore = log-risco relativo; risco absoluto por Breslow."""

    def __init__(self, nome, d_tr, cols=None, espec=None, seed=mc.SEED):
        cols0, esp0 = MODELOS.get(nome, (None, None))
        self.cols = list(cols if cols is not None else cols0)
        self.tipo, self.par, self.n = espec if espec is not None else esp0
        d_tr = d_tr.reset_index(drop=True)
        if self.tipo == "cox":
            self.pre = mc.construir_preprocessador(self.cols, drop_primeira=True).fit(d_tr[self.cols])
            X = self.pre.transform(d_tr[self.cols]).assign(tempo=d_tr.tempo.to_numpy(), evento=d_tr.evento.to_numpy())
            self.mod = CoxPHFitter(penalizer=self.par).fit(X, "tempo", "evento")
        else:
            self.pre = mc.construir_preprocessador(self.cols, padronizar=False).fit(d_tr[self.cols])
            y = np.where(d_tr.evento.to_numpy() == 1, d_tr.tempo.to_numpy(), -d_tr.tempo.to_numpy())
            p = dict(objective="survival:cox", eval_metric="cox-nloglik", tree_method="hist", nthread=1,
                     seed=seed, **self.par)
            self.mod = xgb.train(p, _dm(self.pre.transform(d_tr[self.cols]), y), self.n)
        self.t_tr, self.e_tr = d_tr.tempo.to_numpy(), d_tr.evento.to_numpy()
        self.esc_tr = self.escore(d_tr)

    def escore(self, d):
        X = self.pre.transform(d[self.cols])
        if self.tipo == "cox":
            return self.mod.predict_log_partial_hazard(X).to_numpy()
        return self.mod.predict(_dm(X), output_margin=True)

    def prever(self, d):
        esc = self.escore(d)
        return esc, mc.risco_de_cox(esc, self.t_tr, self.e_tr, self.esc_tr)


# ------------------------------------------------------------------------------------------------
# Métricas
# ------------------------------------------------------------------------------------------------
def metricas(y_tr, t, e, esc, r):
    """modelagem._metricas_conjunto + C de Uno truncado em 3 anos. Chaves 'metrica@h'."""
    out = {f"{m}@{h:g}" if h is not None else m: v for (m, h), v in mc._metricas_conjunto(y_tr, t, e, esc, r).items()}
    try:
        out["uno_c@3"] = concordance_index_ipcw(y_tr, Surv.from_arrays(e.astype(bool), t), esc, tau=3)[0]
    except Exception:
        pass
    return out


def c_harrell(t, e, esc):
    return concordance_index_censored(e.astype(bool), t, esc)[0]


def c_intrafaixa(t, e, esc, idade):
    f = mc.faixa_etaria(idade)
    return float(np.mean([c_harrell(t[f == n], e[f == n], esc[f == n]) for *_, n in mc.FAIXAS]))


def ytr(d):
    return Surv.from_arrays(d.evento.astype(bool).to_numpy(), d.tempo.to_numpy())


def avaliar_simples(d_tr, d_te, esc, r):
    """Ponto (sem IC): métricas gerais + C intrafaixa médio."""
    t, e = d_te.tempo.to_numpy(), d_te.evento.to_numpy()
    m = metricas(ytr(d_tr), t, e, esc, r)
    m["c_intrafaixa"] = c_intrafaixa(t, e, esc, d_te.idade.to_numpy())
    m["n"], m["n_obitos"] = len(d_te), int(e.sum())
    return m


def carregar():
    df = mc.carregar_com_particao()
    h = pd.read_csv(mc.ROOT / "data/raw/DAP_2024_HLES_dog_owner_v1.0.csv",
                    usecols=["dog_id", "od_annual_income_range_usd", "od_max_education"], low_memory=False)
    renda = pd.cut(h.od_annual_income_range_usd.where(h.od_annual_income_range_usd != 98), [0, 2, 4, 6, 10],
                   labels=["<40k", "40-80k", "80-120k", "120k+"]).astype(object)
    h["renda"] = np.where(h.od_annual_income_range_usd == 98, "nao_resp", renda)
    h["renda"] = h["renda"].where(h.od_annual_income_range_usd.notna())
    h["escolaridade"] = pd.cut(h.od_max_education, [-1, 4, 7, 8, 11],
                               labels=["medio_ou_menos", "tecnico_ou_incompleto", "graduacao", "pos"]).astype(object)
    df = df.merge(h[["dog_id", "renda", "escolaridade"]], on="dog_id", how="left")
    ap = pd.read_csv(mc.PROC / "apuracao.csv")
    return df.merge(ap[["dog_id", "tempo_sim", "evento_sim", "incluido_sim"]], on="dog_id", how="left")


# ------------------------------------------------------------------------------------------------
# 1. CV repetida
# ------------------------------------------------------------------------------------------------
def _uma_particao(tr, rep, k, i_tr, i_val):
    a, v = tr.iloc[i_tr], tr.iloc[i_val]
    linhas = []
    for nome in MODELOS:
        aj = Ajuste(nome, a)
        esc, r = aj.prever(v)
        m = avaliar_simples(a, v, esc, r)
        m["c_ajuste"] = c_harrell(aj.t_tr, aj.e_tr, aj.esc_tr)
        m["c_intrafaixa_ajuste"] = c_intrafaixa(aj.t_tr, aj.e_tr, aj.esc_tr, a.idade.to_numpy())
        linhas.append(dict(modelo=nome, rep=rep, fold=k, **m))
    return linhas


def secao_cv(df, jobs):
    tr = df[df.particao == "treino"].reset_index(drop=True)
    tarefas = []
    for rep in range(3):
        f = mc._atribuir_folds(mc._estrato_split(tr), tr.owner_id.to_numpy(), mc.N_FOLDS, seed=mc.SEED + 100 + rep)
        tarefas += [(rep, k, np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(mc.N_FOLDS)]
    res = Parallel(n_jobs=jobs)(delayed(_uma_particao)(tr, *t) for t in tarefas)
    cv = pd.DataFrame([l for r in res for l in r])
    cv.to_csv(OUT / "cv_repetida.csv", index=False)
    resumo = cv.drop(columns=["rep", "fold"]).groupby("modelo").agg(["mean", "std"])
    resumo.columns = [f"{a}_{b}" for a, b in resumo.columns]
    resumo.reindex(list(MODELOS)).to_csv(OUT / "cv_resumo.csv")
    log("cv: " + resumo[["c_index_mean", "c_index_std", "c_intrafaixa_mean", "c_ajuste_mean"]].round(3).to_string())


# ------------------------------------------------------------------------------------------------
# 2. Teste: ajuste final, bootstrap pareado
# ------------------------------------------------------------------------------------------------
def _boot_rep(y_tr, s, preds, idx):
    t, e, idade = s.tempo.to_numpy()[idx], s.evento.to_numpy()[idx], s.idade.to_numpy()[idx]
    out = {}
    for nome, (esc, r) in preds.items():
        m = metricas(y_tr, t, e, esc[idx], r[idx])
        out[nome] = m
    return out


def avaliar_boot(d_tr, d_te, preds, n_boot, jobs, estratos=None, seed=mc.SEED):
    """preds: nome → (escore, risco) alinhados a d_te. Devolve (pontos+IC longo, réplicas por estrato)."""
    y_tr = ytr(d_tr)
    estratos = estratos or {e: (mc.faixa_etaria(d_te.idade) == e) if e != "geral" else np.ones(len(d_te), bool)
                            for e in mc.ESTRATOS}
    linhas, reps_all = [], {}
    for est, mask in estratos.items():
        s = d_te[mask].reset_index(drop=True)
        p = {n: (esc[mask], r[mask]) for n, (esc, r) in preds.items()}
        ponto = _boot_rep(y_tr, s, p, np.arange(len(s)))
        reps = mc.indices_bootstrap(s.owner_id.to_numpy(), n_boot, seed)
        boot = Parallel(n_jobs=jobs, batch_size=8)(delayed(_boot_rep)(y_tr, s, p, i) for i in reps)
        reps_all[est] = boot
        for nome in preds:
            for met, v in ponto[nome].items():
                b = np.array([x[nome].get(met, np.nan) for x in boot])
                linhas.append(dict(modelo=nome, estrato=est, metrica=met, valor=v, ic_lo=np.nanpercentile(b, 2.5),
                                   ic_hi=np.nanpercentile(b, 97.5), dp_boot=np.nanstd(b), n=len(s),
                                   n_obitos=int(s.evento.sum())))
    return pd.DataFrame(linhas), reps_all


def diferencas(reps_all, pontos, pares):
    linhas = []
    for a, b in pares:
        for est, boot in reps_all.items():
            for met in ["c_index", "uno_c@3", "auc_tempo@2", "brier@2"]:
                d = np.array([x[a].get(met, np.nan) - x[b].get(met, np.nan) for x in boot])
                pa = pontos.query("modelo==@a and estrato==@est and metrica==@met").valor
                pb = pontos.query("modelo==@b and estrato==@est and metrica==@met").valor
                if len(pa) and len(pb):
                    linhas.append(dict(modelo=a, referencia=b, estrato=est, metrica=met,
                                       diferenca=pa.iloc[0] - pb.iloc[0], ic_lo=np.nanpercentile(d, 2.5),
                                       ic_hi=np.nanpercentile(d, 97.5)))
    return pd.DataFrame(linhas)


def inclinacao_calibracao(t, e, esc):
    d = pd.DataFrame(dict(t=t, e=e, lp=esc))
    c = CoxPHFitter().fit(d, "t", "e")
    return float(c.params_["lp"]), *map(float, c.confidence_intervals_.loc["lp"].to_numpy())


def secao_teste(df, n_boot, jobs):
    tr, te = df[df.particao == "treino"].reset_index(drop=True), df[df.particao == "teste"].reset_index(drop=True)
    preds, ajuste_c, linhas_gap = {}, {}, []
    for nome in MODELOS:
        aj = Ajuste(nome, tr)
        esc, r = aj.prever(te)
        preds[nome] = (esc, r)
        p = pd.DataFrame(r, columns=mc.COLS_RISCO).assign(dog_id=te.dog_id.to_numpy(), escore=esc)
        p.to_csv(mc.PROC / f"pred_e3_{nome}_teste.csv", index=False)
        sl = inclinacao_calibracao(te.tempo.to_numpy(), te.evento.to_numpy(), esc)
        linhas_gap.append(dict(modelo=nome, c_ajuste=c_harrell(aj.t_tr, aj.e_tr, aj.esc_tr),
                               c_intrafaixa_ajuste=c_intrafaixa(aj.t_tr, aj.e_tr, aj.esc_tr, tr.idade.to_numpy()),
                               c_teste=c_harrell(te.tempo.to_numpy(), te.evento.to_numpy(), esc),
                               c_intrafaixa_teste=c_intrafaixa(te.tempo.to_numpy(), te.evento.to_numpy(), esc,
                                                               te.idade.to_numpy()),
                               inclinacao=sl[0], inclinacao_lo=sl[1], inclinacao_hi=sl[2]))
        log(f"teste: {nome} ajustado")
    # reprodutibilidade: o refit reproduz as predições gravadas pelo 07 / 06?
    rep = []
    for nome in ["xgb_cox", "cox_linear", "cox_idade", "xgb_cox_sem_saude", "cox_linear_sem_saude"]:
        alvo = {"xgb_cox_sem_saude": "xgb_sem_saude", "cox_linear_sem_saude": "cox_sem_saude"}.get(nome, nome)
        g = mc.carregar_predicoes(nome).query("particao=='teste'").set_index("dog_id").loc[te.dog_id]
        rep.append(dict(modelo=nome, corr=np.corrcoef(g.escore, preds[alvo][0])[0, 1],
                        dif_abs_max_risco2a=float(np.abs(g.risco_2a.to_numpy() - preds[alvo][1][:, 1]).max())))
    pd.DataFrame(rep).to_csv(OUT / "reprodutibilidade.csv", index=False)
    log("reprodutibilidade:\n" + pd.DataFrame(rep).to_string())
    pd.DataFrame(linhas_gap).to_csv(OUT / "gap_ajuste_teste.csv", index=False)
    pontos, reps_all = avaliar_boot(tr, te, preds, n_boot, jobs)
    pontos.to_csv(OUT / "teste_metricas.csv", index=False)
    pares = [("xgb_cox", "cox_linear"), ("xgb_cox", "cox_idade"), ("cox_linear", "cox_idade"),
             ("cox_idade_porte", "cox_idade"), ("cox_idade_porte_nsis", "cox_idade_porte"),
             ("cox_linear", "cox_idade_porte_nsis"), ("xgb_cox", "xgb_sem_saude"),
             ("xgb_sem_saude", "xgb_sem_saude_sinais"), ("xgb_sem_saude_sinais", "xgb_estaveis"),
             ("xgb_estaveis", "cox_idade"), ("xgb_estaveis", "cox_estaveis"), ("xgb_sem_saude", "cox_idade")]
    diferencas(reps_all, pontos, pares).to_csv(OUT / "teste_diferencas.csv", index=False)
    log("teste: " + pontos.query("metrica=='c_index'").pivot(index="modelo", columns="estrato", values="valor")
        .round(3).to_string())


# ------------------------------------------------------------------------------------------------
# 3. Sementes
# ------------------------------------------------------------------------------------------------
def secao_sementes(df, jobs):
    tr, te = df[df.particao == "treino"].reset_index(drop=True), df[df.particao == "teste"].reset_index(drop=True)

    def uma(s):
        aj = Ajuste("xgb_cox", tr, seed=s)
        esc, r = aj.prever(te)
        return dict(semente=s, **avaliar_simples(tr, te, esc, r))

    res = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(uma)(s) for s in range(1, 11)))
    res.to_csv(OUT / "sementes_xgb.csv", index=False)
    log("sementes: " + res.drop(columns="semente").agg(["mean", "std"]).round(4).to_string())


# ------------------------------------------------------------------------------------------------
# 4. Leitura binária em 2 anos (IPCW)
# ------------------------------------------------------------------------------------------------
def pesos_ipcw(y_tr, t, e, h=2.0):
    """Casos (óbito ≤ h): 1/G(T); controles (T > h): 1/G(h); censurados antes de h: 0."""
    cde = CensoringDistributionEstimator().fit(y_tr)
    caso = (t <= h) & (e == 1)
    ctrl = t > h
    w = np.zeros(len(t))
    w[caso] = 1 / np.clip(cde.predict_proba(t[caso]), 1e-6, None)
    w[ctrl] = 1 / cde.predict_proba(np.array([h]))[0]
    return caso, ctrl, w


def curvas(score, caso, ctrl, w):
    o = np.argsort(-score, kind="stable")
    wc, wn = np.where(caso, w, 0)[o], np.where(ctrl, w, 0)[o]
    tp, fp = np.cumsum(wc), np.cumsum(wn)
    tpr, fpr = tp / tp[-1], fp / fp[-1]
    prec = tp / np.maximum(tp + fp, 1e-12)
    auc = np.trapz(np.r_[0, tpr], np.r_[0, fpr])
    ap = np.sum(np.diff(np.r_[0, tpr]) * prec)
    return dict(fpr=fpr, tpr=tpr, prec=prec, auc=auc, ap=ap, limiares=score[o])


def matriz(score, lim, caso, ctrl, w):
    pos = score >= lim
    vp, fn = w[caso & pos].sum(), w[caso & ~pos].sum()
    fp, vn = w[ctrl & pos].sum(), w[ctrl & ~pos].sum()
    sens, esp = vp / (vp + fn), vn / (vn + fp)
    vpp = vp / (vp + fp) if vp + fp > 0 else np.nan
    f1 = 2 * vpp * sens / (vpp + sens) if vp > 0 else 0.0
    return dict(VP=vp, FN=fn, FP=fp, VN=vn, sensibilidade=sens, especificidade=esp, vpp=vpp, f1=f1,
                acuracia=(vp + vn) / (vp + fn + fp + vn), acuracia_balanceada=(sens + esp) / 2)


def secao_clf2a(df):
    tr, te = df[df.particao == "treino"].reset_index(drop=True), df[df.particao == "teste"].reset_index(drop=True)
    y_tr = ytr(tr)
    t, e = te.tempo.to_numpy(), te.evento.to_numpy()
    caso, ctrl, w = pesos_ipcw(y_tr, t, e)
    # OOF do treino (06/07) para escolher o limiar sem tocar o teste
    t_tr, e_tr = tr.tempo.to_numpy(), tr.evento.to_numpy()
    caso_tr, ctrl_tr, w_tr = pesos_ipcw(y_tr, t_tr, e_tr)
    linhas, curv = [], {}
    prev = w[caso].sum() / (w[caso].sum() + w[ctrl].sum())
    for nome in ["cox_idade", "cox_idade_porte", "cox_linear", "xgb_cox"]:
        oof = mc.carregar_predicoes(nome).query("particao=='treino'").set_index("dog_id").loc[tr.dog_id]
        c_tr = curvas(oof.risco_2a.to_numpy(), caso_tr, ctrl_tr, w_tr)
        j = np.argmax(c_tr["tpr"] - c_tr["fpr"])
        lim = c_tr["limiares"][j]
        r2 = pd.read_csv(mc.PROC / f"pred_e3_{nome}_teste.csv").set_index("dog_id").loc[te.dog_id].risco_2a.to_numpy()
        cv_ = curvas(r2, caso, ctrl, w)
        curv[nome] = cv_
        sub = te.idade.between(10, 13, inclusive="left").to_numpy()
        curv[nome + "_10_13"] = curvas(r2[sub], caso[sub], ctrl[sub], w[sub])
        linhas.append(dict(modelo=nome, limiar=lim, auc_ipcw=cv_["auc"], ap=cv_["ap"],
                           auc_10_13=curv[nome + "_10_13"]["auc"], **matriz(r2, lim, caso, ctrl, w)))
    linhas.append(dict(modelo="maioria", limiar=np.inf, auc_ipcw=0.5, ap=prev,
                       **matriz(np.zeros(len(t)), 1.0, caso, ctrl, w)))
    res = pd.DataFrame(linhas)
    # contraste: análise ingênua de casos completos (descarta censurados antes de 2 anos)
    conhecidos = caso | ctrl
    extra = dict(prevalencia_ipcw=prev, prevalencia_ingenua=caso.sum() / conhecidos.sum(),
                 km_2a_teste=mc.km_risco(t, e, 2.0), n_teste=len(t), n_casos=int(caso.sum()),
                 n_controles=int(ctrl.sum()), n_indeterminados=int((~conhecidos).sum()),
                 obitos_total=int(e.sum()), prop_obitos_total=float(e.mean()),
                 km_2a_por_faixa={n: mc.km_risco(t[mc.faixa_etaria(te.idade) == n], e[mc.faixa_etaria(te.idade) == n], 2.0)
                                  for *_, n in mc.FAIXAS})
    res.to_csv(OUT / "clf2a.csv", index=False)
    (OUT / "clf2a_extra.json").write_text(json.dumps(extra, indent=1))
    log("clf2a:\n" + res.round(3).to_string() + "\n" + json.dumps(extra, indent=1))
    # figuras ROC/PR
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    cores = {"cox_idade": "#9e9e9e", "cox_idade_porte": "#6a51a3", "cox_linear": "#2171b5", "xgb_cox": "#d94801"}
    estilos = {"cox_idade": ":", "cox_idade_porte": "-.", "cox_linear": "--", "xgb_cox": "-"}
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))
    for nome in cores:
        for i, chave in enumerate([nome, nome + "_10_13"]):
            c = curv[chave]
            ax[i].plot(np.r_[0, c["fpr"]], np.r_[0, c["tpr"]], estilos[nome], color=cores[nome], lw=1.1,
                       label=f"{ROTULO[nome]} ({br(c['auc'])})")
        c = curv[nome]
        ax[2].plot(c["tpr"], c["prec"], estilos[nome], color=cores[nome], lw=1.1, label=f"{ROTULO[nome]} ({br(c['ap'])})")
    for i, tit in enumerate(["(a) ROC, todos os cães", "(b) ROC, 10–13 anos"]):
        ax[i].plot([0, 1], [0, 1], color="k", lw=0.5)
        ax[i].set(xlabel="1 − especificidade", ylabel="Sensibilidade", title=tit, xlim=(0, 1), ylim=(0, 1))
        ax[i].legend(fontsize=5.8, loc="lower right", frameon=False)
    ax[2].axhline(prev, color="k", lw=0.5)
    ax[2].set(xlabel="Sensibilidade", ylabel="VPP (precisão)", title="(c) Precisão–sensibilidade", xlim=(0, 1),
              ylim=(0, 1))
    ax[2].legend(fontsize=5.8, loc="upper right", frameon=False)
    virgula(*ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "roc_pr_2a.pdf")
    plt.close(fig)
    # calibração por decil (2 anos)
    cal = []
    for nome in ["cox_idade", "cox_linear", "xgb_cox"]:
        p = pd.read_csv(mc.PROC / f"pred_e3_{nome}_teste.csv")
        c = mc.calibracao_por_grupo(p, dados=df).query("estrato=='geral'").assign(modelo=nome)
        cal.append(c)
    cal = pd.concat(cal)
    cal.to_csv(OUT / "calibracao_decis.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for nome in ["cox_idade", "cox_linear", "xgb_cox"]:
        c = cal[cal.modelo == nome]
        ax.plot(c.previsto, c.observado, "o" + estilos[nome], color=cores[nome], ms=3, lw=0.9, label=ROTULO[nome])
    ax.plot([0, 1], [0, 1], color="k", lw=0.5)
    m = cal[["previsto", "observado"]].to_numpy().max() * 1.05
    ax.set(xlim=(0, m), ylim=(0, m), xlabel="Risco previsto em 2 anos (média do decil)",
           ylabel="Mortalidade observada (1 − KM)")
    ax.legend(frameon=False, fontsize=7)
    virgula(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "calibracao_2a.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# 5. Vieses
# ------------------------------------------------------------------------------------------------
def c_ponderado(t, e, esc, w):
    """Harrell com pesos por cão (par i,j pesa w_i w_j); empates no escore contam 0,5."""
    num = den = 0.0
    idx = np.flatnonzero(e == 1)
    for lote in np.array_split(idx, max(1, len(idx) // 200)):
        comp = t[None, :] > t[lote, None]
        ww = w[lote, None] * w[None, :] * comp
        dif = esc[lote, None] - esc[None, :]
        num += (ww * ((dif > 0) + 0.5 * (dif == 0))).sum()
        den += ww.sum()
    return num / den


def km_ponderado(t, e, w, h):
    s = 1.0
    for u in np.unique(t[(e == 1) & (t <= h)]):
        s *= 1 - w[(t == u) & (e == 1)].sum() / w[t >= u].sum()
    return 1 - s


def secao_vieses(df, jobs):
    from sklearn.linear_model import LogisticRegression
    tr, te = df[df.particao == "treino"].reset_index(drop=True), df[df.particao == "teste"].reset_index(drop=True)
    sens = []

    def registrar(cenario, modelo, d_tr, d_te, esc, r, **extra):
        m = avaliar_simples(d_tr, d_te, esc, r)
        sens.append(dict(cenario=cenario, modelo=modelo, **m, **extra))

    base = {n: pd.read_csv(mc.PROC / f"pred_e3_{n}_teste.csv").set_index("dog_id").loc[te.dog_id]
            for n in ["cox_idade", "cox_linear", "xgb_cox"]}
    for n, p in base.items():
        registrar("principal", n, tr, te, p.escore.to_numpy(), p[mc.COLS_RISCO].to_numpy())

    # --- 5a. seleção: incluídos × excluídos sem seguimento ---
    pop = pd.read_csv(mc.PROC / "populacao.csv")
    eleg = pop[(pop.incluido == 1) | (pop.motivo_exclusao == "sem_seguimento")].copy()
    ov = pd.read_csv(mc.ROOT / "data/raw/DAP_2024_DogOverview_v1.1.csv",
                     usecols=["dog_id", "Sex_Class_at_HLES", "Breed_Size_Class_at_HLES", "Breed_Status"])
    ov["porte_classe"] = ov.Breed_Size_Class_at_HLES.str.split(" ").str[0].replace({"Toy": "Toy/Small"})
    ov["castrado"] = ov.Sex_Class_at_HLES.str.contains("spayed|neutered").astype(float)
    ov["femea"] = ov.Sex_Class_at_HLES.str.startswith("Female").astype(float)
    ov["raca_pura"] = (ov.Breed_Status == "Purebred").astype(float)
    renda = df[["dog_id", "renda", "escolaridade"]]
    h = pd.read_csv(mc.ROOT / "data/raw/DAP_2024_HLES_dog_owner_v1.0.csv",
                    usecols=["dog_id", "od_annual_income_range_usd", "od_max_education"], low_memory=False)
    eleg = eleg.merge(ov, on="dog_id", how="left").merge(h, on="dog_id", how="left")
    eleg["renda_120k"] = (eleg.od_annual_income_range_usd.between(7, 10)).astype(float)
    eleg["renda_nao_resp"] = (eleg.od_annual_income_range_usd == 98).astype(float)
    eleg["graduacao_ou_mais"] = (eleg.od_max_education >= 8).astype(float)
    eleg["foundation"] = (eleg.Cohort == "Foundation").astype(float)
    eleg["entrada_2023_24"] = (eleg.hles_ano >= 2023).astype(float)
    for p_ in ["Toy/Small", "Medium", "Standard", "Large", "Giant"]:
        eleg[f"porte_{p_}"] = (eleg.porte_classe == p_).astype(float)
    vars_smd = ["idade_hles", "femea", "castrado", "raca_pura", "foundation", "entrada_2023_24", "renda_120k",
                "renda_nao_resp", "graduacao_ou_mais"] + [f"porte_{p_}" for p_ in ["Toy/Small", "Medium", "Standard",
                                                                                   "Large", "Giant"]]
    smd = []
    for v in vars_smd:
        a, b = eleg.loc[eleg.incluido == 1, v], eleg.loc[eleg.incluido == 0, v]
        smd.append(dict(variavel=v, incluidos=a.mean(), excluidos=b.mean(),
                        smd=(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)))
    smd = pd.DataFrame(smd)
    smd.to_csv(OUT / "selecao_smd.csv", index=False)
    anos = eleg.groupby("hles_ano").incluido.agg(["size", "mean"]).reset_index()
    anos.to_csv(OUT / "selecao_por_ano.csv", index=False)
    log("seleção SMD:\n" + smd.round(3).to_string() + "\n" + anos.to_string())
    # IPW: P(ter seguimento | covariáveis de entrada)
    X = pd.get_dummies(eleg[["idade_hles", "femea", "castrado", "raca_pura", "foundation", "porte_classe"]]
                       .assign(ano=eleg.hles_ano.astype(str),
                               renda=eleg.od_annual_income_range_usd.fillna(-1).astype(int).astype(str),
                               educ=eleg.od_max_education.fillna(-1).astype(int).astype(str)),
                       columns=["porte_classe", "ano", "renda", "educ"], dummy_na=True).astype(float).fillna(0)
    X["idade2"] = X.idade_hles ** 2
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, eleg.incluido)
    eleg["p_incl"] = lr.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc_sel = roc_auc_score(eleg.incluido, eleg.p_incl)
    wmap = eleg.set_index("dog_id").p_incl
    w = eleg.incluido.mean() / wmap.loc[te.dog_id].to_numpy()
    w = np.clip(w, None, np.quantile(w, 0.99))
    t, e = te.tempo.to_numpy(), te.evento.to_numpy()
    for n, p in base.items():
        esc = p.escore.to_numpy()
        sens.append(dict(cenario="ipw_selecao", modelo=n, c_index=c_ponderado(t, e, esc, w),
                         **{"cal_oe@2": km_ponderado(t, e, w, 2.0) / np.average(p.risco_2a, weights=w)},
                         c_intrafaixa=np.mean([c_ponderado(t[f], e[f], esc[f], w[f]) for f in
                                               [mc.faixa_etaria(te.idade) == nn for *_, nn in mc.FAIXAS]]),
                         n=len(te), n_obitos=int(e.sum()), auc_selecao=auc_sel, peso_max=float(w.max())))

    # --- 5b. censura simétrica ---
    def sim(d):
        d = d[d.incluido_sim == 1].copy()
        d["tempo"], d["evento"] = d.tempo_sim, d.evento_sim
        return d.reset_index(drop=True)

    tr_s, te_s = sim(tr), sim(te)
    for n, p in base.items():
        q = p.loc[te_s.dog_id]
        registrar("censura_simetrica_avaliacao", n, tr_s, te_s, q.escore.to_numpy(), q[mc.COLS_RISCO].to_numpy())
    for n in ["cox_idade", "cox_linear", "xgb_cox"]:
        esc, r = Ajuste(n, tr_s).prever(te_s)
        registrar("censura_simetrica_retreino", n, tr_s, te_s, esc, r)
    log("vieses: censura simétrica ok")
    # modelo de censura: a censura depende dos atributos?
    trc, tec = tr.copy(), te.copy()
    trc["evento"], tec["evento"] = 1 - tr.evento, 1 - te.evento
    ajc = Ajuste("cens", trc, cols=fc.MODELO_PRINCIPAL + ["renda"], espec=("cox", 0.01, None))
    esc_c = ajc.escore(tec)
    coef = ajc.mod.params_[~ajc.mod.params_.index.str.contains("infrequent")].sort_values(key=np.abs, ascending=False).head(8)
    cens = dict(c_censura_teste=c_harrell(tec.tempo.to_numpy(), tec.evento.to_numpy(), esc_c),
                top_hr={k: float(np.exp(v)) for k, v in coef.items()},
                prop_censurados_antes_2a=float(((te.evento == 0) & (te.tempo < 2)).mean()))

    # --- 5c. sem datas de óbito estimadas ---
    tr_d, te_d = tr[tr.obito_data_estimada == 0].reset_index(drop=True), te[te.obito_data_estimada == 0].reset_index(drop=True)
    for n in ["cox_idade", "cox_linear", "xgb_cox"]:
        esc, r = Ajuste(n, tr_d).prever(te_d)
        registrar("sem_datas_estimadas", n, tr_d, te_d, esc, r)

    # --- 5d. landmark de 1 ano (causalidade reversa: doença prodrômica no início do seguimento) ---
    def lm(d):
        d = d[d.tempo > 1].copy()
        d["tempo"] = d.tempo - 1
        return d.reset_index(drop=True)

    tr_l, te_l = lm(tr), lm(te)
    for n, p in base.items():
        q = p.loc[te_l.dog_id]
        registrar("landmark_1a_avaliacao", n, tr_l, te_l, q.escore.to_numpy(), q[mc.COLS_RISCO].to_numpy())

    # --- 5e. validação temporal: treino com entradas ≤ 2021, teste com entradas ≥ 2022 ---
    tr_t, te_t = tr[tr.hles_ano <= 2021].reset_index(drop=True), te[te.hles_ano >= 2022].reset_index(drop=True)
    for n in ["cox_idade", "cox_linear", "xgb_cox"]:
        esc, r = Ajuste(n, tr_t).prever(te_t)
        registrar("temporal_2022_24", n, tr_t, te_t, esc, r)
    # mesmo teste, modelo do treino completo (isola o efeito "deriva" do efeito "menos treino")
    for n, p in base.items():
        q = p.loc[te_t.dog_id]
        registrar("subconjunto_2022_24_modelo_completo", n, tr, te_t, q.escore.to_numpy(), q[mc.COLS_RISCO].to_numpy())

    # --- 5f. renda/escolaridade e versão da dieta como atributos adicionais ---
    for cen, extra_cols in [("mais_renda_escolaridade", ["renda", "escolaridade"]), ("mais_dieta_versao", ["dieta_versao"])]:
        esc, r = Ajuste("xgb_cox", tr, cols=fc.MODELO_PRINCIPAL + extra_cols).prever(te)
        registrar(cen, "xgb_cox", tr, te, esc, r)
        esc, r = Ajuste("cox_linear", tr, cols=fc.MODELO_PRINCIPAL + extra_cols).prever(te)
        registrar(cen, "cox_linear", tr, te, esc, r)
    pd.DataFrame(sens).to_csv(OUT / "sensibilidades.csv", index=False)
    log("vieses: sensibilidades\n" + pd.DataFrame(sens)[["cenario", "modelo", "c_index", "c_intrafaixa", "cal_oe@2",
                                                          "n", "n_obitos"]].round(3).to_string())

    # --- 5g. subgrupos (teste): C e O/E em 2 anos, XGB × só idade ---
    grupos = {
        "Coorte": te.Cohort, "Idade informada": te.AgeDOB_Estimation_Method.map(
            {"Estimated using owner-reported birth year and month": "ano e mês de nasc.",
             "Estimated using owner-reported birth year": "só ano de nasc.",
             "Estimated using owner-reported age": "idade declarada"}),
        "Raça": te.raca_pura.map({1: "pura", 0: "SRD/mista"}), "Sexo": te.sexo.map({1: "macho", 0: "fêmea"}),
        "Porte": te.porte.map({0: "toy/pequeno", 1: "médio", 2: "padrão", 3: "grande", 4: "gigante"}),
        "Ano de entrada": pd.cut(te.hles_ano, [0, 2020, 2021, 2030], labels=["2019–20", "2021", "2022–24"]).astype(str),
        "Versão da dieta": te.dieta_versao.map({"df": "até 2022", "dt": "após 2022"}),
        "Renda familiar": te.renda.fillna("faltante").map({"<40k": "< US$ 40 mil", "40-80k": "US$ 40–80 mil",
                                                           "80-120k": "US$ 80–120 mil", "120k+": "≥ US$ 120 mil",
                                                           "nao_resp": "não respondeu", "faltante": "não respondeu"}),
    }
    y_tr = ytr(tr)
    sub = []
    for g, val in grupos.items():
        for nivel in val.dropna().unique():
            m = (val == nivel).to_numpy()
            s = te[m]
            t, e = s.tempo.to_numpy(), s.evento.to_numpy()
            linha = dict(grupo=g, nivel=nivel, n=int(m.sum()), n_obitos=int(e.sum()))
            for n in ["xgb_cox", "cox_idade"]:
                p = base[n][m]
                linha[f"c_{n}"] = c_harrell(t, e, p.escore.to_numpy())
                linha[f"oe_{n}"] = mc.km_risco(t, e, 2.0) / p.risco_2a.mean()
            ss = s[s.incluido_sim == 1]
            linha["oe_xgb_cox_simetrica"] = mc.km_risco(ss.tempo_sim, ss.evento_sim, 2.0) / base["xgb_cox"].loc[ss.dog_id].risco_2a.mean()
            reps = mc.indices_bootstrap(s.owner_id.to_numpy(), 200, mc.SEED)
            ex, ei = base["xgb_cox"][m].escore.to_numpy(), base["cox_idade"][m].escore.to_numpy()
            b = np.array([c_harrell(t[i], e[i], ex[i]) - c_harrell(t[i], e[i], ei[i]) for i in reps])
            linha.update(dc=linha["c_xgb_cox"] - linha["c_cox_idade"], dc_lo=np.percentile(b, 2.5),
                         dc_hi=np.percentile(b, 97.5))
            bx = np.array([c_harrell(t[i], e[i], ex[i]) for i in reps])
            linha.update(c_xgb_lo=np.percentile(bx, 2.5), c_xgb_hi=np.percentile(bx, 97.5))
            sub.append(linha)
    sub = pd.DataFrame(sub)
    sub.to_csv(OUT / "subgrupos.csv", index=False)
    log("subgrupos:\n" + sub.round(3).to_string())

    # --- 5h. HR do Cox em especificações aninhadas (confusão e causalidade reversa) ---
    ref = {"castracao": "intacto", "peso_mudanca_12m": "estavel", "apetite": "normal", "alimentacao": "2x",
           "dieta_principal": "comercial_seca", "sobrepeso_ever": "nao", "subpeso_ever": "nao",
           "dieta_consistencia": "muito", "regiao": "div_5", "renda": "80-120k", "escolaridade": "graduacao"}

    def desenho(d, cols, ajuste=None):
        X = pd.DataFrame(index=d.index)
        num = [c for c in cols if c in fc.NUMERICAS + fc.ORDINAIS]
        stats = ajuste or {c: (d[c].median(), d[c].fillna(d[c].median()).std()) for c in num}
        for c in num:
            X[c] = (d[c].fillna(stats[c][0]) - d[c].fillna(stats[c][0]).mean()) / stats[c][1]
        for c in cols:
            if c in fc.BINARIAS:
                X[c] = d[c].fillna(0)
            elif c in fc.CATEGORICAS:
                v = d[c].fillna("faltante").astype(str)
                for nivel in sorted(v.unique()):
                    if nivel != ref.get(c) and (v == nivel).mean() >= 0.005:
                        X[f"{c}={nivel}"] = (v == nivel).astype(float)
        return X

    especs = {"A_estaveis_estilo": fc.SEM_SAUDE, "B_principal": fc.MODELO_PRINCIPAL,
              "C_principal_renda": fc.MODELO_PRINCIPAL + ["renda", "escolaridade"],
              "D_principal_landmark1a": fc.MODELO_PRINCIPAL}
    alvo = ["atividade_nivel", "minutos_ativos_dia", "peso_kg", "porte", "peso_mudanca_12m=perdeu", "apetite=fraco",
            "alimentacao=1x", "castracao=<6m", "castracao=6-12m", "dieta_principal=caseira", "sobrepeso_ever=sim",
            "saude_geral", "n_sistemas_doenca", "renda=<40k", "renda=120k+"]
    hr = []
    for esp, cols in especs.items():
        d = lm(tr) if esp.startswith("D") else tr
        X = desenho(d, cols).assign(tempo=d.tempo.to_numpy(), evento=d.evento.to_numpy())
        c = CoxPHFitter(penalizer=1e-4).fit(X, "tempo", "evento")
        ci = c.confidence_intervals_
        for v in alvo:
            if v in c.params_.index:
                hr.append(dict(especificacao=esp, variavel=v, hr=np.exp(c.params_[v]),
                               lo=np.exp(ci.loc[v].iloc[0]), hi=np.exp(ci.loc[v].iloc[1])))
    hr = pd.DataFrame(hr)
    hr.to_csv(OUT / "hr_especificacoes.csv", index=False)
    (OUT / "censura_modelo.json").write_text(json.dumps(cens, indent=1, default=float))
    log("HR:\n" + hr.pivot(index="variavel", columns="especificacao", values="hr").round(3).to_string()
        + "\ncensura: " + json.dumps(cens, default=float))


def secao_descr(df):
    """Agregados descritivos usados na discussão (eutanásia entre os óbitos com questionário de fim de vida)."""
    pop = pd.read_csv(mc.PROC / "populacao.csv")
    ob = pop[(pop.incluido == 1) & (pop.evento == 1)]
    eu = ob.eol_euthan.dropna().astype(bool)
    out = dict(obitos=len(ob), obitos_com_eols=int(len(eu)), eutanasia=int(eu.sum()), prop_eutanasia=float(eu.mean()),
               prop_obitos_data_estimada=float(df.loc[df.evento == 1, "obito_data_estimada"].mean()),
               prop_renda_120k_mais=float((df.renda == "120k+").mean()),
               prop_graduacao_ou_mais=float(df.escolaridade.isin(["graduacao", "pos"]).mean()))
    (OUT / "descritivo_extra.json").write_text(json.dumps(out, indent=1))
    log("descr: " + json.dumps(out))


# ------------------------------------------------------------------------------------------------
# 6. Vazamento
# ------------------------------------------------------------------------------------------------
def secao_vazamento(df, jobs):
    tr, te = df[df.particao == "treino"].reset_index(drop=True), df[df.particao == "teste"].reset_index(drop=True)

    def perm(s):
        rng = np.random.default_rng(s)
        d = tr.copy()
        o = rng.permutation(len(d))
        d["tempo"], d["evento"] = d.tempo.to_numpy()[o], d.evento.to_numpy()[o]
        esc, _ = Ajuste("xgb_cox", d).prever(te)
        return dict(semente=s, c_teste=c_harrell(te.tempo.to_numpy(), te.evento.to_numpy(), esc),
                    c_intrafaixa=c_intrafaixa(te.tempo.to_numpy(), te.evento.to_numpy(), esc, te.idade.to_numpy()))

    p = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(perm)(s) for s in range(1, 11)))
    p.to_csv(OUT / "vazamento_permutacao.csv", index=False)
    uni = []
    t, e = te.tempo.to_numpy(), te.evento.to_numpy()
    for c in fc.NUMERICAS + fc.ORDINAIS + [b for b in fc.BINARIAS]:
        x = te[c].fillna(te[c].median()).to_numpy().astype(float)
        cc = c_harrell(t, e, x)
        uni.append(dict(atributo=c, c=cc, c_abs=max(cc, 1 - cc)))
    uni = pd.DataFrame(uni).sort_values("c_abs", ascending=False)
    uni.to_csv(OUT / "vazamento_univariado.csv", index=False)
    sobrep = len(set(tr.owner_id) & set(te.owner_id))
    (OUT / "vazamento_extra.json").write_text(json.dumps(dict(tutores_em_treino_e_teste=sobrep)))
    log("vazamento:\n" + p.round(3).to_string() + "\n" + uni.head(6).round(3).to_string() + f"\ntutores comuns: {sobrep}")


# ------------------------------------------------------------------------------------------------
# 7. SHAP
# ------------------------------------------------------------------------------------------------
GRUPO = {**{c: "Estáveis" for c in fc.ESTAVEIS}, **{c: "Estilo de vida" for c in fc.ESTILO_VIDA},
         **{c: "Saúde na entrada" for c in fc.SAUDE}}


def shap_por_feature(aj, d):
    import shap
    X = aj.pre.transform(d[aj.cols])
    sv = shap.TreeExplainer(aj.mod).shap_values(_dm(X))
    mapa = mc.mapa_coluna_original(aj.pre, aj.cols)
    return pd.DataFrame(sv, columns=X.columns).T.groupby(mapa.to_numpy()).sum().T, X, mapa


def secao_shap(df):
    import shap
    from scipy.stats import spearmanr
    tr, te = df[df.particao == "treino"].reset_index(drop=True), df[df.particao == "teste"].reset_index(drop=True)
    aj = Ajuste("xgb_cox", tr)
    S, X, mapa = shap_por_feature(aj, te)
    imp = S.abs().mean().sort_values(ascending=False)
    grupos = pd.Series(GRUPO)
    imp_grupo = pd.DataFrame({g: S[[c for c in S.columns if GRUPO[c] == g]].sum(axis=1).abs().mean()
                              for g in grupos.unique()}, index=["xgb_cox"]).T
    # sem saúde: para onde vai o crédito?
    aj_s = Ajuste("xgb_sem_saude", tr)
    S_s, _, _ = shap_por_feature(aj_s, te)
    imp_s = S_s.abs().mean()
    imp_grupo["xgb_sem_saude"] = pd.Series({g: S_s[[c for c in S_s.columns if GRUPO[c] == g]].sum(axis=1).abs().mean()
                                            for g in ["Estáveis", "Estilo de vida"]})
    # estabilidade entre os 5 modelos de fold
    tr_f = tr.reset_index(drop=True)
    imps = []
    for k, i_tr, _ in mc.iter_folds(tr_f):
        a = Ajuste("xgb_cox", tr_f.iloc[i_tr])
        Sk, _, _ = shap_por_feature(a, te)
        imps.append(Sk.abs().mean().reindex(imp.index))
    imps = pd.concat(imps, axis=1)
    rho = [spearmanr(imps.iloc[:, i], imps.iloc[:, j])[0] for i in range(5) for j in range(i + 1, 5)]
    top10 = [len(set(imps.iloc[:, i].nlargest(10).index) & set(imp.index[:10])) for i in range(5)]
    # concordância com o Cox linear: |beta_j (x_j - média do treino)| somado por atributo
    cl = Ajuste("cox_linear", tr)
    Xc = cl.pre.transform(te[cl.cols])
    Xtr = cl.pre.transform(tr[cl.cols])
    contrib = (Xc - Xtr.mean()) * cl.mod.params_.reindex(Xc.columns).to_numpy()
    mapc = mc.mapa_coluna_original(cl.pre, cl.cols)
    imp_cox = contrib.T.groupby(mapc.to_numpy()).sum().T.abs().mean().reindex(imp.index)
    rho_cox = spearmanr(imp, imp_cox)[0]
    # interações a priori (amostra de 2.000 cães do teste)
    rng = np.random.default_rng(mc.SEED)
    amostra = rng.choice(len(te), 2000, replace=False)
    Xs = X.iloc[amostra]
    iv = shap.TreeExplainer(aj.mod).shap_interaction_values(_dm(Xs))
    cols_o = mapa.to_numpy()
    feats = list(dict.fromkeys(cols_o))
    M = np.zeros((len(Xs), len(feats), len(feats)))
    pos = {f: np.flatnonzero(cols_o == f) for f in feats}
    for a_, fa in enumerate(feats):
        for b_, fb in enumerate(feats):
            M[:, a_, b_] = iv[:, pos[fa]][:, :, pos[fb]].sum(axis=(1, 2))
    fi = {f: i for i, f in enumerate(feats)}
    pares = [("idade", "porte"), ("idade", "castracao"), ("porte", "castracao"), ("idade", "atividade_nivel"),
             ("idade", "saude_geral")]
    inter = []
    for a_, b_ in pares:
        v = 2 * M[:, fi[a_], fi[b_]]
        inter.append(dict(par=f"{a_} x {b_}", media_abs=float(np.abs(v).mean())))
    tot_main = float(np.abs(np.einsum("nii->ni", M)).sum(axis=1).mean())
    tot_int = float((np.abs(M).sum(axis=(1, 2)) - np.abs(np.einsum("nii->ni", M)).sum(axis=1)).mean())
    # top interações entre todos os pares (descritivo)
    off = np.abs(M).mean(axis=0) * 2
    np.fill_diagonal(off, 0)
    iu = np.dstack(np.unravel_index(np.argsort(-np.triu(off).ravel())[:5], off.shape))[0]
    top_int = [dict(par=f"{feats[i]} x {feats[j]}", media_abs=float(off[i, j])) for i, j in iu]
    resumo = dict(rho_folds_media=float(np.mean(rho)), rho_folds_min=float(np.min(rho)), top10_overlap=top10,
                  rho_xgb_cox=float(rho_cox), soma_abs_efeitos_principais=tot_main, soma_abs_interacoes=tot_int,
                  interacoes_a_priori=inter, top_interacoes=top_int)
    out = pd.DataFrame(dict(grupo=[GRUPO[c] for c in imp.index], shap_xgb=imp, shap_xgb_sem_saude=imp_s.reindex(imp.index),
                            contrib_cox=imp_cox, dp_folds=imps.std(axis=1), min_folds=imps.min(axis=1),
                            max_folds=imps.max(axis=1)))
    out.to_csv(OUT / "shap_importancia.csv")
    imp_grupo.to_csv(OUT / "shap_grupos.csv")
    (OUT / "shap_resumo.json").write_text(json.dumps(resumo, indent=1))
    log("shap:\n" + out.head(15).round(4).to_string() + "\n" + imp_grupo.round(4).to_string() + "\n" + json.dumps(resumo))
    # figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
    cor_g = {"Estáveis": "#6a51a3", "Estilo de vida": "#41ab5d", "Saúde na entrada": "#d94801"}
    top = out.head(15)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax[0].barh(range(len(top)), top.shap_xgb, xerr=[top.shap_xgb - top.min_folds, top.max_folds - top.shap_xgb],
               color=[cor_g[g] for g in top.grupo], error_kw=dict(lw=0.6, capsize=1.5))
    ax[0].set_yticks(range(len(top)), [NOMES.get(c, c) for c in top.index])
    ax[0].set_xlabel("média |SHAP| (log-risco)")
    ax[0].set_title("(a) Importância global (teste)")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=c, label=g) for g, c in cor_g.items()], frameon=False, fontsize=6.5,
                 loc="lower right")
    idade = te.idade.to_numpy()
    porte = te.porte.to_numpy()
    nomes_p = ["toy/pequeno", "médio", "padrão", "grande", "gigante"]
    cm = plt.get_cmap("viridis")
    for p_ in range(5):
        m = porte == p_
        ax[1].scatter(idade[m], (S["idade"] + S["porte"] + S["peso_kg"]).to_numpy()[m], s=1.2, alpha=0.35,
                      color=cm(p_ / 4), label=nomes_p[p_], rasterized=True)
    ax[1].set(xlabel="Idade na entrada (anos)", ylabel="SHAP idade + porte + peso")
    ax[1].set_title("(b) Idade × tamanho")
    ax[1].legend(frameon=False, fontsize=6, markerscale=5, loc="upper left")
    virgula(*ax)
    ax[0].set_yticks(range(len(top)), [NOMES.get(c, c) for c in top.index])
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "shap.pdf", dpi=300)
    plt.close(fig)


NOMES = {"idade": "idade", "saude_geral": "saúde geral (tutor)", "porte": "porte", "peso_kg": "peso",
         "n_sistemas_doenca": "nº de sistemas c/ doença", "castracao": "castração", "minutos_ativos_dia": "min. ativos/dia",
         "apetite": "apetite", "peso_mudanca_12m": "mudança de peso 12m", "doenca_cancer": "câncer (diag.)",
         "atividade_nivel": "nível de atividade", "raca_pura": "raça pura", "regiao": "região",
         "doenca_cardiaca": "cardíaca (diag.)", "doenca_neurologica": "neurológica (diag.)", "sexo": "sexo",
         "subpeso_ever": "já abaixo do peso", "sobrepeso_ever": "já acima do peso", "dieta_principal": "dieta principal",
         "alimentacao": "refeições/dia", "atividade_intensidade": "intensidade da atividade",
         "doenca_renal_urinaria": "renal/urinária (diag.)", "doenca_endocrina": "endócrina (diag.)",
         "doenca_ortopedica": "ortopédica (diag.)", "passeio_guia_freq": "passeio com guia",
         "brincadeira_freq": "brincadeira", "passeio_solto_freq": "passeio solto",
         "doenca_respiratoria": "respiratória (diag.)", "doenca_gastrointestinal": "gastrointestinal (diag.)",
         "doenca_figado_pancreas": "fígado/pâncreas (diag.)", "doenca_hematologica": "hematológica (diag.)",
         "doenca_imunomediada": "imunomediada (diag.)", "dieta_consistencia": "consistência da dieta"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--secoes", default="cv,teste,sementes,clf2a,vieses,vazamento,shap,descr")
    a.add_argument("--n-boot", type=int, default=1000)
    a.add_argument("--jobs", type=int, default=6)
    a = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = carregar()
    log(f"dados: {len(df)} cães, {int(df.evento.sum())} óbitos")
    for s in a.secoes.split(","):
        log(f"== {s}")
        {"cv": lambda: secao_cv(df, a.jobs), "teste": lambda: secao_teste(df, a.n_boot, a.jobs),
         "sementes": lambda: secao_sementes(df, a.jobs), "clf2a": lambda: secao_clf2a(df),
         "vieses": lambda: secao_vieses(df, a.jobs), "vazamento": lambda: secao_vazamento(df, a.jobs),
         "shap": lambda: secao_shap(df), "descr": lambda: secao_descr(df)}[s]()
    log("fim")


if __name__ == "__main__":
    main()
