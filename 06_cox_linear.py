"""Fase 4c — Cox linear (lifelines, empates de Efron) com penalização ridge escolhida por CV.

Modelos: cox_linear (MODELO_PRINCIPAL, 33 features) e cox_linear_sem_saude (ablação sem o bloco de saúde).
Penalizador: grade abaixo, critério = log-verossimilhança parcial média nas partições de validação (5 folds
do treino). Predições: fora-do-fold no treino, modelo do treino inteiro no teste.

Saídas: data/processed/pred_cox_linear*.csv (fora do git)
        resultados/cox_linear_cv.csv, cox_linear_hr.csv, cox_linear_metricas.csv, cox_linear_delta.csv
Uso: python 06_cox_linear.py [--n-boot 1000]
"""
import argparse

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import features_config as fc
import modelagem as mc

GRADE = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
MODELOS = {"cox_linear": fc.MODELO_PRINCIPAL, "cox_linear_sem_saude": fc.SEM_SAUDE}


def cv_penalizador(tr, cols):
    linhas = []
    for k, i_tr, i_val in mc.iter_folds(tr):
        a, v = tr.iloc[i_tr], tr.iloc[i_val]
        pre = mc.construir_preprocessador(cols, drop_primeira=True).fit(a[cols])
        Xa = pre.transform(a[cols]).assign(tempo=a.tempo.to_numpy(), evento=a.evento.to_numpy())
        Xv = pre.transform(v[cols]).assign(tempo=v.tempo.to_numpy(), evento=v.evento.to_numpy())
        for pen in GRADE:
            cph = CoxPHFitter(penalizer=pen).fit(Xa, "tempo", "evento")
            linhas.append(dict(fold=k, penalizador=pen, ll_val=cph.score(Xv, scoring_method="log_likelihood"),
                               c_val=cph.score(Xv, scoring_method="concordance_index")))
    cv = pd.DataFrame(linhas).groupby("penalizador").agg(ll_medio=("ll_val", "mean"), ll_dp=("ll_val", "std"),
                                                         c_medio=("c_val", "mean")).reset_index()
    return cv, float(cv.loc[cv.ll_medio.idxmax(), "penalizador"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=mc.N_BOOT)
    n_boot = ap.parse_args().n_boot
    df = mc.carregar_com_particao()
    tr = df[df.particao == "treino"].reset_index(drop=True)
    base = mc.carregar_predicoes("cox_idade")
    cvs, hrs, mets, deltas = [], [], [], []
    for nome, cols in MODELOS.items():
        cv, pen = cv_penalizador(tr, cols)
        print(f"[{nome}] penalizador escolhido: {pen}")
        cvs.append(cv.assign(modelo=nome, escolhido=cv.penalizador == pen))
        pred, cph, _ = mc.predicoes_cox(df, cols, penalizador=pen)
        mc.salvar_predicoes(nome, pred)
        s = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index()
        hrs.append(s.assign(modelo=nome, penalizador=pen,
                            nota="numéricas/ordinais padronizadas: HR por desvio-padrão; dummies vs 1ª categoria"))
        for part in ["teste", "treino"]:
            mets.append(mc.avaliar(pred, nome, particao=part, n_boot=n_boot, dados=df))
        deltas.append(mc.comparar(pred, base, nome, "cox_idade", n_boot=n_boot, dados=df))
    met = pd.concat(mets, ignore_index=True)
    met["particao"] = met.particao.replace({"treino": "treino_oof"})
    met.to_csv(mc.RES / "cox_linear_metricas.csv", index=False)
    pd.concat(cvs).to_csv(mc.RES / "cox_linear_cv.csv", index=False)
    pd.concat(hrs).to_csv(mc.RES / "cox_linear_hr.csv", index=False)
    delta = pd.concat(deltas, ignore_index=True)
    delta.to_csv(mc.RES / "cox_linear_delta.csv", index=False)
    print(mc.resumir(met[met.particao == "teste"]).to_string())
    print("\nGanho pareado sobre cox_idade (C-index, teste):")
    print(delta[delta.metrica == "c_index"][["modelo", "estrato", "diferenca", "ic_lo", "ic_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
