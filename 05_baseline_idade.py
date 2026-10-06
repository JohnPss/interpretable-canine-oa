"""Fase 4b — referências: Cox só com idade e Cox com idade + porte.

É contra estes modelos que o ganho dos demais é medido: sem eles não dá para dizer que o modelo aprendeu
algo além de "cão velho morre mais" (e "cão grande envelhece mais rápido").

Saídas: data/processed/pred_cox_idade.csv, pred_cox_idade_porte.csv (fora do git)
        resultados/baseline_metricas.csv, resultados/baseline_delta.csv, resultados/baseline_coef.csv
Uso: python 05_baseline_idade.py [--n-boot 1000]
"""
import argparse

import pandas as pd

import modelagem as mc

MODELOS = {"cox_idade": ["idade"], "cox_idade_porte": ["idade", "porte"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=mc.N_BOOT)
    n_boot = ap.parse_args().n_boot
    df = mc.carregar_com_particao()
    mets, coefs, preds = [], [], {}
    for nome, cols in MODELOS.items():
        pred, cph, _ = mc.predicoes_cox(df, cols)
        mc.salvar_predicoes(nome, pred)
        preds[nome] = pred
        s = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]].reset_index()
        coefs.append(s.assign(modelo=nome, nota="colunas padronizadas (HR por desvio-padrão)"))
        for part in ["teste", "treino"]:
            mets.append(mc.avaliar(pred, nome, particao=part, n_boot=n_boot, dados=df))
    met = pd.concat(mets, ignore_index=True)
    met["particao"] = met.particao.replace({"treino": "treino_oof"})
    met.to_csv(mc.RES / "baseline_metricas.csv", index=False)
    pd.concat(coefs).to_csv(mc.RES / "baseline_coef.csv", index=False)
    delta = mc.comparar(preds["cox_idade_porte"], preds["cox_idade"], "cox_idade_porte", "cox_idade",
                        n_boot=n_boot, dados=df)
    delta.to_csv(mc.RES / "baseline_delta.csv", index=False)
    print(mc.resumir(met[met.particao == "teste"]).to_string())
    print("\nGanho pareado idade+porte − idade (teste):")
    print(delta[delta.metrica == "c_index"][["estrato", "diferenca", "ic_lo", "ic_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
