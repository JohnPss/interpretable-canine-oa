"""Fase 4d — gradient boosting de sobrevivência: XGBoost com objetivo de Cox (survival:cox).

Modelos: xgb_cox (MODELO_PRINCIPAL) e xgb_cox_sem_saude (ablação). Mesmo split, folds e métricas.

Empates: no desenho de OA (tempo anual, 4 valores) o objetivo nativo dependia da ordem das linhas dentro
dos empates. Aqui o tempo é contínuo (dias), os empates são raros e pequenos, e o nativo é usado. O script
CONFERE isso: treina o modelo final com duas ordens de linhas e reporta a correlação das margens.

Hiperparâmetros: busca aleatória (N_CONFIGS sorteadas + 1 de referência), 5 folds do treino. Em cada fold,
o treino do fold é subdividido 80/20 por tutor e o early stopping usa só esse pedaço interno; o fold de
validação só pontua. Critério: log-verossimilhança parcial negativa (Breslow, modelagem.nll_parcial) média de
validação. n_estimators final = mediana das melhores iterações; modelo final no treino inteiro.
Risco absoluto: risco basal de Breslow estimado com as margens do treino (modelagem.risco_de_cox).

Saídas: data/processed/pred_xgb_cox*.csv, xgb_cox*.ubj (fora do git)
        resultados/xgb_cox_busca.csv, xgb_cox_hiperparametros.json, xgb_cox_metricas.csv, xgb_cox_delta.csv,
        xgb_cox_ordem_linhas.csv
Uso: python 07_xgb_cox.py [--n-boot 1000] [--configs 40] [--jobs 6] [--jobs-boot 4] [--reusar-busca]

Robustez (a rodada de 30/09 morreu depois da busca, sem gravar métricas): as saídas são regravadas ao fim de
CADA modelo; a busca de cada modelo fica em cache (data/processed/xgb_busca_<modelo>.csv) e --reusar-busca a
reaproveita se o número de configurações bater; o bootstrap usa --jobs-boot processos (n_jobs=-1 com 12
núcleos e pouca RAM livre é o suspeito da queda).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit

import features_config as fc
import modelagem as mc

MODELOS = {"xgb_cox": fc.MODELO_PRINCIPAL, "xgb_cox_sem_saude": fc.SEM_SAUDE}
ESPACO = {"max_depth": ("inteiro", 1, 6), "eta": ("log", 0.01, 0.15), "min_child_weight": ("log", 0.5, 40.0),
          "subsample": ("uniforme", 0.5, 1.0), "colsample_bytree": ("uniforme", 0.4, 1.0),
          "reg_lambda": ("log", 0.1, 30.0)}
REFERENCIA = dict(max_depth=3, eta=0.05, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
PACIENCIA, MAX_ARVORES = 100, 4000


def sortear(n, seed=mc.SEED):
    rng = np.random.default_rng(seed)
    cfgs = [dict(REFERENCIA)]
    for _ in range(n):
        c = {}
        for k, (tipo, lo, hi) in ESPACO.items():
            c[k] = int(rng.integers(lo, hi + 1)) if tipo == "inteiro" else \
                float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if tipo == "log" else float(rng.uniform(lo, hi))
        cfgs.append(c)
    return cfgs


def rotulo(d):
    return np.where(d.evento.to_numpy() == 1, d.tempo.to_numpy(), -d.tempo.to_numpy())


def params(c):
    return dict(objective="survival:cox", eval_metric="cox-nloglik", tree_method="hist", nthread=1,
                seed=mc.SEED, **c)


def matriz(pre, d, cols):
    X = pre.transform(d[cols])
    return xgb.DMatrix(X, label=rotulo(d), feature_names=[c.replace("<", "lt").replace("[", "(").replace("]", ")")
                                                          for c in X.columns])


def avaliar_config(cid, c, tr, cols):
    res = []
    for k, i_tr, i_val in mc.iter_folds(tr):
        a, v = tr.iloc[i_tr], tr.iloc[i_val]
        gi, ge = next(GroupShuffleSplit(1, test_size=0.2, random_state=mc.SEED + k).split(a, groups=a.owner_id))
        a_in, a_es = a.iloc[gi], a.iloc[ge]
        pre = mc.construir_preprocessador(cols, padronizar=False).fit(a_in[cols])
        bst = xgb.train(params(c), matriz(pre, a_in, cols), MAX_ARVORES,
                        evals=[(matriz(pre, a_es, cols), "es")], early_stopping_rounds=PACIENCIA, verbose_eval=False)
        n = bst.best_iteration + 1
        margem = bst.predict(matriz(pre, v, cols), output_margin=True, iteration_range=(0, n))
        nll = mc.nll_parcial(margem, v.tempo, v.evento)
        res.append(dict(config_id=cid, fold=k, n_arvores=n, nll_val=nll))
    return res


def ajustar_final(d, cols, c, n, ordem_seed=None):
    if ordem_seed is not None:
        d = d.sample(frac=1, random_state=ordem_seed)
    pre = mc.construir_preprocessador(cols, padronizar=False).fit(d[cols])
    return pre, xgb.train(params(c), matriz(pre, d, cols), n)


def predicoes(df, cols, c, n):
    tr = df[df.particao == "treino"].reset_index(drop=True)
    te = df[df.particao == "teste"].reset_index(drop=True)
    partes = []

    def bloco(pre, bst, d_tr, d):
        esc_tr = bst.predict(matriz(pre, d_tr, cols), output_margin=True)
        esc = bst.predict(matriz(pre, d, cols), output_margin=True)
        out = pd.DataFrame(mc.risco_de_cox(esc, d_tr.tempo, d_tr.evento, esc_tr), columns=mc.COLS_RISCO)
        out.insert(0, "fold", d.fold.to_numpy()), out.insert(0, "particao", d.particao.to_numpy())
        out.insert(0, "dog_id", d.dog_id.to_numpy())
        out["escore"] = esc
        return out

    for k, i_tr, i_val in mc.iter_folds(tr):
        pre, bst = ajustar_final(tr.iloc[i_tr], cols, c, n)
        partes.append(bloco(pre, bst, tr.iloc[i_tr], tr.iloc[i_val]))
    pre, bst = ajustar_final(tr, cols, c, n)
    partes.append(bloco(pre, bst, tr, te))
    return pd.concat(partes, ignore_index=True), pre, bst


def buscar(nome, cfgs, tr, cols, jobs, reusar):
    cache = mc.PROC / f"xgb_busca_{nome}.csv"
    if reusar and cache.exists():
        b = pd.read_csv(cache)
        if b.config_id.nunique() == len(cfgs) and len(b) == len(cfgs) * mc.N_FOLDS:
            print(f"[{nome}] busca reaproveitada de {cache.name}", flush=True)
            return b
        print(f"[{nome}] cache {cache.name} não bate com {len(cfgs)} configs; refazendo", flush=True)
    r = Parallel(n_jobs=jobs)(delayed(avaliar_config)(i, c, tr, cols) for i, c in enumerate(cfgs))
    b = pd.DataFrame([x for lst in r for x in lst])
    b.to_csv(cache, index=False)
    return b


def gravar(buscas, hp, cfgs, ordem, mets, deltas, n_boot):
    pd.concat(buscas).to_csv(mc.RES / "xgb_cox_busca.csv", index=False)
    (mc.RES / "xgb_cox_hiperparametros.json").write_text(json.dumps(dict(
        espaco=ESPACO, referencia=REFERENCIA, n_configs=len(cfgs), paciencia=PACIENCIA, max_arvores=MAX_ARVORES,
        criterio="nll parcial de Breslow média de validação (5 folds), modelagem.nll_parcial", n_boot=n_boot,
        modelos=hp), indent=2, ensure_ascii=False))
    pd.DataFrame(ordem).to_csv(mc.RES / "xgb_cox_ordem_linhas.csv", index=False)
    met = pd.concat(mets, ignore_index=True)
    met["particao"] = met.particao.replace({"treino": "treino_oof"})
    met.to_csv(mc.RES / "xgb_cox_metricas.csv", index=False)
    delta = pd.concat(deltas, ignore_index=True)
    delta.to_csv(mc.RES / "xgb_cox_delta.csv", index=False)
    return met, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=mc.N_BOOT)
    ap.add_argument("--configs", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--jobs-boot", type=int, default=4)
    ap.add_argument("--reusar-busca", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    marca = lambda msg: print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)
    df = mc.carregar_com_particao()
    tr = df[df.particao == "treino"].reset_index(drop=True)
    te = df[df.particao == "teste"].reset_index(drop=True)
    base = mc.carregar_predicoes("cox_idade")
    cfgs = sortear(a.configs)
    buscas, hp, mets, deltas, ordem = [], {}, [], [], []
    for nome, cols in MODELOS.items():
        marca(f"{nome}: busca ({len(cfgs)} configs x {mc.N_FOLDS} folds)")
        b = buscar(nome, cfgs, tr, cols, a.jobs, a.reusar_busca)
        res = b.groupby("config_id").agg(nll_medio=("nll_val", "mean"), nll_dp=("nll_val", "std"),
                                         n_arvores=("n_arvores", "median")).reset_index()
        res = res.join(pd.DataFrame(cfgs), on="config_id")
        melhor = res.loc[res.nll_medio.idxmin()]
        c = {k: (int(melhor[k]) if k == "max_depth" else float(melhor[k])) for k in ESPACO}
        n = int(melhor.n_arvores)
        hp[nome] = dict(config_id=int(melhor.config_id), n_estimators=n, params=c, nll_cv=float(melhor.nll_medio))
        print(f"[{nome}] melhor config {int(melhor.config_id)}: {c} | n_arvores={n} | nll_cv={melhor.nll_medio:.5f}",
              flush=True)
        buscas.append(res.assign(modelo=nome))

        marca(f"{nome}: predições fora-do-fold e modelo final")
        pred, pre, bst = predicoes(df, cols, c, n)
        mc.salvar_predicoes(nome, pred)
        bst.save_model(mc.PROC / f"{nome}.ubj")
        # Conferência de dependência da ordem das linhas (empates): mesmos hiperparâmetros, mas SEM
        # subamostragem (subsample = colsample = 1), para que a única diferença entre os dois ajustes seja a
        # ordem das linhas. Com subamostragem, a ordem muda a amostra sorteada e a comparação não isola empates.
        marca(f"{nome}: checagem da ordem das linhas")
        c_det = dict(c, subsample=1.0, colsample_bytree=1.0)
        pre1, bst1 = ajustar_final(tr, cols, c_det, n)
        pre2, bst2 = ajustar_final(tr, cols, c_det, n, ordem_seed=7)
        m1 = bst1.predict(matriz(pre1, te, cols), output_margin=True)
        m2 = bst2.predict(matriz(pre2, te, cols), output_margin=True)
        ordem.append(dict(modelo=nome, corr_margens=float(np.corrcoef(m1, m2)[0, 1]),
                          dif_abs_max=float(np.abs(m1 - m2).max())))
        print(pd.DataFrame(ordem[-1:]).to_string(index=False), flush=True)
        jb = a.jobs_boot
        for part in ["teste", "treino"]:
            marca(f"{nome}: métricas ({part}, {a.n_boot} réplicas)")
            mets.append(mc.avaliar(pred, nome, particao=part, n_boot=a.n_boot, dados=df, n_jobs=jb))
        marca(f"{nome}: diferença pareada vs cox_idade")
        deltas.append(mc.comparar(pred, base, nome, "cox_idade", n_boot=a.n_boot, dados=df, n_jobs=jb))
        if nome == "xgb_cox":
            marca(f"{nome}: diferença pareada vs cox_linear")
            lin = mc.carregar_predicoes("cox_linear")
            deltas.append(mc.comparar(pred, lin, nome, "cox_linear", n_boot=a.n_boot, dados=df, n_jobs=jb))
        met, delta = gravar(buscas, hp, cfgs, ordem, mets, deltas, a.n_boot)
        marca(f"{nome}: saídas gravadas")
    print(pd.DataFrame(ordem).to_string(index=False))
    print(mc.resumir(met[met.particao == "teste"]).to_string())
    print(delta[delta.metrica == "c_index"][["modelo", "referencia", "estrato", "diferenca", "ic_lo", "ic_hi"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
