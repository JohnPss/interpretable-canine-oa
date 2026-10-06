"""Base compartilhada da modelagem de longevidade: dados, split, folds, pré-processamento, formato das
predições e métricas. TODOS os modelos importam daqui, para que os números sejam comparáveis.

    import modelagem as mc

Tempo contínuo em anos desde o HLES (01_populacao.py), evento = óbito até 2024-12-31.
Adaptado de modelagem_comum.py do desenho de OA (branch oa-00-base): o split/folds e o pré-processador
são os mesmos; as métricas trocaram a escala anual discreta (1..4) por tempo contínuo, onde os estimadores
padrão do scikit-survival funcionam sem ajuste.

Métricas (particao de teste, ou fora-do-fold no treino):
  - C-index de Harrell (sksurv.concordance_index_censored), escore = log-hazard (ou risco em 3 anos).
  - AUC dependente do tempo (Uno, IPCW; sksurv.cumulative_dynamic_auc) em 1, 2 e 3 anos; a distribuição
    da censura é estimada no conjunto de TREINO, como o sksurv exige.
  - Calibração em 2 anos: risco previsto vs. Kaplan-Meier por decil (quintil nas faixas); O/E.
  - Brier em 2 anos com IPCW (sksurv.brier_score).
  - Todas por faixa etária (FAIXAS). IC 95% por bootstrap de TUTORES (owner_id), seed fixa → réplicas
    iguais entre modelos → diferença pareada válida.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

import features_config as fc

ROOT = Path(__file__).parent
PROC, RES = ROOT / "data" / "processed", ROOT / "resultados"

SEED = 42
N_FOLDS = 5
N_BOOT = 1000
HORIZONTES = (1, 2, 3)      # anos; seguimento mediano ~2,1, máximo ~5
H_CALIB = 2
MIN_FREQ_RARA = 0.01

# Faixas etárias na entrada (lo <= idade < hi). 4 faixas: acima de 10 anos a mortalidade muda muito
# entre 10–13 e 13+ (no bruto, 80% dos cães com 13+ no HLES já morreram).
FAIXAS = [(0, 7, "< 7 anos"), (7, 10, "7–10 anos"), (10, 13, "10–13 anos"), (13, 99, "13+ anos")]
ESTRATOS = ["geral"] + [n for *_, n in FAIXAS]

COLS_RISCO = [f"risco_{h}a" for h in HORIZONTES]
COLS_PRED = ["dog_id", "particao", "fold"] + COLS_RISCO + ["escore"]


# =====================================================================================
# 1. Dados, split e folds
# =====================================================================================
def carregar_dados() -> pd.DataFrame:
    return pd.read_csv(PROC / "features.csv")


def faixa_codigo(idade) -> np.ndarray:
    cortes = [lo for lo, _, _ in FAIXAS[1:]]
    return np.searchsorted(cortes, np.asarray(idade, dtype=float), side="right")


def faixa_etaria(idade) -> np.ndarray:
    return np.array([n for *_, n in FAIXAS])[faixa_codigo(idade)]


def _estrato_split(df):
    # evento × faixa etária (8 estratos): mesma proporção de óbitos e mesmo perfil etário em cada parte.
    return (df["evento"].to_numpy() * len(FAIXAS) + faixa_codigo(df["idade"])).astype(int)


def _atribuir_folds(estrato, grupo, n_folds: int, seed: int) -> np.ndarray:
    """Fold de cada linha, estratificado por `estrato` e agrupado por `grupo` (tutor nunca dividido).
    Guloso: grupos em ordem aleatória (seed), os com estratos mais raros primeiro; cada grupo vai para o
    fold em que o desvio quadrático normalizado da contagem por estrato cresce menos. (Mesmo algoritmo
    do desenho de OA; mais apertado que o StratifiedGroupKFold do sklearn.)"""
    estrato, grupo = np.asarray(estrato), np.asarray(grupo)
    classes, y = np.unique(estrato, return_inverse=True)
    _, gi = np.unique(grupo, return_inverse=True)
    n_g = gi.max() + 1
    cont = np.zeros((n_g, len(classes)))
    np.add.at(cont, (gi, y), 1)
    alvo = cont.sum(axis=0) / n_folds
    rng = np.random.default_rng(seed)
    ordem = rng.permutation(n_g)
    raridade = (cont / cont.sum(axis=0)).max(axis=1)[ordem]
    ordem = ordem[np.argsort(-raridade, kind="stable")]
    atual, tam = np.zeros((n_folds, len(classes))), np.zeros(n_folds)
    fold_g = np.empty(n_g, dtype=int)
    for g in ordem:
        v = cont[g]
        custo = ((((atual + v - alvo) / alvo) ** 2) - (((atual - alvo) / alvo) ** 2)).sum(axis=1)
        f = np.lexsort((tam, np.round(custo, 12)))[0]
        fold_g[g] = f
        atual[f] += v
        tam[f] += v.sum()
    return fold_g[gi]


def criar_split_e_folds(df=None, seed: int = SEED):
    """Split 80/20 (parte 0 de 5 = teste) + 5 folds de CV só no treino (seed+1). Determinístico."""
    df = carregar_dados() if df is None else df
    y, g = _estrato_split(df), df["owner_id"].to_numpy()
    eh_teste = _atribuir_folds(y, g, N_FOLDS, seed) == 0
    split = pd.DataFrame({"dog_id": df["dog_id"].to_numpy(), "particao": np.where(eh_teste, "teste", "treino")})
    fold = _atribuir_folds(y[~eh_teste], g[~eh_teste], N_FOLDS, seed + 1)
    folds = pd.DataFrame({"dog_id": df["dog_id"].to_numpy()[~eh_teste], "fold": fold})
    return split, folds


def salvar_split_e_folds(forcar: bool = False):
    ps, pf = PROC / "split.csv", PROC / "folds.csv"
    if (ps.exists() or pf.exists()) and not forcar:
        raise FileExistsError("split.csv/folds.csv já existem; forcar=True só de propósito (invalida os modelos).")
    split, folds = criar_split_e_folds()
    split.to_csv(ps, index=False)
    folds.to_csv(pf, index=False)
    return split, folds


def carregar_com_particao() -> pd.DataFrame:
    """features.csv + `particao` e `fold` (-1 no teste). Erro se o split não bate com o features.csv."""
    if not (PROC / "split.csv").exists():
        salvar_split_e_folds()
    df, split, folds = carregar_dados(), pd.read_csv(PROC / "split.csv"), pd.read_csv(PROC / "folds.csv")
    if set(split.dog_id) != set(df.dog_id) or len(split) != len(df):
        raise ValueError("split.csv não bate com features.csv: regere de propósito e refaça todos os modelos.")
    out = df.merge(split, on="dog_id").merge(folds, on="dog_id", how="left")
    out["fold"] = out["fold"].fillna(-1).astype(int)
    return out


def iter_folds(tr: pd.DataFrame):
    """(k, idx_treino, idx_val) em POSIÇÕES de `tr` (só treino)."""
    f = tr["fold"].to_numpy()
    for k in range(N_FOLDS):
        yield k, np.flatnonzero(f != k), np.flatnonzero(f == k)


def verificar_split(df=None) -> pd.DataFrame:
    df = carregar_com_particao() if df is None else df
    tr, te = df[df.particao == "treino"], df[df.particao == "teste"]
    assert set(tr.owner_id).isdisjoint(te.owner_id), "owner_id nos dois lados do split"
    for k in range(N_FOLDS):
        assert set(tr.owner_id[tr.fold == k]).isdisjoint(tr.owner_id[tr.fold != k]), f"owner_id em 2 folds ({k})"
    linhas = []
    for nome, p in [("treino", tr), ("teste", te)] + [(f"fold{k}", tr[tr.fold == k]) for k in range(N_FOLDS)]:
        linha = dict(particao=nome, n_caes=len(p), n_tutores=p.owner_id.nunique(), obitos=int(p.evento.sum()),
                     prop_obitos=round(p.evento.mean(), 4), cao_anos=round(p.tempo.sum()))
        fx = faixa_codigo(p.idade)
        for i, (*_, rot) in enumerate(FAIXAS):
            linha[f"n {rot}"] = int((fx == i).sum())
        linhas.append(linha)
    return pd.DataFrame(linhas)


# =====================================================================================
# 2. Pré-processamento (ajustar SÓ no treino)
# =====================================================================================
def construir_preprocessador(colunas, padronizar=True, min_freq=MIN_FREQ_RARA, drop_primeira=False):
    """NUMERICAS+ORDINAIS: mediana (+ escala); BINARIAS: passthrough; CATEGORICAS: 'faltante' + one-hot,
    categorias < min_freq do treino → '<feat>_infrequent_sklearn'. drop_primeira=True para Cox/GLM."""
    colunas = list(colunas)
    desconhecidas = [c for c in colunas if c not in fc.NUMERICAS + fc.ORDINAIS + fc.BINARIAS + fc.CATEGORICAS]
    if desconhecidas:
        raise ValueError(f"colunas fora do features_config: {desconhecidas}")
    num = [c for c in colunas if c in fc.NUMERICAS + fc.ORDINAIS]
    binarias = [c for c in colunas if c in fc.BINARIAS]
    cat = [c for c in colunas if c in fc.CATEGORICAS]
    passos = [("imputar", SimpleImputer(strategy="median"))] + ([("escalar", StandardScaler())] if padronizar else [])
    blocos = []
    if num:
        blocos.append(("num", Pipeline(passos), num))
    if binarias:
        blocos.append(("bin", "passthrough", binarias))
    if cat:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq, sparse_output=False,
                            drop="first" if drop_primeira else None)
        blocos.append(("cat", Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="faltante")),
                                        ("onehot", ohe)]), cat))
    ct = ColumnTransformer(blocos, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def mapa_coluna_original(pre: ColumnTransformer, colunas) -> pd.Series:
    """coluna transformada → feature original (para somar SHAP/coeficientes por feature)."""
    nomes = list(pre.get_feature_names_out())
    orig = sorted(colunas, key=len, reverse=True)
    return pd.Series([next(o for o in orig if n == o or n.startswith(o + "_")) for n in nomes], index=nomes)


# =====================================================================================
# 3. Predições
# =====================================================================================
def salvar_predicoes(modelo: str, pred: pd.DataFrame):
    """Valida e grava data/processed/pred_<modelo>.csv (fora do git): todos os cães, riscos em [0,1] e
    não decrescentes no horizonte; treino = fora-do-fold, teste = modelo do treino inteiro."""
    df = carregar_com_particao()[["dog_id", "particao", "fold"]]
    faltam = [c for c in COLS_PRED if c not in pred.columns]
    assert not faltam, faltam
    assert set(pred.dog_id) == set(df.dog_id) and pred.dog_id.is_unique, "predição não cobre todos os cães"
    chk = pred[["dog_id", "particao", "fold"]].merge(df, on="dog_id", suffixes=("", "_ok"))
    assert (chk.particao == chk.particao_ok).all() and (chk.fold == chk.fold_ok).all(), "partição/fold divergentes"
    r = pred[COLS_RISCO].to_numpy()
    assert np.isfinite(r).all() and (r >= 0).all() and (r <= 1).all(), "risco fora de [0,1]"
    assert (np.diff(r, axis=1) >= -1e-9).all(), "risco decrescente no horizonte"
    pred.to_csv(PROC / f"pred_{modelo}.csv", index=False)


def carregar_predicoes(modelo: str) -> pd.DataFrame:
    return pd.read_csv(PROC / f"pred_{modelo}.csv")


def nll_parcial(escore, tempo, evento) -> float:
    """−log-verossimilhança parcial de Cox (empates de Breslow) dividida pelo nº de eventos.
    Conjunto de risco de i = {j : t_j >= t_i}. Usada como critério de validação dos modelos de árvores."""
    escore, tempo, evento = (np.asarray(x, float) for x in (escore, tempo, evento))
    ordem = np.argsort(-tempo, kind="stable")                 # tempo decrescente
    m, t, e = escore[ordem], tempo[ordem], evento[ordem]
    c = m.max()
    acum = np.log(np.cumsum(np.exp(m - c))) + c               # log sum exp dos com tempo >= t (sem empates)
    # com empates, o conjunto de risco inclui todos os empatados: usar o último índice de cada tempo
    fim_grupo = pd.Series(np.arange(len(t))).groupby(t).transform("max").to_numpy()
    log_risco = acum[fim_grupo]
    return float(-(m[e == 1] - log_risco[e == 1]).sum() / e.sum())


def risco_de_cox(escore, tempo_tr, evento_tr, escore_tr):
    """Risco acumulado 1 - S(h | x) = 1 - exp(-H0(h) e^escore), H0 de Breslow estimado no treino
    (com os escores do treino). Serve a qualquer modelo de riscos proporcionais (Cox, XGBoost Cox)."""
    tempo_tr, evento_tr, escore_tr = map(np.asarray, (tempo_tr, evento_tr, escore_tr))
    ordem = np.argsort(tempo_tr)
    t, e, ex = tempo_tr[ordem], evento_tr[ordem], np.exp(escore_tr[ordem] - escore_tr.max())
    soma_risco = np.cumsum(ex[::-1])[::-1]                 # sum exp(escore) dos em risco em t_i
    tu, idx = np.unique(t, return_index=True)
    d = np.add.reduceat(e, idx)
    h = d / soma_risco[idx]
    H = np.array([h[tu <= hh].sum() for hh in HORIZONTES]) * np.exp(-escore_tr.max())
    return -np.expm1(-np.outer(np.exp(np.asarray(escore, float)), H))


def predicoes_cox(df, colunas, penalizador=0.0):
    """Cox (lifelines, Efron) com pré-processador drop_primeira: predições fora-do-fold no treino e do modelo
    do treino inteiro no teste. Devolve (pred no formato COLS_PRED, modelo final, pré-processador final)."""
    from lifelines import CoxPHFitter
    tr = df[df.particao == "treino"].reset_index(drop=True)
    te = df[df.particao == "teste"].reset_index(drop=True)

    def ajustar(d):
        pre = construir_preprocessador(colunas, drop_primeira=True).fit(d[colunas])
        X = pre.transform(d[colunas]).assign(tempo=d.tempo.to_numpy(), evento=d.evento.to_numpy())
        cph = CoxPHFitter(penalizer=penalizador).fit(X, "tempo", "evento")
        return pre, cph, cph.predict_log_partial_hazard(X.drop(columns=["tempo", "evento"])).to_numpy()

    def prever(pre, cph, esc_tr, d_tr, d):
        esc = cph.predict_log_partial_hazard(pre.transform(d[colunas])).to_numpy()
        risco = risco_de_cox(esc, d_tr.tempo, d_tr.evento, esc_tr)
        out = pd.DataFrame(risco, columns=COLS_RISCO)
        out.insert(0, "fold", d.fold.to_numpy()), out.insert(0, "particao", d.particao.to_numpy())
        out.insert(0, "dog_id", d.dog_id.to_numpy())
        out["escore"] = esc
        return out

    partes = []
    for k, i_tr, i_val in iter_folds(tr):
        pre, cph, esc_tr = ajustar(tr.iloc[i_tr])
        partes.append(prever(pre, cph, esc_tr, tr.iloc[i_tr], tr.iloc[i_val]))
    pre, cph, esc_tr = ajustar(tr)
    partes.append(prever(pre, cph, esc_tr, tr, te))
    return pd.concat(partes, ignore_index=True), cph, pre


# =====================================================================================
# 4. Métricas
# =====================================================================================
def km_risco(tempo, evento, h) -> float:
    """1 - S_KM(h)."""
    tempo, evento = np.asarray(tempo), np.asarray(evento)
    s = 1.0
    for t in np.unique(tempo[(evento == 1) & (tempo <= h)]):
        n = np.sum(tempo >= t)
        s *= 1 - np.sum((tempo == t) & (evento == 1)) / n
    return 1 - s


def _metricas_conjunto(y_tr, tempo, evento, escore, risco):
    """Métricas de UM conjunto (sem IC). y_tr = Surv do treino (distribuição da censura p/ IPCW)."""
    out = {}
    ev = evento.astype(bool)
    if ev.sum() < 2:
        return out
    out[("c_index", None)] = concordance_index_censored(ev, tempo, escore)[0]
    y = Surv.from_arrays(ev, tempo)
    tmax_tr = y_tr["time"].max()
    for i, h in enumerate(HORIZONTES):
        casos = (tempo <= h) & ev
        ctrl = tempo > h
        if casos.sum() >= 2 and ctrl.sum() >= 2 and h < tempo.max() and h < tmax_tr:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[("auc_tempo", h)] = cumulative_dynamic_auc(y_tr, y, risco[:, i], [h])[0][0]
    i2 = HORIZONTES.index(H_CALIB)
    if H_CALIB < tempo.max():
        obs = km_risco(tempo, evento, H_CALIB)
        out[("cal_oe", H_CALIB)] = obs / risco[:, i2].mean()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[("brier", H_CALIB)] = brier_score(y_tr, y, 1 - risco[:, i2], [H_CALIB])[1][0]
    return out


def indices_bootstrap(grupo, n_boot=N_BOOT, seed=SEED):
    """Reamostragem de tutores: lista de arrays de posições. Mesma seed + mesmo conjunto → mesmas réplicas."""
    grupo = np.asarray(grupo)
    ug, inv = np.unique(grupo, return_inverse=True)
    membros = [np.flatnonzero(inv == g) for g in range(len(ug))] if len(ug) < len(grupo) else None
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        gs = rng.integers(0, len(ug), len(ug))
        reps.append(gs if membros is None else np.concatenate([membros[g] for g in gs]))
    return reps


def avaliar(pred: pd.DataFrame, modelo: str, particao="teste", n_boot=N_BOOT, seed=SEED, n_jobs=-1,
            dados=None) -> pd.DataFrame:
    """Métricas (geral + por faixa) com IC 95% de bootstrap por tutor. Formato longo:
    modelo, particao, estrato, metrica, horizonte, valor, ic_lo, ic_hi, n, n_obitos, n_boot_ok."""
    df = carregar_com_particao() if dados is None else dados
    tr = df[df.particao == "treino"]
    y_tr = Surv.from_arrays(tr.evento.astype(bool), tr.tempo)
    d = df[df.particao == particao].merge(pred[["dog_id"] + COLS_RISCO + ["escore"]], on="dog_id")
    linhas = []
    for estrato in ESTRATOS:
        s = d if estrato == "geral" else d[faixa_etaria(d.idade) == estrato]
        t, e = s.tempo.to_numpy(), s.evento.to_numpy()
        esc, r = s.escore.to_numpy(), s[COLS_RISCO].to_numpy()
        ponto = _metricas_conjunto(y_tr, t, e, esc, r)
        reps = indices_bootstrap(s.owner_id.to_numpy(), n_boot, seed)
        boot = Parallel(n_jobs=n_jobs)(delayed(_metricas_conjunto)(y_tr, t[i], e[i], esc[i], r[i]) for i in reps)
        for (met, h), v in ponto.items():
            b = np.array([x.get((met, h), np.nan) for x in boot], float)
            ok = np.isfinite(b)
            linhas.append(dict(modelo=modelo, particao=particao, estrato=estrato, metrica=met, horizonte=h,
                               valor=v, ic_lo=np.nanpercentile(b, 2.5) if ok.any() else np.nan,
                               ic_hi=np.nanpercentile(b, 97.5) if ok.any() else np.nan,
                               n=len(s), n_obitos=int(e.sum()), n_boot_ok=int(ok.sum())))
    return pd.DataFrame(linhas)


def comparar(pred_novo, pred_base, nome_novo, nome_base, particao="teste", n_boot=N_BOOT, seed=SEED, n_jobs=-1,
             dados=None) -> pd.DataFrame:
    """Diferença (novo − base) com IC de bootstrap PAREADO (mesmas réplicas de tutores)."""
    df = carregar_com_particao() if dados is None else dados
    tr = df[df.particao == "treino"]
    y_tr = Surv.from_arrays(tr.evento.astype(bool), tr.tempo)
    cols = ["dog_id"] + COLS_RISCO + ["escore"]
    d = df[df.particao == particao].merge(pred_novo[cols], on="dog_id").merge(pred_base[cols], on="dog_id",
                                                                                  suffixes=("", "_base"))
    cb = [c + "_base" for c in COLS_RISCO]
    linhas = []
    for estrato in ESTRATOS:
        s = d if estrato == "geral" else d[faixa_etaria(d.idade) == estrato]
        t, e = s.tempo.to_numpy(), s.evento.to_numpy()
        a = (s.escore.to_numpy(), s[COLS_RISCO].to_numpy())
        b = (s.escore_base.to_numpy(), s[cb].to_numpy())
        f = lambda i: (_metricas_conjunto(y_tr, t[i], e[i], a[0][i], a[1][i]),
                       _metricas_conjunto(y_tr, t[i], e[i], b[0][i], b[1][i]))
        tudo = np.arange(len(s))
        pa, pb = f(tudo)
        boot = Parallel(n_jobs=n_jobs)(delayed(f)(i) for i in indices_bootstrap(s.owner_id.to_numpy(), n_boot, seed))
        for k in pa:
            if k not in pb or k[0] in ("cal_oe",):
                continue
            dif = np.array([x.get(k, np.nan) - y.get(k, np.nan) for x, y in boot], float)
            linhas.append(dict(modelo=nome_novo, referencia=nome_base, particao=particao, estrato=estrato,
                               metrica=k[0], horizonte=k[1], valor_modelo=pa[k], valor_referencia=pb[k],
                               diferenca=pa[k] - pb[k], ic_lo=np.nanpercentile(dif, 2.5),
                               ic_hi=np.nanpercentile(dif, 97.5), n=len(s), n_obitos=int(e.sum())))
    return pd.DataFrame(linhas)


def resumir(met: pd.DataFrame) -> pd.DataFrame:
    """Tabela larga 'valor [IC]' para imprimir."""
    m = met.copy()
    m["txt"] = m.apply(lambda r: f"{r.valor:.3f} [{r.ic_lo:.3f}–{r.ic_hi:.3f}]", axis=1)
    m["col"] = m.metrica + m.horizonte.map(lambda h: "" if pd.isna(h) else f"@{h:g}a")
    return m.pivot_table(index=["modelo", "estrato"], columns="col", values="txt", aggfunc="first")


if __name__ == "__main__":
    # Gera (uma vez) e confere o split.
    if not (PROC / "split.csv").exists():
        salvar_split_e_folds()
    RES.mkdir(exist_ok=True)
    v = verificar_split()
    v.to_csv(RES / "split_verificacao.csv", index=False)
    print(v.to_string(index=False))
