"""Etapa III — gera as tabelas LaTeX do artigo a partir de resultados/etapa3/ (saídas do 08_etapa3.py) e copia
as figuras. Nenhum número é digitado à mão. Uso: python 08b_tabelas_tex.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
R = ROOT / "resultados" / "etapa3"
ART = ROOT.parent / "Artigo" / "artigo"
TAB, FIGA = ART / "tabelas", ART / "figuras"
CAB = "% Gerado por interpretable-canine-oa/08b_tabelas_tex.py a partir de resultados/etapa3/ — não editar à mão.\n"


def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    s = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s.replace("-", "$-$")


def n0(x):
    return f"{int(round(x)):,}".replace(",", ".")


ROT = {"cox_idade": "Só idade", "cox_idade_porte": "Idade + porte",
       "cox_idade_porte_nsis": "Idade + porte + nº sistemas", "cox_estaveis": "Cox, só estáveis",
       "cox_sem_saude_sinais": "Cox, sem saúde e sinais", "cox_sem_saude": "Cox, sem saúde",
       "cox_linear": "\\textbf{Cox linear}", "xgb_estaveis": "XGB, só estáveis",
       "xgb_sem_saude_sinais": "XGB, sem saúde e sinais", "xgb_sem_saude": "XGB, sem saúde",
       "xgb_cox": "\\textbf{XGBoost Cox}"}
ORDEM = list(ROT)


def tab_resultados():
    m = pd.read_csv(R / "teste_metricas.csv")
    cv = pd.read_csv(R / "cv_resumo.csv", index_col=0)
    g = lambda mod, est, met: m.query("modelo==@mod and estrato==@est and metrica==@met").iloc[0]
    L = [CAB, r"""\begin{table*}[!t]
\caption{Desempenho no conjunto de teste (8.073 cães; 2.417 óbitos) e na validação cruzada (CV) repetida no treino.
C: C-index de Harrell, geral (IC 95\% por \textit{bootstrap} de tutores) e dentro de cada faixa etária (anos);
Uno: C de Uno truncado em 3 anos; AUC e Brier dependentes do tempo e razão observado/esperado (O/E) em 2 anos;
CV: média $\pm$ desvio-padrão em 15 partições (5 \textit{folds} $\times$ 3 repetições); $\bar{C}_{\text{faixa}}$: média dos C-index das quatro faixas}
\label{tab:resultados}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{|l|c|cccc|c|c|c|c|c|c|}
\hline
 & \multicolumn{9}{c|}{\textbf{Teste}} & \multicolumn{2}{c|}{\textbf{CV no treino (média $\pm$ dp)}} \\
\cline{2-12}
\textbf{Modelo} & \textbf{C geral [IC 95\%]} & \textbf{$<$7} & \textbf{7--10} & \textbf{10--13} & \textbf{$\geq$13} &
\textbf{Uno} & \textbf{AUC 2a} & \textbf{Brier 2a} & \textbf{O/E 2a} & \textbf{C geral} & \textbf{$\bar{C}_{\text{faixa}}$} \\
\hline"""]
    for mod in ORDEM:
        c = g(mod, "geral", "c_index")
        faixas = [f(g(mod, e, "c_index").valor) for e in ["< 7 anos", "7–10 anos", "10–13 anos", "13+ anos"]]
        L.append(f"{ROT[mod]} & {f(c.valor)} [{f(c.ic_lo)}--{f(c.ic_hi)}] & " + " & ".join(faixas) + " & "
                 + " & ".join(f(g(mod, 'geral', k).valor) for k in ["uno_c@3", "auc_tempo@2", "brier@2", "cal_oe@2"])
                 + f" & {f(cv.loc[mod, 'c_index_mean'])} $\\pm$ {f(cv.loc[mod, 'c_index_std'])}"
                 + f" & {f(cv.loc[mod, 'c_intrafaixa_mean'])} $\\pm$ {f(cv.loc[mod, 'c_intrafaixa_std'])} \\\\")
        if mod in ("cox_idade_porte_nsis", "cox_linear"):
            L.append("\\hline")
    L.append("\\hline\n\\end{tabular}\n\\end{table*}\n")
    (TAB / "tab_resultados.tex").write_text("\n".join(L))


def tab_clf():
    c = pd.read_csv(R / "clf2a.csv").set_index("modelo")
    ex = json.loads((R / "clf2a_extra.json").read_text())
    cols = ["maioria", "cox_idade", "cox_idade_porte", "cox_linear", "xgb_cox"]
    cab = ["Maioria", "Idade", "Idade + porte", "Cox lin.", "XGB"]
    linhas = [("Limiar (risco 2a)", "limiar", 3), ("VP", "VP", 0), ("FN", "FN", 0), ("FP", "FP", 0), ("VN", "VN", 0),
              ("Sensibilidade", "sensibilidade", 3), ("Especificidade", "especificidade", 3), ("VPP (precisão)", "vpp", 3),
              ("F1", "f1", 3), ("Acurácia", "acuracia", 3), ("Acurácia balanceada", "acuracia_balanceada", 3),
              ("AUC (ROC)", "auc_ipcw", 3), ("AP (precisão média)", "ap", 3)]
    L = [CAB, r"""\begin{table}[!t]
\caption{Leitura binária do risco em 2 anos no teste, com pesos pelo inverso da probabilidade de censura (IPCW).
Óbito em até 2 anos é a classe positiva (prevalência ponderada de """ + f(100 * ex["prevalencia_ipcw"], 1) + r"""\%).
VP, FN, FP e VN são contagens ponderadas (matriz de confusão); limiar de Youden escolhido nas predições fora do
\textit{fold} do treino; ``Maioria'' prevê que nenhum cão morre}
\label{tab:clf}
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|l|r|r|r|r|r|}
\hline
 & """ + " & ".join(f"\\textbf{{{x}}}" for x in cab) + " \\\\\n\\hline"]
    for rot, k, d in linhas:
        vals = []
        for m in cols:
            v = c.loc[m, k]
            if k == "limiar" and m == "maioria":
                vals.append("---")
            elif d == 0:
                vals.append(n0(v))
            else:
                vals.append(f(v, d))
        L.append(f"{rot} & " + " & ".join(vals) + " \\\\")
        if k in ("limiar", "VN", "acuracia_balanceada"):
            L.append("\\hline")
    L.append("\\hline\n\\end{tabular}\n\\end{table}\n")
    (TAB / "tab_clf.tex").write_text("\n".join(L))


def tab_sens():
    s = pd.read_csv(R / "sensibilidades.csv")
    perm = pd.read_csv(R / "vazamento_permutacao.csv")
    cen = [("principal", "Análise principal"), ("ipw_selecao", "IPW de seleção (ter seguimento)"),
           ("censura_simetrica_avaliacao", "Censura simétrica, avaliação"),
           ("censura_simetrica_retreino", "Censura simétrica, retreino"),
           ("sem_datas_estimadas", "Sem óbitos de data estimada"), ("landmark_1a_avaliacao", "Vivos em 1 ano (landmark)"),
           ("temporal_2022_24", "Temporal: treino $\\leq$2021, teste $\\geq$2022"),
           ("subconjunto_2022_24_modelo_completo", "Teste $\\geq$2022, modelo principal"),
           ("mais_renda_escolaridade", "+ renda e escolaridade"), ("mais_dieta_versao", "+ versão da dieta")]
    L = [CAB, r"""\begin{table}[!t]
\caption{Análises de sensibilidade a vieses de seleção, censura, medida e confusão (teste). C: C-index de Harrell;
$\bar{C}_{\text{faixa}}$: média dos C-index nas quatro faixas etárias; O/E em 2 anos do XGBoost. No \textit{landmark}, o
O/E não é comparável (o risco previsto é para os 2 anos a partir da entrada)}
\label{tab:sens}
\centering
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{|>{\raggedright\arraybackslash}p{2.7cm}|r|r|c|c|c|c|}
\hline
 & & & \multicolumn{2}{c|}{\textbf{C geral}} & & \\
\cline{4-5}
\textbf{Cenário} & \textbf{Cães} & \textbf{Óbitos} & \textbf{Idade} & \textbf{XGB} &
\textbf{\shortstack{$\bar{C}_{\text{faixa}}$\\XGB}} & \textbf{\shortstack{O/E\\XGB}} \\
\hline"""]
    for k, rot in cen:
        d = s[s.cenario == k].set_index("modelo")
        get = lambda m, col: d.loc[m, col] if m in d.index and col in d.columns else np.nan
        oe = "---" if k.startswith("landmark") else f(get("xgb_cox", "cal_oe@2"))
        L.append(f"{rot} & {n0(get('xgb_cox', 'n'))} & {n0(get('xgb_cox', 'n_obitos'))} & {f(get('cox_idade', 'c_index'))}"
                 f" & {f(get('xgb_cox', 'c_index'))} & "
                 f"{f(get('xgb_cox', 'c_intrafaixa'))} & {oe} \\\\")
    L.append(f"Rótulos permutados ({len(perm)} sementes) & 8.073 & 2.417 & --- & "
             f"{f(perm.c_teste.mean())}$\\pm${f(perm.c_teste.std())} & {f(perm.c_intrafaixa.mean())}$\\pm${f(perm.c_intrafaixa.std())} & --- \\\\")
    L.append("\\hline\n\\end{tabular}\n\\end{table}\n")
    (TAB / "tab_sens.tex").write_text("\n".join(L))


def tab_sub():
    s = pd.read_csv(R / "subgrupos.csv")
    L = [CAB, r"""\begin{table}[!t]
\caption{Desempenho do XGBoost por subgrupo no teste. C: C-index de Harrell;
$\Delta C$: ganho sobre o modelo só com idade (IC 95\%, 200 réplicas de tutores); O/E em 2 anos do XGBoost e do modelo só com idade}
\label{tab:sub}
\centering
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{|l|r|r|c|c|c|c|}
\hline
\textbf{Subgrupo} & \textbf{Cães} & \textbf{Óbitos} & \textbf{\shortstack{C\\XGB}} & \textbf{$\Delta C$ vs idade} & \textbf{\shortstack{O/E\\XGB}} & \textbf{\shortstack{O/E\\idade}} \\
\hline"""]
    for grupo, d in s.groupby("grupo", sort=False):
        L.append(f"\\multicolumn{{7}}{{|l|}}{{\\textit{{{grupo}}}}} \\\\")
        for _, r in d.iterrows():
            L.append(f"\\quad {r.nivel.replace('–', '--').replace('≥', '$\\geq$').replace('<', '$<$')} & {n0(r.n)} & {n0(r.n_obitos)} & "
                     f"{f(r.c_xgb_cox)} & {f(r.dc)} [{f(r.dc_lo)}--{f(r.dc_hi)}] & "
                     f"{f(r.oe_xgb_cox, 2)} & {f(r.oe_cox_idade, 2)} \\\\")
        L.append("\\hline")
    L.append("\\end{tabular}\n\\end{table}\n")
    (TAB / "tab_sub.tex").write_text("\n".join(L).replace("US$", "US\\$"))


def tab_hr():
    h = pd.read_csv(R / "hr_especificacoes.csv")
    rot = [("peso_mudanca_12m=perdeu", "Perdeu peso nos últimos 12 meses (vs estável)"),
           ("apetite=fraco", "Apetite fraco (vs normal)"), ("sobrepeso_ever=sim", "Já esteve acima do peso (vs não)"),
           ("atividade_nivel", "Nível de atividade (por dp)"), ("minutos_ativos_dia", "Minutos ativos por dia (por dp)"),
           ("alimentacao=1x", "Uma refeição por dia (vs duas)"), ("dieta_principal=caseira", "Dieta caseira (vs seca comercial)"),
           ("castracao=<6m", "Castrado antes dos 6 meses (vs intacto)"), ("castracao=6-12m", "Castrado aos 6--12 meses (vs intacto)"),
           ("peso_kg", "Peso (por dp)"), ("porte", "Porte (por dp)"), ("saude_geral", "Saúde geral pior (por dp)"),
           ("renda=<40k", "Renda $<$ US\\$ 40 mil (vs 80--120 mil)"), ("renda=120k+", "Renda $\\geq$ US\\$ 120 mil (vs 80--120 mil)")]
    esp = ["A_estaveis_estilo", "B_principal", "C_principal_renda", "D_principal_landmark1a"]
    L = [CAB, r"""\begin{table*}[!t]
\caption{Razões de risco (IC 95\%) do modelo de Cox no treino em especificações aninhadas. A: estáveis + estilo de vida;
B: modelo principal (A + saúde na entrada); C: B + renda e escolaridade do tutor; D: B restrito aos cães vivos em 1 ano,
com o tempo contado a partir daí. Associações condicionais ao conjunto de ajuste, não efeitos causais}
\label{tab:hr}
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Atributo} & \textbf{A: estáveis + estilo} & \textbf{B: principal} & \textbf{C: B + renda e escolaridade} & \textbf{D: B, vivos em 1 ano} \\
\hline"""]
    for k, r in rot:
        vals = []
        for e in esp:
            x = h[(h.variavel == k) & (h.especificacao == e)]
            vals.append("---" if x.empty else f"{f(x.hr.iloc[0], 2)} ({f(x.lo.iloc[0], 2)}--{f(x.hi.iloc[0], 2)})")
        L.append(f"{r} & " + " & ".join(vals) + " \\\\")
    L.append("\\hline\n\\end{tabular}\n\\end{table*}\n")
    (TAB / "tab_hr.tex").write_text("\n".join(L))


def figuras():
    for nome in ["roc_pr_2a.pdf", "calibracao_2a.pdf", "shap.pdf"]:
        shutil.copy(ROOT / "figuras" / "etapa3" / nome, FIGA / nome)


if __name__ == "__main__":
    tab_resultados(), tab_clf(), tab_sens(), tab_sub(), tab_hr(), figuras()
    print("tabelas e figuras gravadas em", ART)
