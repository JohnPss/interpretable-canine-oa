"""Fase 4a — descritivo da população de longevidade (só agregados; nenhum dado individual sai daqui).

Entrada:  data/processed/features.csv
Saídas:   resultados/descritivo_populacao.csv, resultados/km_faixa_etaria.csv
          figuras/km_faixa_etaria.png
          ../Artigo/artigo/tabelas/tab_populacao.tex (Tabela da seção Materiais e Métodos)

Faixas etárias: as mesmas de modelagem.py (lo <= idade < hi). Mortalidade acumulada = 1 − Kaplan-Meier.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import modelagem as mc

ROOT = Path(__file__).parent
RES, FIG = ROOT / "resultados", ROOT / "figuras"


def br(x, casas=1):
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(serie):
    return br(100 * serie.mean())


def coluna(d):
    t, e = d.tempo.to_numpy(), d.evento.to_numpy()
    q1, q2, q3 = d.idade.quantile([0.25, 0.5, 0.75])
    return {
        "Cães": br(len(d), 0),
        "Óbitos": br(e.sum(), 0),
        "Cão-anos": br(t.sum(), 0),
        "Óbitos por 1.000 cão-anos": br(1000 * e.sum() / t.sum()),
        "Mortalidade em 2 anos (\\%)": br(100 * mc.km_risco(t, e, 2)),
        "Idade, mediana (anos)": br(q2),
        "Idade, IIQ (anos)": f"{br(q1)}--{br(q3)}",
        "Fêmeas (\\%)": pct(d.sexo == 1),
        "Castrados (\\%)": pct(d.castracao != "intacto"),
        "Raça pura (\\%)": pct(d.raca_pura == 1),
        "Grande ou gigante (\\%)": pct(d.porte >= 3),
        "Saúde excelente ou muito boa (\\%)": pct(d.saude_geral <= 1),
        "Alguma doença na entrada (\\%)": pct(d.n_sistemas_doenca > 0),
        "Câncer na entrada (\\%)": pct(d.doenca_cancer == 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabelas", default=str(ROOT.parent / "Artigo" / "artigo" / "tabelas"))
    dir_tab = Path(ap.parse_args().tabelas)
    RES.mkdir(exist_ok=True), FIG.mkdir(exist_ok=True), dir_tab.mkdir(parents=True, exist_ok=True)

    f = mc.carregar_dados()
    fx = mc.faixa_etaria(f.idade)
    grupos = [("Geral", f)] + [(rot.replace(" anos", ""), f[fx == rot]) for *_, rot in mc.FAIXAS]
    tab = pd.DataFrame({nome: coluna(d) for nome, d in grupos})
    tab.to_csv(RES / "descritivo_populacao.csv")
    print(tab.to_string())

    # --- Tabela LaTeX ---
    cab = {"Geral": "Geral", "< 7": "$<$7", "7–10": "7--10", "10–13": "10--13", "13+": "$\\geq$13"}
    linhas = [
        "% Gerado por interpretable-canine-oa/04_descritivo.py — não editar à mão.",
        "\\begin{table}[!t]",
        "\\caption{População do estudo, geral e por faixa etária na entrada (anos). Mortalidade em 2 anos "
        "por Kaplan-Meier; IIQ: intervalo interquartil; saúde e doenças relatadas pelo tutor na entrada}",
        "\\label{tab:populacao}",
        "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{|>{\\raggedright\\arraybackslash}p{2.75cm}|r|r|r|r|r|}", "\\hline",
        " & " + " & ".join(f"\\textbf{{{cab[c]}}}" for c in tab.columns) + " \\\\", "\\hline",
    ]
    for rot, vals in tab.iterrows():
        linhas += [f"{rot} & " + " & ".join(vals) + " \\\\", "\\hline"]
    linhas += ["\\end{tabular}", "\\end{table}", ""]
    (dir_tab / "tab_populacao.tex").write_text("\n".join(linhas), encoding="utf-8")

    # --- Kaplan-Meier por faixa etária ---
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    km_linhas = []
    for *_, rot in mc.FAIXAS:
        s = f[fx == rot]
        km = KaplanMeierFitter().fit(s.tempo, s.evento, label=f"{rot} (n={len(s):,})".replace(",", "."))
        km.plot_survival_function(ax=ax, ci_show=True, linewidth=1.4)
        for h in mc.HORIZONTES:
            ic = km.confidence_interval_survival_function_.asof(h)
            km_linhas.append(dict(faixa=rot, horizonte=h, n_risco=int((s.tempo >= h).sum()),
                                  mortalidade=1 - km.predict(h), ic_lo=1 - ic.iloc[1], ic_hi=1 - ic.iloc[0]))
    ax.set_xlabel("Anos desde a entrada no estudo")
    ax.set_ylabel("Sobrevida (Kaplan-Meier)")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.legend(title="Idade na entrada", fontsize=7, title_fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "km_faixa_etaria.png", dpi=200)
    km_tab = pd.DataFrame(km_linhas)
    km_tab.to_csv(RES / "km_faixa_etaria.csv", index=False)
    print("\n", km_tab.round(3).to_string(index=False))
    print(f"\nSalvo: {dir_tab / 'tab_populacao.tex'}, {FIG / 'km_faixa_etaria.png'}")


if __name__ == "__main__":
    main()
