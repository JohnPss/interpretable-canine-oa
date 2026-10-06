"""Tabelas descritivas do artigo (seção Materiais e Métodos), geradas do dado — nenhum número à mão.

Entrada:  data/processed/features.csv (03_features.py) — população em risco do desenho (C).
Saída:    ../Artigo/artigo/tabelas/tab_populacao.tex (só agregados; nenhum dado individual sai daqui).

Faixas etárias: mesmos cortes do 02b_km_faixa_etaria.py e do modelagem_comum (lo <= idade < hi).
Incidência acumulada: Kaplan-Meier em tempo discreto, n em risco no intervalo k = #(tempo >= k).

Uso: python 20_tabelas_artigo.py [--saida DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FAIXAS = [("Geral", 0, np.inf), ("$<$7", 0, 7), ("7--10", 7, 10), ("$\\geq$10", 10, np.inf)]


def km_acumulada(tempo, evento, ate):
    """1 - S(ate), S(k) = prod_{j<=k} (1 - d_j / n_j)."""
    s = 1.0
    for k in range(1, ate + 1):
        n = np.sum(tempo >= k)
        d = np.sum((tempo == k) & (evento == 1))
        if n:
            s *= 1 - d / n
    return 1 - s


def br(x, casas=1):
    """Número no formato brasileiro (milhar com ponto, decimal com vírgula)."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(serie):
    return br(100 * serie.mean())


def coluna(d):
    t, e = d.tempo.to_numpy(), d.evento.to_numpy()
    q1, q2, q3 = d.idade.quantile([0.25, 0.5, 0.75])
    return {
        "Cães": br(len(d), 0),
        "Casos incidentes": br(e.sum(), 0),
        "Cão-anos": br(t.sum(), 0),
        "Casos por 1.000 cão-anos": br(1000 * e.sum() / t.sum()),
        "Incidência em 3 anos (\\%)": br(100 * km_acumulada(t, e, 3)),
        "Idade, mediana (anos)": br(q2),
        "Idade, IIQ (anos)": f"{br(q1)}--{br(q3)}",
        "Fêmeas (\\%)": pct(d.sexo == 1),
        "Castrados (\\%)": pct(d.castracao != "intacto"),
        "Raça pura (\\%)": pct(d.raca_pura == 1),
        "Grande ou gigante (\\%)": pct(d.porte >= 3),
        "Displasia de quadril (\\%)": pct(d.displasia_quadril == 1),
        "Ruptura de cruzado (\\%)": pct(d.ruptura_cruzado == 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(ROOT.parent / "Artigo" / "artigo" / "tabelas"))
    saida = Path(ap.parse_args().saida)
    saida.mkdir(parents=True, exist_ok=True)

    f = pd.read_csv(PROC / "features.csv")
    cols = {nome: coluna(f[(f.idade >= lo) & (f.idade < hi)]) for nome, lo, hi in FAIXAS}
    tab = pd.DataFrame(cols)
    print(tab.to_string())

    linhas = [
        "% Gerado por interpretable-canine-oa/20_tabelas_artigo.py — não editar à mão.",
        "\\begin{table}[!t]",
        "\\caption{População em risco, geral e por faixa etária na entrada (anos). Incidência em 3 anos por Kaplan-Meier; IIQ: intervalo interquartil; displasia e ruptura de cruzado relatadas na entrada}",
        "\\label{tab:populacao}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{|>{\\raggedright\\arraybackslash}p{3.0cm}|r|r|r|r|}",
        "\\hline",
        " & " + " & ".join(f"\\textbf{{{c}}}" for c in tab.columns) + " \\\\",
        "\\hline",
    ]
    for rot, vals in tab.iterrows():
        linhas.append(f"{rot} & " + " & ".join(vals) + " \\\\")
        linhas.append("\\hline")
    linhas += ["\\end{tabular}", "\\end{table}", ""]
    (saida / "tab_populacao.tex").write_text("\n".join(linhas), encoding="utf-8")
    print(f"\nSalvo: {saida / 'tab_populacao.tex'}")


if __name__ == "__main__":
    main()
