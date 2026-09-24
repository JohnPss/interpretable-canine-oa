"""Fase 2 (descritivo) — incidência acumulada de OA por faixa etária no HLES (Kaplan-Meier).

Complemento descritivo ao Cox: mostra a incidência por estrato sem modelo.
Tempo anual discreto (1..4), igual ao 02_incidencia.py; morte = censura.

Entrada:  data/processed/incidencia.csv
Saída:    figuras/km_faixa_etaria.png e figuras/km_faixa_etaria.csv (tabela da figura)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PROC, FIG = ROOT / "data" / "processed", ROOT / "figuras"

FAIXAS = [(0, 7, "< 7 anos"), (7, 10, "7–10 anos"), (10, 99, "10+ anos")]
CORES = ["#2a78d6", "#eb6834", "#1baf7a"]  # paleta categórica validada (dataviz), ordem fixa
TINTA, TINTA_2, GRADE = "#1f1f1e", "#5f5e58", "#e4e3dd"


def kaplan_meier(tempo, evento):
    """KM em tempo discreto. Retorna tabela por tempo com S(t) e IC 95% (Greenwood, log-log)."""
    linhas, s, var_acc = [], 1.0, 0.0
    for t in sorted(np.unique(tempo)):
        n_risco = (tempo >= t).sum()
        d = ((tempo == t) & (evento == 1)).sum()
        s *= 1 - d / n_risco
        if n_risco > d:
            var_acc += d / (n_risco * (n_risco - d))
        # IC log(-log) para S(t)
        se = np.sqrt(var_acc) / abs(np.log(s)) if 0 < s < 1 else 0.0
        lo, hi = s ** np.exp(1.96 * se), s ** np.exp(-1.96 * se)
        linhas.append(dict(tempo=t, em_risco=n_risco, eventos=d, S=s, S_lo=lo, S_hi=hi))
    return pd.DataFrame(linhas)


def main():
    df = pd.read_csv(PROC / "incidencia.csv")
    FIG.mkdir(exist_ok=True)

    tabs = []
    for lo, hi, nome in FAIXAS:
        g = df[(df.Estimated_Age_Years_at_HLES >= lo) & (df.Estimated_Age_Years_at_HLES < hi)]
        km = kaplan_meier(g.tempo.to_numpy(), g.evento.to_numpy())
        # Incidência acumulada = 1 − S(t), em %.
        km = km.assign(faixa=nome, n_faixa=len(g), inc=100 * (1 - km.S),
                       inc_lo=100 * (1 - km.S_hi), inc_hi=100 * (1 - km.S_lo))
        tabs.append(km)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(FIG / "km_faixa_etaria.csv", index=False)

    # Anos 1–3: linha contínua. Ano 4: tracejado + faixa de IC mais clara. Só a coorte
    # HLES 2019–2020 chega ao ano 4, com 23% (< 7), 19% (7–10) e 8% (10+) do n do ano 1.
    # Na faixa 10+ o IC do risco no intervalo passa de ±0,8 p.p. (ano 1) para ±3,0 p.p.
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.6, 5.0), dpi=200, sharex=True,
                                  gridspec_kw=dict(height_ratios=[4, 1], hspace=0.08))
    for i, ((lo, hi, nome), cor) in enumerate(zip(FAIXAS, CORES)):
        k = tab[tab.faixa == nome].set_index("tempo")
        t = np.array([0, 1, 2, 3])
        inc = np.r_[0, k.inc.loc[[1, 2, 3]]]
        ilo, ihi = np.r_[0, k.inc_lo.loc[[1, 2, 3]]], np.r_[0, k.inc_hi.loc[[1, 2, 3]]]
        ax.step(np.r_[t, 4], np.r_[inc, inc[-1]], where="post", color=cor, lw=2)
        ax.fill_between(np.r_[t, 4], np.r_[ilo, ilo[-1]], np.r_[ihi, ihi[-1]], step="post", color=cor, alpha=0.15, lw=0)
        # Ano 4 (menor precisão)
        i4, l4, h4 = k.inc.loc[4], k.inc_lo.loc[4], k.inc_hi.loc[4]
        ax.plot([4, 4, 4.5], [inc[-1], i4, i4], color=cor, lw=1.6, ls=(0, (3, 2)))
        ax.fill_between([4, 4.5], [l4, l4], [h4, h4], color=cor, alpha=0.07, lw=0)
        ax.annotate(f"{inc[-1]:.1f}% (3 a)  ·  {i4:.1f}% (4 a)", xy=(4.5, i4), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TINTA)
        n = int(k.n_faixa.iloc[0])
        ax.plot([], [], color=cor, lw=2, label=f"{nome} (n = {n:,})".replace(",", "."))
        # Tabela de número em risco
        for tt in range(1, 5):
            axr.text(tt - 0.5, 2 - i, f"{int(k.em_risco.loc[tt]):,}".replace(",", "."), ha="center",
                     va="center", fontsize=7.5, color=TINTA if tt < 4 else TINTA_2,
                     style="normal" if tt < 4 else "italic")
        axr.text(-0.08, 2 - i, nome, ha="right", va="center", fontsize=7.5, color=cor, fontweight="bold")

    ax.axvspan(4, 4.5, color=GRADE, alpha=0.35, lw=0)
    ax.text(4.25, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] else 0.5, "ano 4:\nmenor n",
            ha="center", va="bottom", fontsize=7, color=TINTA_2)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, None)
    ax.set_ylabel("Incidência acumulada de OA (%)", color=TINTA_2, fontsize=9)
    ax.grid(axis="y", color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    for a in (ax, axr):
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            a.spines[sp].set_color(GRADE)
        a.tick_params(colors=TINTA_2, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=TINTA)
    ax.set_title("Incidência acumulada de osteoartrite por idade no HLES\n(Kaplan-Meier, IC 95%)",
                 fontsize=10, color=TINTA, loc="left")
    axr.set_ylim(-0.6, 2.9)
    axr.set_yticks([])
    axr.spines["left"].set_visible(False)
    axr.set_title("Cães em risco no início de cada intervalo", fontsize=7.5, color=TINTA_2, loc="left", pad=2)
    axr.set_xticks(range(5))
    axr.set_xlabel("Anos de acompanhamento após o HLES", color=TINTA_2, fontsize=9)
    fig.text(0.01, -0.01,
             "Ano 4 (tracejado): só a coorte HLES 2019–2020 chega a esse ponto (8–23% do n inicial da faixa);\n"
             "estimativa menos precisa, sobretudo em 10+ anos. Ano 1 inclui diagnósticos do mesmo ano-calendário do HLES.",
             fontsize=7, color=TINTA_2, ha="left", va="top")
    fig.savefig(FIG / "km_faixa_etaria.png", bbox_inches="tight")

    print(tab[["faixa", "tempo", "em_risco", "eventos", "inc", "inc_lo", "inc_hi"]].round(2).to_string(index=False))
    print(f"Salvo: {FIG / 'km_faixa_etaria.png'}")


if __name__ == "__main__":
    main()
