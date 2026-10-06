"""Fase 1b — apuração assimétrica do óbito e critério de participação ativa (D22, risco R03).

Problema: o cão vivo é censurado no último contato (Survival_Date, qualquer questionário), mas o óbito de um
cão cujo tutor parou de responder ainda pode chegar depois (status do estudo, EOLS). Esses óbitos entram
como evento com um tempo em risco que os vivos na mesma situação não têm → censura possivelmente informativa.

Medida: para cada óbito incluído, a LACUNA entre o último contato com o cão vivo e a data do óbito.
  - Contatos datados na release aberta: HLES (DAP_Pack_Date), AFUS (afus_complete_date) e CSLB (cslb_date).
    Os demais questionários (jogos cognitivos, My Profile, morfometria...) só têm flag por ano, sem data;
    por isso o último contato observado é um LIMITE INFERIOR e a lacuna é um LIMITE SUPERIOR (conservador).
  - ~18% das linhas do AFUS não têm afus_complete_date. Variante "aprox": AFUS do ano k sem data
    = HLES + k anos (só para checar se a conclusão muda).
Critério (D22): participação ativa no óbito = algum contato com o cão vivo nos LIMIAR_MESES antes do óbito.
  18 meses = o AFUS é anual; tolera um questionário respondido com atraso, mas não um ano inteiro pulado.
Sensibilidade "censura simétrica": óbito SEM participação ativa vira censura no último contato vivo
  observado (a mesma regra aplicada aos vivos); se esse contato é o próprio HLES, o cão sai (tempo = 0, D06).

Entrada:  data/raw/ DogOverview, AFUS_dog_owner, CSLB; data/processed/populacao.csv
Saída:    data/processed/apuracao.csv (individual, fora do git)
          resultados/apuracao_obito.csv (agregados), resultados/apuracao_km.csv (mortalidade em 2 anos)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW, PROC, RES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "resultados"
DIAS_ANO = 365.25
LIMIAR_MESES = 18
LIMIARES = (12, 18, 24)
FAIXAS = [(0, 7, "< 7 anos"), (7, 10, "7–10 anos"), (10, 13, "10–13 anos"), (13, 99, "13+ anos")]


def km_risco(tempo, evento, h=2.0):
    s = 1.0
    for t in np.unique(tempo[(evento == 1) & (tempo <= h)]):
        s *= 1 - np.sum((tempo == t) & (evento == 1)) / np.sum(tempo >= t)
    return 1 - s


def faixa(idade):
    return pd.cut(idade, [lo for lo, *_ in FAIXAS] + [99], right=False, labels=[n for *_, n in FAIXAS])


def contatos(ov, aprox):
    af = pd.read_csv(RAW / "DAP_2024_AFUS_dog_owner_v1.1.csv",
                     usecols=["dog_id", "afus_followup_year", "afus_complete_date"])
    cs = pd.read_csv(RAW / "DAP_2024_CSLB_v1.0.csv", usecols=["dog_id", "cslb_date"])
    af["data"] = pd.to_datetime(af.afus_complete_date, errors="coerce")
    if aprox:
        hles = af.dog_id.map(ov.set_index("dog_id").DAP_Pack_Date)
        sem = af.data.isna()
        af.loc[sem, "data"] = hles[sem] + pd.to_timedelta(af.afus_followup_year[sem] * DIAS_ANO, unit="D")
    cs["data"] = pd.to_datetime(cs.cslb_date, errors="coerce")
    return pd.concat([ov[["dog_id", "DAP_Pack_Date"]].rename(columns={"DAP_Pack_Date": "data"}),
                      af[["dog_id", "data"]], cs[["dog_id", "data"]]]).dropna()


def ultimo_vivo(ov, mortos, aprox):
    """Último contato datado ESTRITAMENTE antes do óbito (no dia do óbito o questionário pode ser o próprio aviso)."""
    c = contatos(ov, aprox).merge(mortos[["dog_id", "data_obito"]], on="dog_id")
    c = c[c.data < c.data_obito]
    return c.groupby("dog_id").data.max()


def main():
    ov = pd.read_csv(RAW / "DAP_2024_DogOverview_v1.1.csv", usecols=["dog_id", "DAP_Pack_Date", "Survival_Date"])
    for c in ["DAP_Pack_Date", "Survival_Date"]:
        ov[c] = pd.to_datetime(ov[c], errors="coerce")
    pop = pd.read_csv(PROC / "populacao.csv")
    inc = pop[pop.incluido == 1].merge(ov, on="dog_id")
    inc["faixa"] = faixa(inc.idade_hles)
    mortos = inc[inc.evento == 1].rename(columns={"Survival_Date": "data_obito"})

    linhas, saida = [], inc[["dog_id"]].copy()
    for variante, aprox in [("datas_observadas", False), ("afus_sem_data_aprox", True)]:
        uv = ultimo_vivo(ov, mortos, aprox)
        m = mortos.assign(ultimo=mortos.dog_id.map(uv))
        assert m.ultimo.notna().all(), "óbito sem nenhum contato vivo antes (nem o HLES)"
        m["lacuna_meses"] = (m.data_obito - m.ultimo).dt.days / (DIAS_ANO / 12)
        for estrato, s in [("geral", m)] + [(n, m[m.faixa == n]) for *_, n in FAIXAS] + \
                          [(f"fonte: {f}", g) for f, g in m.groupby("Survival_Source")]:
            linha = dict(variante=variante, estrato=estrato, obitos=len(s),
                         lacuna_mediana_meses=round(s.lacuna_meses.median(), 1),
                         lacuna_p90_meses=round(s.lacuna_meses.quantile(0.9), 1))
            for L in LIMIARES:
                linha[f"n_lacuna_gt{L}m"] = int((s.lacuna_meses > L).sum())
                linha[f"pct_lacuna_gt{L}m"] = round(100 * (s.lacuna_meses > L).mean(), 1)
            linhas.append(linha)
        if not aprox:
            # Censura simétrica (variante principal, conservadora)
            inativo = m.lacuna_meses > LIMIAR_MESES
            t_sim = (m.ultimo - m.DAP_Pack_Date).dt.days / DIAS_ANO
            saida = saida.merge(pd.DataFrame({"dog_id": m.dog_id, "lacuna_meses": m.lacuna_meses,
                                              "obito_sem_participacao": inativo.astype(int),
                                              "tempo_censura_simetrica": t_sim.where(inativo)}),
                                on="dog_id", how="left")
    res = pd.DataFrame(linhas)

    d = inc.merge(saida, on="dog_id")
    d["obito_sem_participacao"] = d.obito_sem_participacao.fillna(0).astype(int)
    d["tempo_sim"] = np.where(d.obito_sem_participacao == 1, d.tempo_censura_simetrica, d.tempo)
    d["evento_sim"] = np.where(d.obito_sem_participacao == 1, 0, d.evento)
    d["incluido_sim"] = (d.tempo_sim > 0).astype(int)
    d[["dog_id", "lacuna_meses", "obito_sem_participacao", "tempo_sim", "evento_sim", "incluido_sim"]].to_csv(
        PROC / "apuracao.csv", index=False)

    km = []
    for estrato, s in [("geral", d)] + [(n, d[d.faixa == n]) for *_, n in FAIXAS]:
        s2 = s[s.incluido_sim == 1]
        km.append(dict(estrato=estrato, caes=len(s), obitos=int(s.evento.sum()),
                       mort2a_original=round(100 * km_risco(s.tempo.to_numpy(), s.evento.to_numpy()), 1),
                       caes_sim=len(s2), obitos_sim=int(s2.evento_sim.sum()),
                       mort2a_simetrica=round(100 * km_risco(s2.tempo_sim.to_numpy(), s2.evento_sim.to_numpy()), 1)))
    km = pd.DataFrame(km)

    RES.mkdir(exist_ok=True)
    res.to_csv(RES / "apuracao_obito.csv", index=False)
    km.to_csv(RES / "apuracao_km.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print(km.to_string(index=False))
    print(f"Critério: contato vivo nos {LIMIAR_MESES} meses antes do óbito. Óbitos sem participação ativa: "
          f"{d.obito_sem_participacao.sum()} de {d.evento.sum()}; saem da análise simétrica (tempo 0): "
          f"{int((d.incluido_sim == 0).sum())}")


if __name__ == "__main__":
    main()
