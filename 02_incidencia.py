"""Fase 2 — desfecho de incidência de OA no acompanhamento anual (AFUS).

Desenho (C): features medidas no HLES (baseline), desfecho = OA nova no AFUS.
É o único desenho em que as features são anteriores ao diagnóstico.

Entrada:  data/processed/amostra.csv (saída do 01_amostra.py)
          data/raw/DAP_2024_AFUS_health_condition_v1.0.csv
Saída:    data/processed/incidencia.csv — um cão em risco por linha, com:
            tempo, evento        → análise principal (Cox em tempo discreto anual)
            oa_2anos             → sensibilidade (janela fixa de 2 anos; NaN = não elegível)

Decisões documentadas em metodologia_projeto.md (Fase 2).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

OA = 1113
ANOS_AFUS = [1, 2, 3, 4]  # AFUS_Year5 está vazio nesta release
JANELA = 2
POS_SAUDE = 8  # posição da seção Health Status em AFUS_YearN_Complete (ver abaixo)
REGEX_ARTRITE = r"arthrit|\bDJD\b|degenerative joint"


def corrige_ano(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 100, s + 2000)


def main():
    df = pd.read_csv(PROC / "amostra.csv", parse_dates=["hles_data"], low_memory=False)
    afus = pd.read_csv(RAW / "DAP_2024_AFUS_health_condition_v1.0.csv", low_memory=False)
    afus = afus[afus.dog_id.isin(df.dog_id)]
    n0 = len(df)

    # --- Anos de AFUS observados ---
    # AFUS_YearN_Complete tem 12 caracteres, um por seção (instrumento) do AFUS. O
    # codebook não diz a ordem; ela foi determinada cruzando com AFUS_dog_owner
    # (concordância de 100% entre cada posição e a presença de dados de um bloco):
    #   1 oc | 2 dd | 3 pa | 4 de | 5 db | 6 dieta | 7 mp | 8 hs (Health Status) |
    #   9 od | 10 dora | 11 mdors | 12 ac
    # Ano observado = seção de saúde (posição 8) respondida.
    for k in ANOS_AFUS:
        df[f"afus{k}"] = df[f"AFUS_Year{k}_Complete"].map(
            lambda v: isinstance(v, str) and len(v) >= POS_SAUDE and v[POS_SAUDE - 1] == "Y")
    # Último ano observado = tempo de censura. Anos com buraco (ex.: respondeu 1 e 3,
    # pulou o 2) contam como observados: a pergunta de condições do AFUS é "Since you
    # last filled out the Health Status survey, has your dog ever been diagnosed...",
    # então o questionário seguinte cobre o intervalo pulado (codebook, conferido).
    obs = df[[f"afus{k}" for k in ANOS_AFUS]].to_numpy()
    df["ultimo_ano_obs"] = np.where(obs.any(axis=1), obs.shape[1] - np.argmax(obs[:, ::-1], axis=1), 0)

    # --- Primeiro relato de OA no AFUS ---
    ev = (afus[afus.afus_hs_new_condition == OA]
          .sort_values("afus_followup_year").drop_duplicates("dog_id")
          [["dog_id", "afus_followup_year", "afus_hs_new_condition_diagnosis_year"]])
    ev["dx_ano"] = corrige_ano(ev.afus_hs_new_condition_diagnosis_year)
    df = df.merge(ev, on="dog_id", how="left")
    df["relato_oa_afus"] = df.afus_followup_year.notna()
    # Diferença em anos-calendário entre diagnóstico e HLES (mês não é confiável).
    df["d"] = df.dx_ano - df.hles_ano

    # --- População em risco: exclusões, na ordem ---
    exclusoes = {}
    # 1. OA já no HLES: prevalente, não está em risco. Inclui os cães que reaparecem
    #    como condição "nova" no AFUS (ruído do autorrelato, não caso novo).
    m = df.oa_hles
    exclusoes["OA no HLES (prevalente)"] = m.sum()
    exclusoes["  dos quais reaparecem como 'nova' no AFUS"] = (m & df.relato_oa_afus).sum()
    df = df[~m]
    # 2. "Lembrança tardia": relatado no AFUS, mas diagnosticado em ano anterior ao
    #    HLES. O cão já tinha OA no baseline e o tutor não citou: é prevalente, sai
    #    da população em risco desde o início (mesmo tratamento do item 1).
    m = df.relato_oa_afus & (df.d < 0)
    exclusoes["Lembrança tardia (dx antes do HLES)"] = m.sum()
    df = df[~m]
    # 3. Texto livre de artrite no HLES sem 1113: possível OA prevalente. Tanto
    #    "recodificar como positivo" quanto "excluir" tiram o cão da população em
    #    risco, então aqui a decisão pendente não muda nada.
    m = df.flag_texto_artrite_hles
    exclusoes["Texto livre de artrite no HLES"] = m.sum()
    df = df[~m]
    # 4. Seção de saúde nunca respondida no AFUS: não contribui tempo em risco.
    #    Inclui todo HLES 2024 (AFUS ainda não aconteceu). Cão com relato de OA mas
    #    sem 'Y' na posição 8 naquele ano fica: o relato é evidência de que a seção
    #    foi respondida.
    m = (df.ultimo_ano_obs == 0) & ~df.relato_oa_afus
    exclusoes["Sem AFUS observado"] = m.sum()
    df = df[~m].copy()

    # --- Tempo e evento (escala anual discreta, 1..4) ---
    # Tempo do evento = intervalo anual em que o diagnóstico ocorreu, estimado pelo
    # ano-calendário: max(d, 1). d == 0 (diagnóstico no mesmo ano do HLES, 175 casos
    # após as exclusões, impossíveis de ordenar sem o mês) vira intervalo 1 = "evento no primeiro ano".
    # Limitado ao ano de relato: o diagnóstico não pode ser depois do questionário
    # que o relatou (poucos casos com d > ano de relato = erro de data).
    df["evento"] = df.relato_oa_afus.astype(int)
    t_ev = np.minimum(np.maximum(df.d, 1), df.afus_followup_year)
    df["tempo"] = np.where(df.evento == 1, t_ev, df.ultimo_ano_obs).astype(int)
    df["flag_dx_mesmo_ano_hles"] = df.relato_oa_afus & (df.d == 0)
    df["flag_dx_apos_relato"] = df.relato_oa_afus & (df.d > df.afus_followup_year)
    # Evento relatado num ano em que a string não marca a seção de saúde como completa.
    ano_rel = df.afus_followup_year.fillna(1).astype(int)
    saude_ok = pd.Series([df.at[i, f"afus{k}"] for i, k in zip(df.index, ano_rel)], index=df.index)
    df["flag_evento_sem_saude_Y"] = df.relato_oa_afus & ~saude_ok
    # Morte = censura não informativa (simplificação; ver Limitações). O cão que morreu
    # simplesmente para de ter AFUS e é censurado no último ano observado.

    # --- Sensibilidade: janela fixa de 2 anos ---
    # Caso: evento com tempo <= 2. Não-caso: sem evento até 2 e observado até o ano
    # 2 ou depois. Censurado antes de 2 sem evento = não elegível (NaN).
    caso = (df.evento == 1) & (df.tempo <= JANELA)
    nao_caso = ~caso & (df.ultimo_ano_obs >= JANELA)
    df["oa_2anos"] = np.where(caso, 1.0, np.where(nao_caso, 0.0, np.nan))

    # --- Texto livre de artrite no AFUS sem 1113: fica como não-evento, com flag ---
    txt = afus.afus_hs_new_condition_other_description.astype(str).str.contains(REGEX_ARTRITE, case=False, regex=True)
    df["flag_texto_artrite_afus"] = df.dog_id.isin(afus.loc[txt, "dog_id"]) & (df.evento == 0)

    cols = ["dog_id", "owner_id", "Cohort", "hles_ano", "Estimated_Age_Years_at_HLES",
            "AgeDOB_Estimation_Method", "Survival_Status", "ultimo_ano_obs",
            "tempo", "evento", "oa_2anos", "afus_followup_year", "dx_ano",
            "flag_dx_mesmo_ano_hles", "flag_dx_apos_relato", "flag_evento_sem_saude_Y",
            "flag_texto_artrite_afus"]
    df[cols].to_csv(PROC / "incidencia.csv", index=False)

    # --- Relatório ---
    print(f"Amostra de entrada: {n0}")
    for k, v in exclusoes.items():
        print(f"  - {k}: {v}")
    print(f"População em risco: {len(df)} | eventos: {df.evento.sum()} "
          f"| cão-anos: {df.tempo.sum()} | taxa: {1000 * df.evento.sum() / df.tempo.sum():.1f}/1000 cão-ano")
    print(f"  dx no mesmo ano do HLES (tempo=1): {df.flag_dx_mesmo_ano_hles.sum()}")
    print(f"  dx após o ano de relato (tempo truncado): {df.flag_dx_apos_relato.sum()}")
    print(f"  evento em ano sem seção de saúde marcada 'Y': {df.flag_evento_sem_saude_Y.sum()}")
    print(f"  texto livre artrite no AFUS (não-evento, flag): {df.flag_texto_artrite_afus.sum()}")
    print("Eventos / censuras por tempo:")
    print(pd.crosstab(df.tempo, df.evento.map({1: "evento", 0: "censura"})))
    print(f"Janela {JANELA} anos: elegíveis {df.oa_2anos.notna().sum()}, "
          f"casos {int(df.oa_2anos.sum())} ({df.oa_2anos.mean():.2%})")
    print(f"Salvo: {PROC / 'incidencia.csv'}")


if __name__ == "__main__":
    main()
