"""Fase 1 — população e desfecho de longevidade (morte por qualquer causa após o HLES).

Entrada:  data/raw/ DogOverview, HLES_dog_owner (owner_id), EOLS (causa da morte, só descritivo)
Saída:    data/processed/populacao.csv — um cão por linha (todos os 49.077 das coortes), com
          `incluido` (entra na análise), tempo/evento e flags.

Definições (metodologia_projeto.md, Fase 1):
  - Coortes DAP Pack + Foundation. Precision fica fora (exames, fora do escopo).
  - Origem do tempo = data do HLES (= DAP_Pack_Date; conferido: diferença 0 dia em todos).
  - Evento = óbito (Survival_Status == "Dead"); data = Survival_Date.
  - CORTE ADMINISTRATIVO em 2024-12-31. O status "vivo" só é apurado até essa data (máximo de
    Survival_Date entre os vivos), mas há óbitos registrados até 2026. Sem o corte, depois de 2024
    só os cães que morreram continuariam "em risco" e o risco seria superestimado. Óbito após o corte
    = censura no corte (o cão estava vivo em 2024-12-31).
  - Censura = último contato com o cão vivo (Survival_Date dos vivos, de qualquer questionário).
  - Excluídos: sem Survival_Days (75, óbito sem data) e sem nenhum acompanhamento após o HLES
    (tempo = 0: último contato no próprio HLES).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

COHORTS = ["DAP Pack", "Foundation"]
CORTE = pd.Timestamp("2024-12-31")
DIAS_ANO = 365.25

# Códigos conferidos no codebook (EOLS, eol_cause_of_death_primary / eol_illness_type).
CAUSA = {1: "velhice", 2: "doenca", 3: "trauma", 4: "toxina", 5: "anestesia_cirurgia", 6: "comportamento",
         7: "subita", 8: "pessoal", 98: "outra"}
TIPO_DOENCA = {1: "cancer", 2: "infeccao", 3: "outra_doenca", 99: "desconhecida"}


def main():
    ov = pd.read_csv(RAW / "DAP_2024_DogOverview_v1.1.csv", low_memory=False)
    own = pd.read_csv(RAW / "DAP_2024_HLES_dog_owner_v1.0.csv", usecols=["dog_id", "owner_id"])
    eol = pd.read_csv(RAW / "DAP_2024_EOLS_v1.0.csv", low_memory=False,
                      usecols=["dog_id", "eol_cause_of_death_primary", "eol_illness_type", "eol_euthan"])

    d = ov[ov.Cohort.isin(COHORTS)].merge(own, on="dog_id", how="left")
    assert d.dog_id.is_unique and d.owner_id.notna().all()
    for c in ["DAP_Pack_Date", "Survival_Date"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")

    vivo = d.Survival_Status.eq("Alive")
    assert d.loc[vivo, "Survival_Date"].max() <= CORTE, "há vivo apurado depois do corte: rever CORTE"

    morto = d.Survival_Status.eq("Dead")
    obito_pos_corte = morto & (d.Survival_Date > CORTE)
    fim = d.Survival_Date.where(~obito_pos_corte, CORTE)
    d["evento"] = (morto & ~obito_pos_corte).astype(int)
    d["tempo"] = (fim - d.DAP_Pack_Date).dt.days / DIAS_ANO
    d["obito_pos_corte"] = obito_pos_corte.astype(int)
    d["obito_data_estimada"] = (morto & d.Death_Source_Precision.ne("Actual Date")).astype(int)

    sem_data = d.Survival_Date.isna()
    sem_seguimento = ~sem_data & (d.tempo <= 0)
    d["incluido"] = (~sem_data & ~sem_seguimento).astype(int)
    d["motivo_exclusao"] = np.select([sem_data, sem_seguimento], ["sem_data_sobrevida", "sem_seguimento"], "")

    # Causa da morte (EOLS) — só descritivo / riscos competitivos em trabalho futuro.
    d = d.merge(eol, on="dog_id", how="left")
    d["causa_morte"] = d.eol_cause_of_death_primary.map(CAUSA)
    d["tipo_doenca_fatal"] = d.eol_illness_type.map(TIPO_DOENCA)
    d.loc[d.evento == 0, ["causa_morte", "tipo_doenca_fatal"]] = np.nan

    d["hles_ano"] = d.DAP_Pack_Date.dt.year
    d = d.rename(columns={"Estimated_Age_Years_at_HLES": "idade_hles"})
    cols = ["dog_id", "owner_id", "Cohort", "hles_ano", "idade_hles", "AgeDOB_Estimation_Method",
            "incluido", "motivo_exclusao", "tempo", "evento", "obito_pos_corte", "obito_data_estimada",
            "Survival_Source", "causa_morte", "tipo_doenca_fatal", "eol_euthan"]
    PROC.mkdir(parents=True, exist_ok=True)
    d[cols].to_csv(PROC / "populacao.csv", index=False)

    # --- Relatório (só agregados) ---
    inc = d[d.incluido == 1]
    print(f"Coortes {COHORTS}: {len(d)} cães | fora (outras coortes): {ov[~ov.Cohort.isin(COHORTS)].Cohort.value_counts().to_dict()}")
    print(f"  excluídos sem data de sobrevida: {sem_data.sum()} | sem acompanhamento após o HLES: {sem_seguimento.sum()}")
    print(f"Incluídos: {len(inc)} cães | {inc.owner_id.nunique()} tutores | óbitos até {CORTE.date()}: {inc.evento.sum()} "
          f"| óbitos após o corte (censurados): {inc.obito_pos_corte.sum()}")
    print(f"  cão-anos: {inc.tempo.sum():.0f} | taxa: {1000 * inc.evento.sum() / inc.tempo.sum():.1f} por 1.000 cão-anos")
    print(f"  seguimento (anos): mediana {inc.tempo.median():.2f}, máx {inc.tempo.max():.2f}")
    print(f"  óbitos com data estimada: {inc.obito_data_estimada[inc.evento == 1].sum()}")
    dias_obito = (inc.tempo[inc.evento == 1] * DIAS_ANO).round()
    print(f"  empates: {dias_obito.nunique()} tempos distintos (dias) para {len(dias_obito)} óbitos; "
          f"{dias_obito.duplicated(keep=False).mean():.1%} dos óbitos em tempo empatado")
    ev = inc[inc.evento == 1]
    print(f"  óbitos com EOLS: {ev.causa_morte.notna().sum()} ({ev.causa_morte.notna().mean():.0%})")
    print("  causa:", ev.causa_morte.value_counts().to_dict())
    print("  tipo de doença fatal:", ev.tipo_doenca_fatal.value_counts().to_dict())
    print("  tutores com >1 cão entre os incluídos:", int((inc.owner_id.value_counts() > 1).sum()))
    print("  por ano de HLES:", inc.groupby("hles_ano").agg(n=("dog_id", "size"), obitos=("evento", "sum")).T.to_dict())


if __name__ == "__main__":
    main()
