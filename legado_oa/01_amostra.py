"""Fase 1 — definição da amostra e do status de OA no HLES (baseline).

Entrada:  data/raw/ (DogOverview, HLES_dog_owner, HLES_health_condition)
Saída:    data/processed/amostra.csv — um cão por linha, com flags usadas nas fases seguintes.

Decisões documentadas em metodologia_projeto.md (Fase 1).
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "processed"

OA = 1113  # hs_condition "Osteoarthritis" (bloco 11, Bone/Orthopedic)

# Cohorts incluídos. Foundation entra porque a prevalência de OA por faixa etária é
# igual à do DAP Pack; a prevalência bruta menor dele é só efeito de idade.
# Precision fica fora (escopo exclui exames laboratoriais).
COHORTS = ["DAP Pack", "Foundation"]

# Texto livre que indica artrite fora do código 1113 (em geral no 1198 "outro").
# Não vira positivo automaticamente — só flag, para nunca virar negativo silencioso.
REGEX_ARTRITE = r"arthrit|\bDJD\b|degenerative joint"


def corrige_ano(s: pd.Series) -> pd.Series:
    """Ano de diagnóstico digitado com 2 dígitos (ex.: 18 → 2018)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 100, s + 2000)


def main():
    ov = pd.read_csv(RAW / "DAP_2024_DogOverview_v1.1.csv", low_memory=False)
    own = pd.read_csv(
        RAW / "DAP_2024_HLES_dog_owner_v1.0.csv",
        usecols=["dog_id", "owner_id", "st_hles_completion_date"],
    )
    hc = pd.read_csv(RAW / "DAP_2024_HLES_health_condition_v1.1.csv", low_memory=False)

    df = ov[ov.Cohort.isin(COHORTS)].merge(own, on="dog_id", how="left")
    df["hles_data"] = pd.to_datetime(df.st_hles_completion_date)
    df["hles_ano"] = df.hles_data.dt.year
    df["dob"] = pd.to_datetime(df.Estimated_DOB)

    # --- OA no HLES (prevalente) ---
    # Cada cão tem no máximo uma linha 1113. Cão sem linha do bloco 11 respondeu
    # hs_health_conditions_orthopedic == 0 (conferido), então ausência = "não", não faltante.
    oa = hc.loc[hc.hs_condition == OA, ["dog_id", "hs_diagnosis_year"]]
    oa = oa.assign(oa_hles_dx_ano=corrige_ano(oa.hs_diagnosis_year)).drop(columns="hs_diagnosis_year")
    df = df.merge(oa, on="dog_id", how="left")
    df["oa_hles"] = df.oa_hles_dx_ano.notna()

    # Idade no diagnóstico só por ano: o mês tem pico artificial em janeiro (provável
    # default de "não lembro"), então não é usado. Resolução resultante: ±1 ano.
    df["oa_hles_idade_dx"] = df.oa_hles_dx_ano - df.dob.dt.year
    # Idade negativa = erro de data. O cão sai de análises que dependem da idade no
    # diagnóstico — nunca é reclassificado como negativo.
    df["flag_idade_dx_neg"] = df.oa_hles_idade_dx < 0
    # < 1 ano é implausível para OA, mas não impossível: só marca para revisão.
    df["flag_idade_dx_lt1"] = df.oa_hles_idade_dx.between(0, 0.99)

    # --- Texto livre de artrite sem código 1113 ---
    txt = hc.hs_condition_other_description.astype(str).str.contains(REGEX_ARTRITE, case=False, regex=True)
    df["flag_texto_artrite_hles"] = df.dog_id.isin(hc.loc[txt, "dog_id"]) & ~df.oa_hles

    OUT.mkdir(parents=True, exist_ok=True)
    cols = [
        "dog_id", "owner_id", "Cohort", "hles_data", "hles_ano", "Estimated_DOB",
        "Estimated_Age_Years_at_HLES", "AgeDOB_Estimation_Method", "Survival_Status",
        "oa_hles", "oa_hles_dx_ano", "oa_hles_idade_dx",
        "flag_idade_dx_neg", "flag_idade_dx_lt1", "flag_texto_artrite_hles",
    ] + [f"AFUS_Year{i}_Complete" for i in range(1, 6)]
    df[cols].to_csv(OUT / "amostra.csv", index=False)

    print(f"Cães ({' + '.join(COHORTS)}): {len(df)}")
    print(df.groupby("Cohort").oa_hles.agg(n="size", oa="sum", prev="mean").round(3))
    print(f"OA no HLES: {df.oa_hles.sum()} ({df.oa_hles.mean():.1%})")
    print(f"  ano dx corrigido de 2 dígitos: {(oa.dog_id.isin(df.dog_id) & (pd.to_numeric(hc.loc[oa.index, 'hs_diagnosis_year']) < 100)).sum()}")
    print(f"  idade dx < 0: {df.flag_idade_dx_neg.sum()} | idade dx < 1: {df.flag_idade_dx_lt1.sum()}")
    print(f"Texto livre artrite sem 1113: {df.flag_texto_artrite_hles.sum()}")
    print(f"Salvo: {OUT / 'amostra.csv'}")


if __name__ == "__main__":
    main()
