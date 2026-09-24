"""Fase 3 — engenharia de features para a população em risco do desenho (C).

Só limpeza e recodificação determinística (sem parâmetros ajustados nos dados):
encoding one-hot, imputação e escala ficam no pipeline da Fase 4, só no treino.

Entrada:  data/processed/incidencia.csv (população + desfecho, 02_incidencia.py)
          data/processed/dieta.csv (dieta harmonizada, 03_dieta.py)
          data/raw/ (HLES_dog_owner, DogOverview, HLES_health_condition)
Saída:    data/processed/features.csv — um cão em risco por linha: auxiliares,
          desfechos e todas as features de features_config.py.

Decisões documentadas em metodologia_projeto.md (Fase 3).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import features_config as fc

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

DISPLASIA_QUADRIL, RUPTURA_CRUZADO = 1110, 1105
LB_KG = 0.45359237


def main():
    base = pd.read_csv(PROC / "incidencia.csv")
    own = pd.read_csv(RAW / "DAP_2024_HLES_dog_owner_v1.0.csv", low_memory=False, usecols=[
        "dog_id", "dd_sex", "dd_spayed_or_neutered", "dd_spay_or_neuter_age", "dd_weight_lbs",
        "pa_activity_level", "pa_avg_daily_active_hours", "pa_avg_daily_active_minutes",
        "pa_avg_activity_intensity", "pa_on_leash_off_leash_walk", "pa_on_leash_walk_frequency",
        "pa_off_leash_walk_frequency", "pa_physical_games_frequency",
        "oc_primary_residence_census_division"])
    ov = pd.read_csv(RAW / "DAP_2024_DogOverview_v1.1.csv", low_memory=False,
                     usecols=["dog_id", "Breed_Status", "Breed_Size_Class_at_HLES"])
    hc = pd.read_csv(RAW / "DAP_2024_HLES_health_condition_v1.1.csv", low_memory=False,
                     usecols=["dog_id", "hs_condition"])
    dieta = pd.read_csv(PROC / "dieta.csv")

    d = base.merge(own, on="dog_id", how="left").merge(ov, on="dog_id", how="left") \
            .merge(dieta, on="dog_id", how="left")
    f = pd.DataFrame(index=d.index)

    # ================= ESTÁVEIS =================
    f["idade"] = d.Estimated_Age_Years_at_HLES  # dd_age tem 83% de faltante; esta vem do DogOverview
    f["sexo"] = (d.dd_sex == 2).astype(int)     # 1 = fêmea (codebook: 1 Male, 2 Female)

    # Castração + idade na castração numa variável só (idade só existe para castrados).
    # Códigos do codebook: 1 <6m | 2 6–12m | 3 12–18m | 4 18–24m | 5 2–4a | 6 4–6a | 7 6–8a | 8 >8a | 99 desconhecida
    idade_castr = d.dd_spay_or_neuter_age.map({1: "<6m", 2: "6-12m", 3: "12-18m", 4: "18-24m",
                                               5: "2-4a", 6: "4-6a", 7: "6-8a", 8: ">8a",
                                               99: "castrado_idade_desconhecida"})
    f["castracao"] = np.where(d.dd_spayed_or_neutered, idade_castr, "intacto")

    # Porte: Breed_Size_Class_at_HLES é derivada do PESO informado no HLES (codebook), não
    # da raça. Só o porte (5 níveis ordinais) entra; a parte "AKC vs misto" do rótulo vem
    # de raca_pura. Tratada como estável: o porte muda pouco com ganho de peso, mas é
    # "quase estável" — obesidade pode mudar a classe (ver Riscos).
    porte = {"Toy and Small": 0, "Medium": 1, "Standard": 2, "Large": 3, "Giant": 4}
    f["porte"] = d.Breed_Size_Class_at_HLES.map(
        lambda s: next((v for k, v in porte.items() if isinstance(s, str) and s.startswith(k)), np.nan))
    f["raca_pura"] = (d.Breed_Status == "Purebred").astype(int)
    f["regiao"] = "div_" + d.oc_primary_residence_census_division.astype("Int64").astype(str)

    # ================= ESTADO ATUAL =================
    # Peso atual. < 1 lb é implausível (erro de digitação) → faltante.
    peso = d.dd_weight_lbs.where(d.dd_weight_lbs >= 1)
    f["peso_kg"] = (peso * LB_KG).round(2)

    f["atividade_nivel"] = d.pa_activity_level - 1            # 0 pouco ativo | 1 moderado | 2 muito ativo
    # Horas (0..8+) e minutos (0..50, em blocos de 10) viram minutos/dia. "8+" horas = 480 (piso).
    f["minutos_ativos_dia"] = d.pa_avg_daily_active_hours * 60 + d.pa_avg_daily_active_minutes * 10
    f["atividade_intensidade"] = d.pa_avg_activity_intensity - 1  # 0 caminhada | 1 trote | 2 corrida

    # Passeios: a pergunta filtro pa_on_leash_off_leash_walk (0 não passeia | 1 só com guia |
    # 2 só solto | 3 ambos) explica 100% dos faltantes das frequências. Quem não faz aquele
    # tipo de passeio tem frequência 0 — não é faltante, é "nunca".
    # Frequência: 1 <1x/mês ... 6 >1x/dia → mantida como 1..6, com 0 = nunca.
    gate = d.pa_on_leash_off_leash_walk
    f["passeio_guia_freq"] = np.where(gate.isin([0, 2]), 0, d.pa_on_leash_walk_frequency)
    f["passeio_solto_freq"] = np.where(gate.isin([0, 1]), 0, d.pa_off_leash_walk_frequency)
    # Brincadeiras: codebook 1 diário ... 5 nunca → invertido para 0 nunca ... 4 diário.
    f["brincadeira_freq"] = 5 - d.pa_physical_games_frequency

    for c in ["sobrepeso_ever", "subpeso_ever", "peso_mudanca_12m", "dieta_principal",
              "alimentacao", "dieta_consistencia", "apetite"]:
        f[c] = d[c]  # já harmonizadas em 03_dieta.py ("nao_sabe" é categoria própria)

    # ================= HISTÓRICO ORTOPÉDICO (HLES) =================
    f["displasia_quadril"] = d.dog_id.isin(hc.loc[hc.hs_condition == DISPLASIA_QUADRIL, "dog_id"]).astype(int)
    f["ruptura_cruzado"] = d.dog_id.isin(hc.loc[hc.hs_condition == RUPTURA_CRUZADO, "dog_id"]).astype(int)

    # ================= ABLAÇÃO / SENSIBILIDADE =================
    f["suplemento_articular"] = d.suplemento_articular.map({True: 1, False: 0, "True": 1, "False": 0})
    f["dieta_versao"] = d.dieta_versao

    # --- Saída ---
    aux = d[fc.AUXILIARES + fc.ALVO_SOBREVIVENCIA + [fc.ALVO_JANELA]]
    todas = fc.MODELO_PRINCIPAL + fc.ABLACAO_MARCADOR + fc.SENSIBILIDADE
    assert set(todas) == set(f.columns), set(todas) ^ set(f.columns)
    tipadas = fc.NUMERICAS + fc.ORDINAIS + fc.BINARIAS + fc.CATEGORICAS
    assert sorted(tipadas) == sorted(todas), set(tipadas) ^ set(todas)
    out = pd.concat([aux, f[todas]], axis=1)
    out.to_csv(PROC / "features.csv", index=False)

    # --- Relatório ---
    print(f"Cães: {len(out)} | eventos: {out.evento.sum()} | features: {len(todas)} "
          f"(principal: {len(fc.MODELO_PRINCIPAL)})")
    falt = f[todas].isna().mean()
    print("Faltantes > 0:", falt[falt > 0].round(4).to_dict() or "nenhum")
    print("\nDistribuições (numéricas):")
    print(f[fc.NUMERICAS].describe().round(1).T[["mean", "min", "50%", "max"]])
    print("\nOrdinais/binárias (valor: n):")
    for c in fc.ORDINAIS + fc.BINARIAS:
        print(f"  {c}: {f[c].value_counts(dropna=False).sort_index().to_dict()}")
    print("\nCategóricas (n níveis):", {c: f[c].nunique() for c in fc.CATEGORICAS})
    print("\nRisco bruto de OA incidente por feature-chave (evento / n):")
    for c in ["porte", "castracao", "displasia_quadril", "ruptura_cruzado", "suplemento_articular"]:
        print(f"  {c}:", out.groupby(c).evento.agg(lambda s: f"{s.mean():.1%} (n={len(s)})").to_dict())
    print(f"\nSalvo: {PROC / 'features.csv'}")


if __name__ == "__main__":
    main()
