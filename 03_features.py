"""Fase 3 — engenharia de features para a população de longevidade (cães incluídos no 01_populacao).

Só limpeza e recodificação determinística (sem parâmetros ajustados nos dados): encoding one-hot,
imputação e escala ficam no pipeline de modelagem, só no treino.

Entrada:  data/processed/populacao.csv (01_populacao.py), data/processed/dieta.csv (02_dieta.py),
          data/raw/ (HLES_dog_owner, DogOverview)
Saída:    data/processed/features.csv — um cão incluído por linha: auxiliares, desfecho e as features
          de features_config.py.

As features estáveis e de estilo de vida têm a mesma derivação do desenho de OA
(legado_oa/03_features.py). Novo: bloco de histórico de saúde (SAUDE).
Decisões documentadas em metodologia_projeto.md (Fase 3).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import features_config as fc

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
LB_KG = 0.45359237

# Sistemas que entram na contagem de carga de doença: os 16 sistemas de órgãos da pergunta-filtro do HLES.
# Ficam fora infecção/parasitas, ingestão de toxina e trauma (eventos agudos/exposições, não doença de órgão)
# e "outro" (codificação diferente e 95% faltante).
SISTEMAS_CONTAGEM = ["eye", "ear", "oral", "skin", "cardiac", "respiratory", "gastrointestinal", "liver",
                     "kidney", "reproductive", "orthopedic", "neurological", "endocrine", "hematologic",
                     "immune", "cancer"]


def main():
    base = pd.read_csv(PROC / "populacao.csv")
    base = base[base.incluido == 1].drop(columns=["incluido", "motivo_exclusao"])
    gates = [f"hs_health_conditions_{s}" for s in SISTEMAS_CONTAGEM]
    own = pd.read_csv(RAW / "DAP_2024_HLES_dog_owner_v1.0.csv", low_memory=False, usecols=[
        "dog_id", "dd_sex", "dd_spayed_or_neutered", "dd_spay_or_neuter_age", "dd_weight_lbs",
        "pa_activity_level", "pa_avg_daily_active_hours", "pa_avg_daily_active_minutes",
        "pa_avg_activity_intensity", "pa_on_leash_off_leash_walk", "pa_on_leash_walk_frequency",
        "pa_off_leash_walk_frequency", "pa_physical_games_frequency",
        "oc_primary_residence_census_division", "hs_general_health"] + gates)
    ov = pd.read_csv(RAW / "DAP_2024_DogOverview_v1.1.csv", low_memory=False,
                     usecols=["dog_id", "Breed_Status", "Breed_Size_Class_at_HLES"])
    dieta = pd.read_csv(PROC / "dieta.csv")

    d = base.merge(own, on="dog_id", how="left").merge(ov, on="dog_id", how="left") \
            .merge(dieta, on="dog_id", how="left")
    assert len(d) == len(base)
    f = pd.DataFrame(index=d.index)

    # ================= ESTÁVEIS =================
    f["idade"] = d.idade_hles                      # DogOverview; dd_age tem 83% de faltante
    f["sexo"] = (d.dd_sex == 2).astype(int)        # 1 = fêmea (codebook: 1 Male, 2 Female)
    # Castração + idade na castração. Códigos: 1 <6m | 2 6–12m | 3 12–18m | 4 18–24m | 5 2–4a | 6 4–6a
    # | 7 6–8a | 8 >8a | 99 desconhecida.
    idade_castr = d.dd_spay_or_neuter_age.map({1: "<6m", 2: "6-12m", 3: "12-18m", 4: "18-24m",
                                               5: "2-4a", 6: "4-6a", 7: "6-8a", 8: ">8a",
                                               99: "castrado_idade_desconhecida"})
    f["castracao"] = np.where(d.dd_spayed_or_neutered, idade_castr, "intacto")
    # Porte: Breed_Size_Class_at_HLES é derivada do PESO informado no HLES (codebook), não da raça.
    porte = {"Toy and Small": 0, "Medium": 1, "Standard": 2, "Large": 3, "Giant": 4}
    f["porte"] = d.Breed_Size_Class_at_HLES.map(
        lambda s: next((v for k, v in porte.items() if isinstance(s, str) and s.startswith(k)), np.nan))
    f["raca_pura"] = (d.Breed_Status == "Purebred").astype(int)
    f["regiao"] = "div_" + d.oc_primary_residence_census_division.astype("Int64").astype(str)

    # ================= ESTILO DE VIDA =================
    peso = d.dd_weight_lbs.where(d.dd_weight_lbs >= 1)   # < 1 lb é implausível → faltante
    f["peso_kg"] = (peso * LB_KG).round(2)
    f["atividade_nivel"] = d.pa_activity_level - 1            # 0 não ativo | 1 moderado | 2 muito ativo
    f["minutos_ativos_dia"] = d.pa_avg_daily_active_hours * 60 + d.pa_avg_daily_active_minutes * 10
    f["atividade_intensidade"] = d.pa_avg_activity_intensity - 1  # 0 caminhada | 1 trote | 2 corrida
    # Pergunta-filtro pa_on_leash_off_leash_walk (0 não passeia | 1 só com guia | 2 só solto | 3 ambos)
    # explica 100% dos faltantes das frequências: quem não faz aquele passeio tem frequência 0 ("nunca").
    gate = d.pa_on_leash_off_leash_walk
    f["passeio_guia_freq"] = np.where(gate.isin([0, 2]), 0, d.pa_on_leash_walk_frequency)
    f["passeio_solto_freq"] = np.where(gate.isin([0, 1]), 0, d.pa_off_leash_walk_frequency)
    f["brincadeira_freq"] = 5 - d.pa_physical_games_frequency     # codebook 1 diário..5 nunca → 0 nunca..4 diário
    for c in ["sobrepeso_ever", "subpeso_ever", "peso_mudanca_12m", "dieta_principal",
              "alimentacao", "dieta_consistencia", "apetite", "dieta_versao"]:
        f[c] = d[c]                                               # harmonizadas em 02_dieta.py

    # ================= HISTÓRICO DE SAÚDE (HLES) =================
    # hs_general_health: 1 excelente .. 6 muito ruim → 0..5 (maior = pior).
    f["saude_geral"] = d.hs_general_health - 1
    # Pergunta-filtro por sistema: 0 nenhum | 1 só congênito | 2 só não congênito | 3 ambos → alguma vez = >0.
    for nome, suf in fc.SISTEMAS.items():
        f[nome] = (d[f"hs_health_conditions_{suf}"] > 0).astype(int)
    f["n_sistemas_doenca"] = (d[gates] > 0).sum(axis=1)

    # --- Saída ---
    todas = fc.MODELO_PRINCIPAL + fc.SENSIBILIDADE
    assert set(todas) == set(f.columns), set(todas) ^ set(f.columns)
    tipadas = fc.NUMERICAS + fc.ORDINAIS + fc.BINARIAS + fc.CATEGORICAS
    assert sorted(tipadas) == sorted(todas), set(tipadas) ^ set(todas)
    aux = d[[c for c in fc.AUXILIARES if c in d.columns] + fc.ALVO]
    out = pd.concat([aux, f[todas]], axis=1)
    out.to_csv(PROC / "features.csv", index=False)

    # --- Relatório (agregados) ---
    print(f"Cães: {len(out)} | óbitos: {out.evento.sum()} | features: {len(fc.MODELO_PRINCIPAL)} no modelo principal "
          f"({len(fc.ESTAVEIS)} estáveis, {len(fc.ESTILO_VIDA)} estilo de vida, {len(fc.SAUDE)} saúde)")
    falt = f[todas].isna().mean()
    print("Faltantes > 0:", falt[falt > 0].round(4).to_dict() or "nenhum")
    print(f[fc.NUMERICAS].describe().round(1).T[["mean", "min", "50%", "max"]])
    tx = lambda g: out.groupby(g).apply(lambda s: f"{1000 * s.evento.sum() / s.tempo.sum():.0f}/1000 (n={len(s)})").to_dict()
    print("\nTaxa bruta de óbito (por 1.000 cão-anos; confundida por idade, só descritivo):")
    for c in ["porte", "saude_geral", "atividade_nivel", "doenca_cancer", "castracao"]:
        print(f"  {c}:", tx(c))
    print(f"\nSalvo: {PROC / 'features.csv'}")


if __name__ == "__main__":
    main()
