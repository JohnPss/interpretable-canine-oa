"""Fase 2 — harmonização do bloco de dieta entre as duas versões do HLES.

O HLES trocou o bloco de dieta em 2022: a versão antiga usa variáveis `df_*`
(39.268 cães, HLES 2019–2022) e a nova usa `dt_*` (9.809 cães, HLES 2022–2024).
Cada cão tem uma versão só. Sem harmonizar, dieta fica 100% faltante numa das
metades da amostra.

Critério de "mapeamento limpo": pergunta equivalente no codebook E proporções
parecidas entre as versões dentro da mesma faixa etária (a versão nova tem cães
mais jovens, então comparar sem estratificar confunde versão com idade).

Entrada:  data/raw/DAP_2024_HLES_dog_owner_v1.0.csv, data/processed/populacao.csv (01_populacao.py)
Saída:    data/processed/dieta.csv — variáveis harmonizadas (categorias em texto;
          o encoding é feito no pipeline da Fase 3).

Decisões documentadas em metodologia_projeto.md (Fase 2). A harmonização foi feita e validada
no desenho de OA (mesmos cães, mesmas perguntas) e vale igual aqui: não depende do desfecho.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

JUNTA = ["glucosamine", "chondroitin", "other_joint"]
SIM_NAO = {1: "sim", 0: "nao", 99: "nao_sabe"}


def main():
    own = pd.read_csv(RAW / "DAP_2024_HLES_dog_owner_v1.0.csv", low_memory=False,
                      usecols=lambda c: c == "dog_id" or c.startswith(("df_", "dt_")))
    am = pd.read_csv(PROC / "populacao.csv", usecols=["dog_id", "idade_hles"]) \
           .rename(columns={"idade_hles": "Estimated_Age_Years_at_HLES"})
    d = am.merge(own, on="dog_id", how="left")

    # Versão do bloco: cada cão respondeu exatamente uma.
    antiga = d.df_ever_overweight.notna() | d.df_primary_diet_component.notna()
    d["dieta_versao"] = np.where(antiga, "df", np.where(d.dt_primary_diet_type.notna(), "dt", pd.NA))
    v_df = d.dieta_versao == "df"

    out = d[["dog_id", "dieta_versao"]].copy()

    # --- Sobrepeso / subpeso alguma vez: mesma pergunta, mesmos códigos (1/0/99) ---
    out["sobrepeso_ever"] = np.where(v_df, d.df_ever_overweight, d.dt_overrweight_hles)
    out["subpeso_ever"] = np.where(v_df, d.df_ever_underweight, d.dt_underweight_hles)
    for c in ["sobrepeso_ever", "subpeso_ever"]:
        out[c] = out[c].map(SIM_NAO)

    # --- Mudança de peso em 12 meses ---
    # ATENÇÃO à codificação invertida: em df, 0 = "ficou igual"; em dt, 1 = "ficou igual"
    # e o sentido da mudança vem numa segunda pergunta (dt_weight_change_how).
    wc_df = d.df_weight_change_last_year.map({0: "estavel", 1: "perdeu", 2: "ganhou", 3: "filhote", 99: "nao_sabe"})
    wc_dt = pd.Series(np.select(
        [d.dt_weight_change == 1,
         (d.dt_weight_change == 0) & (d.dt_weight_change_how == 1),
         (d.dt_weight_change == 0) & (d.dt_weight_change_how == 2),
         d.dt_weight_change == 2, d.dt_weight_change == 99],
        ["estavel", "perdeu", "ganhou", "filhote", "nao_sabe"], default=None), index=d.index)
    out["peso_mudanca_12m"] = np.where(v_df, wc_df, wc_dt)

    # --- Componente principal da dieta (colapsado em 4 categorias) ---
    # df tem 8 opções numa pergunta; dt separa "comercial vs caseira" e depois o tipo
    # de comercial. Detalhes que não existem nas duas (cru vs cozido na dieta caseira)
    # são perdidos. Kit comercial para preparo caseiro (dt 7/8) → caseira, que é como
    # o tutor provavelmente responderia na versão antiga.
    dp_df = d.df_primary_diet_component.map(
        {1: "comercial_seca", 2: "comercial_outra", 3: "comercial_outra", 4: "comercial_outra",
         5: "comercial_outra", 6: "caseira", 7: "caseira", 98: "outra"})
    dp_dt = pd.Series(np.select(
        [d.dt_primary_diet_type == 2,
         (d.dt_primary_diet_type == 1) & (d.dt_comm1_dietcomp == 1),
         (d.dt_primary_diet_type == 1) & d.dt_comm1_dietcomp.between(2, 6),
         (d.dt_primary_diet_type == 1) & d.dt_comm1_dietcomp.isin([7, 8])],
        ["caseira", "comercial_seca", "comercial_outra", "caseira"], default=None), index=d.index)
    out["dieta_principal"] = np.where(v_df, dp_df, dp_dt)

    # --- Padrão de alimentação: refeições/dia ou livre demanda ---
    al_df = d.df_feedings_per_day.map({1: "1x", 2: "2x", 3: "3+", 4: "livre"})
    al_dt = pd.Series(np.select(
        [d.dt_feeding_pattern == 2,
         (d.dt_feeding_pattern == 1) & (d.dt_pattern_feed_timesday == 1),
         (d.dt_feeding_pattern == 1) & (d.dt_pattern_feed_timesday == 2),
         (d.dt_feeding_pattern == 1) & (d.dt_pattern_feed_timesday >= 3)],
        ["livre", "1x", "2x", "3+"], default=None), index=d.index)
    out["alimentacao"] = np.where(v_df, al_df, al_dt)

    # --- Consistência da dieta e apetite: mesma pergunta e mesmas opções ---
    # (df tem "Other" = 98 na consistência, 35 cães → faltante.)
    cons = {1: "muito", 2: "pouco", 3: "nada"}
    out["dieta_consistencia"] = np.where(v_df, d.df_diet_consistency.map(cons), d.dt_pattern_consistency.map(cons))
    ap = {1: "fraco", 2: "normal", 3: "voraz"}
    out["apetite"] = np.where(v_df, d.df_appetite.map(ap), d.dt_pattern_appetite_overall.map(ap))

    # --- Suplemento articular (glucosamina, condroitina, outro) ---
    # df distingue "suplemento separado" de "a ração contém"; dt só pergunta o tipo.
    # Dentro da faixa etária, dt bate com df "separado OU na ração" (ex.: 7–10 anos:
    # 42–45% nas duas), e não com "só separado". Por isso a definição é a ampla.
    # Qualquer frequência (diária ou menos frequente) conta.
    cols_df = [f"df_{t}_supplements_{s}" for t in ["daily", "infrequent"] for s in JUNTA]
    junta_df = d[cols_df].isin([1, 2]).any(axis=1)
    junta_dt = d[["dt_supplements_type_glucosamine", "dt_supplements_type_chondroitin",
                  "dt_supplements_type_otherjoint"]].eq(True).any(axis=1)
    out["suplemento_articular"] = pd.Series(np.where(v_df, junta_df, junta_dt), index=d.index).astype("boolean")
    out.loc[out.dieta_versao.isna(), "suplemento_articular"] = pd.NA

    # NÃO harmonizados (mapeamento não limpo → trabalhos futuros):
    # - petiscos: df pergunta frequência geral; dt pergunta frequência por tipo de petisco.
    # - suplementos em geral: df inclui "ração que contém suplemento", dt não.
    # - composição detalhada (proteína, carboidrato, marca, aditivos): só existe em dt.

    out.to_csv(PROC / "dieta.csv", index=False)

    # --- Validação: proporções por versão dentro da faixa etária ---
    chk = out.merge(am, on="dog_id")
    chk["faixa"] = pd.cut(chk.Estimated_Age_Years_at_HLES, [0, 3, 7, 10, 30], include_lowest=True)
    val = chk.groupby(["faixa", "dieta_versao"], observed=True).agg(
        n=("dog_id", "size"),
        sobrepeso=("sobrepeso_ever", lambda s: (s == "sim").mean()),
        perdeu_peso=("peso_mudanca_12m", lambda s: (s == "perdeu").mean()),
        ganhou_peso=("peso_mudanca_12m", lambda s: (s == "ganhou").mean()),
        comercial_seca=("dieta_principal", lambda s: (s == "comercial_seca").mean()),
        caseira=("dieta_principal", lambda s: (s == "caseira").mean()),
        livre=("alimentacao", lambda s: (s == "livre").mean()),
        apetite_voraz=("apetite", lambda s: (s == "voraz").mean()),
        sup_articular=("suplemento_articular", "mean"))
    print(out.dieta_versao.value_counts(dropna=False).to_dict())
    print("Faltantes por variável harmonizada:")
    print(out.drop(columns=["dog_id", "dieta_versao"]).isna().mean().round(3).to_dict())
    print("\nProporções por versão dentro da faixa etária (devem ser parecidas):")
    print(val.round(3).to_string())
    print(f"\nSalvo: {PROC / 'dieta.csv'}")


if __name__ == "__main__":
    main()
