"""Grupos e tipos das features (Fase 3, desenho de longevidade). Importado pela modelagem.

Nenhum parâmetro ajustado aqui: encoding/imputação/escala são feitos no pipeline, só com dados de treino.
Versão do desenho de OA: legado_oa/features_config.py.
"""

# --- Grupos (metodologia_projeto.md, Fase 3) ---
# Estáveis: não mudam (ou quase) com o estado de saúde.
ESTAVEIS = ["idade", "sexo", "castracao", "porte", "raca_pura", "regiao"]

# Estilo de vida / estado atual: atividade, peso e dieta. Podem refletir doença já presente na entrada
# (causalidade reversa: cão doente se mexe menos, perde peso, come menos).
ESTILO_VIDA = [
    "peso_kg", "atividade_nivel", "minutos_ativos_dia", "atividade_intensidade",
    "passeio_guia_freq", "passeio_solto_freq", "brincadeira_freq",
    "sobrepeso_ever", "subpeso_ever", "peso_mudanca_12m",
    "dieta_principal", "alimentacao", "dieta_consistencia", "apetite",
]

# Histórico de saúde na entrada (HLES): saúde geral avaliada pelo tutor + doença alguma vez diagnosticada
# em cada sistema (pergunta-filtro por sistema do HLES). Entra no modelo principal para que o efeito do
# estilo de vida seja estimado "entre cães com a mesma carga de doença"; a ablação sem este bloco mede
# quanto do desempenho vem de "cão já doente".
SISTEMAS = {  # nome da feature: sufixo da variável hs_health_conditions_<sufixo> no HLES
    "doenca_cancer": "cancer", "doenca_cardiaca": "cardiac", "doenca_respiratoria": "respiratory",
    "doenca_gastrointestinal": "gastrointestinal", "doenca_figado_pancreas": "liver",
    "doenca_renal_urinaria": "kidney", "doenca_ortopedica": "orthopedic", "doenca_neurologica": "neurological",
    "doenca_endocrina": "endocrine", "doenca_hematologica": "hematologic", "doenca_imunomediada": "immune",
}
SAUDE = ["saude_geral", "n_sistemas_doenca"] + list(SISTEMAS)

# Só em análise de sensibilidade (troca de versão do bloco de dieta em 2022).
SENSIBILIDADE = ["dieta_versao"]

MODELO_PRINCIPAL = ESTAVEIS + ESTILO_VIDA + SAUDE
SEM_SAUDE = ESTAVEIS + ESTILO_VIDA          # ablação principal

# --- Tipos ---
NUMERICAS = ["idade", "peso_kg", "minutos_ativos_dia", "n_sistemas_doenca"]
# Ordinais já codificadas como inteiros crescentes (0 = menor/nunca/melhor).
ORDINAIS = ["porte", "atividade_nivel", "atividade_intensidade",
            "passeio_guia_freq", "passeio_solto_freq", "brincadeira_freq", "saude_geral"]
BINARIAS = ["sexo", "raca_pura"] + list(SISTEMAS)
CATEGORICAS = ["castracao", "regiao", "sobrepeso_ever", "subpeso_ever", "peso_mudanca_12m",
               "dieta_principal", "alimentacao", "dieta_consistencia", "apetite", "dieta_versao"]

# Desfecho e colunas auxiliares (não são features).
ALVO = ["tempo", "evento"]
AUXILIARES = ["dog_id", "owner_id", "hles_ano", "Cohort", "AgeDOB_Estimation_Method",
              "obito_pos_corte", "obito_data_estimada", "causa_morte", "tipo_doenca_fatal"]
