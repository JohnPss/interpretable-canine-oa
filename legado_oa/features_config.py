"""Grupos e tipos das features (Fase 3). Importado pelos scripts de modelagem (Fase 4).

Nenhum parâmetro ajustado aqui: encoding/imputação/escala são feitos no pipeline,
só com dados de treino.
"""

# --- Grupos (metodologia_projeto.md, Fase 3) ---
# Estáveis: não mudam (ou quase) com o desenvolvimento de OA.
ESTAVEIS = ["idade", "sexo", "castracao", "porte", "raca_pura", "regiao"]

# Estado atual: podem refletir OA subclínica no baseline (causalidade reversa residual).
ESTADO_ATUAL = [
    "peso_kg", "atividade_nivel", "minutos_ativos_dia", "atividade_intensidade",
    "passeio_guia_freq", "passeio_solto_freq", "brincadeira_freq",
    "sobrepeso_ever", "subpeso_ever", "peso_mudanca_12m",
    "dieta_principal", "alimentacao", "dieta_consistencia", "apetite",
]

# Histórico ortopédico no HLES (anterior ao desfecho em (C)).
HISTORICO_ORTO = ["displasia_quadril", "ruptura_cruzado"]

# Só em ablação: confundimento por indicação (tutor dá suplemento porque percebe problema).
ABLACAO_MARCADOR = ["suplemento_articular"]

# Só em análise de sensibilidade (troca de versão do bloco de dieta).
SENSIBILIDADE = ["dieta_versao"]

MODELO_PRINCIPAL = ESTAVEIS + ESTADO_ATUAL + HISTORICO_ORTO

# --- Tipos ---
NUMERICAS = ["idade", "peso_kg", "minutos_ativos_dia"]
# Ordinais já codificadas como inteiros crescentes (0 = menor/nunca).
ORDINAIS = ["porte", "atividade_nivel", "atividade_intensidade",
            "passeio_guia_freq", "passeio_solto_freq", "brincadeira_freq"]
BINARIAS = ["sexo", "raca_pura", "displasia_quadril", "ruptura_cruzado", "suplemento_articular"]
CATEGORICAS = ["castracao", "regiao", "sobrepeso_ever", "subpeso_ever", "peso_mudanca_12m",
               "dieta_principal", "alimentacao", "dieta_consistencia", "apetite", "dieta_versao"]

# Desfechos e colunas auxiliares (não são features).
ALVO_SOBREVIVENCIA = ["tempo", "evento"]
ALVO_JANELA = "oa_2anos"
AUXILIARES = ["dog_id", "owner_id", "hles_ano", "Cohort", "AgeDOB_Estimation_Method"]
