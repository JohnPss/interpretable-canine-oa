# Legado — desenho de osteoartrite (OA), abandonado em 29/09/2026

O projeto pivotou de "risco de OA incidente" para **longevidade (morte por qualquer causa)**.
Motivo e números em `../../Etapas/analise_temas.md`; documentos da fase OA em `../../legado_oa/`.

Aqui ficam só os scripts que eram específicos de OA, para referência (não fazem parte do pipeline atual):
`01_amostra.py`, `02_incidencia.py`, `02b_km_faixa_etaria.py`, `03_features.py`, `features_config.py`
(versão OA), `20_tabelas_artigo.py` e `figuras/` (Kaplan-Meier de OA). As saídas deles foram movidas para
`data/processed/legado_oa/` (fora do git). As branches `oa-00-base`, `oa-01-cox` e `oa-02-gbm` guardam a
modelagem de OA (split, métricas em tempo discreto, Cox linear, XGBoost Cox).

Os caminhos dentro destes scripts apontam para a raiz antiga (`data/raw`, `data/processed`): para rodar de
novo, copie-os de volta para a raiz ou ajuste `ROOT`.
