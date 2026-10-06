# interpretable-canine-oa → longevidade canina (DAP)

> O nome do repositório ficou do tema inicial (osteoartrite). Desde 29/09/2026 o projeto é
> **predição interpretável da longevidade (morte por qualquer causa)** com dados do Dog Aging Project.
> O código de OA está em `legado_oa/` e nas branches `oa-*`.

Dados em `data/raw/` (fora do git — DUA do DAP). Venv: `/home/johnpss/Desktop/Dogs/.venv` (Python 3.12,
versões em `requirements.txt`; **xgboost < 3.1** por causa do shap 0.49.1).

## Pipeline (rodar em ordem)
| Script | Faz | Saída |
|---|---|---|
| `01_populacao.py` | coortes, tempo desde o HLES, óbito, corte em 2024-12-31, exclusões, causa (EOLS) | `data/processed/populacao.csv` |
| `02_dieta.py` | harmonização do bloco de dieta (versões df/dt) | `data/processed/dieta.csv` |
| `03_features.py` | 33 atributos (estáveis, estilo de vida, saúde) | `data/processed/features.csv` |
| `04_descritivo.py` | Tabela da população, Kaplan-Meier por faixa | `resultados/`, `figuras/`, `../Artigo/artigo/tabelas/tab_populacao.tex` |
| `modelagem.py` | split/folds (gera `split.csv`/`folds.csv` na 1ª vez) e métricas | `resultados/split_verificacao.csv` |
| `05_baseline_idade.py` | Cox só idade e idade + porte | `resultados/baseline_*.csv` |
| `06_cox_linear.py` | Cox ridge (+ ablação sem saúde) | `resultados/cox_linear_*.csv` |
| `07_xgb_cox.py` | XGBoost Cox (+ ablação, busca, checagem de ordem) | `resultados/xgb_cox_*.csv/json` |

`--n-boot 1000` para a rodada final (o desenvolvimento usou 200). Testes: `python test_modelagem.py`.
Decisões e números: `../metodologia_projeto.md`.
