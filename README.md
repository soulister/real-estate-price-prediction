# real-estate-price-prediction

Dynamic real estate valuation service using gradient boosting models and stacking.

## Pipeline

Настройки в `config.yaml` в корне проекта. Режим задаётся полем `mode`:

| mode | Описание |
|------|----------|
| `predict` | Загрузить сохранённые модели и сделать сабмит |
| `retrain` | Переобучить модели на полном train и сделать сабмит |
| `retune` | Optuna-тюнинг на dev, затем retrain |

Запуск из корня проекта:

```bash
.venv\Scripts\python.exe -m src.pipeline
```

Перед первым запуском:

```bash
pip install -r requirements.txt
```

## Структура

- `src/preprocessing.py` — препроцессинг и feature engineering
- `src/modeling.py` — CV и обучение бустингов
- `src/stacking.py` — мета-модель Ridge
- `src/pipeline.py` — оркестратор
- `notebooks/` — исследовательские ноутбуки
