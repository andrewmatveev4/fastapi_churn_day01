# Churn Prediction Service

Сервис на FastAPI, предсказывающий отток клиентов (churn) на основе их поведения и характеристик. Принимает признаки клиента, обучает ML-модель (логистическая регрессия или случайный лес) и возвращает вероятность оттока.

## Формат датасета

`data/churn_dataset.csv` — 2000 строк. Девять признаков и целевая переменная `churn`.

| Колонка | Тип | Описание |
|---|---|---|
| `monthly_fee` | float | Месячная плата |
| `usage_hours` | float | Часы использования |
| `support_requests` | int | Обращений в поддержку |
| `account_age_months` | int | Возраст аккаунта (месяцы) |
| `failed_payments` | int | Неудавшихся платежей |
| `region` | str | `america` / `europe` / `asia` |
| `device_type` | str | `desktop` / `mobile` |
| `payment_method` | str | `card` / `paypal` |
| `autopay_enabled` | int | Автоплатёж: 0 или 1 |
| `churn` | int | Целевая: ушёл клиент (1) или нет (0) |

Классы несбалансированы (примерно 80/20 в пользу оставшихся).

## Структура проекта

main.py — точка входа: app, lifespan, обработчики ошибок, подключение роутера
api/routes.py — HTTP-эндпоинты
ml/pipeline.py — подготовка данных, обучение, метрики
ml/model_store.py — сохранение/загрузка модели (joblib)
ml/history_store.py — история обучений (JSON)
schemas/churn.py — Pydantic-модели запросов и ответов
core/errors.py — кастомное исключение сервиса
core/state.py — состояние модели в памяти
tests/ — pytest: unit и интеграционные тесты


## Запуск локально

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Сервис на `http://127.0.0.1:8000`. Документация — `http://127.0.0.1:8000/docs`.

## Запуск в Docker

```bash
docker build -t churn-service .
docker run -d -p 8000:8000 --name churn churn-service
```

Проверка: `http://127.0.0.1:8000/health` и `http://127.0.0.1:8000/docs`.

## Эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Состояние сервиса (модель, датасет) |
| POST | `/model/train` | Обучить модель |
| POST | `/predict` | Предсказать отток |
| GET | `/model/status` | Статус текущей модели |
| GET | `/model/metrics` | История обучений и метрики |
| GET | `/model/schema` | Схема входных признаков |
| GET | `/dataset/preview` | Первые строки датасета |
| GET | `/dataset/info` | Информация о датасете |
| GET | `/dataset/split-info` | Разбиение train/test |

## Примеры запросов

Обучение:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"max_iter": 1000}}'
```

Ответ:

```json
{"metrics": {"accuracy": 0.79, "f1": 0.05, "roc_auc": 0.61}, "trained_at": "2026-09-21T..."}
```

Предсказание:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 50.0, "usage_hours": 10.0, "support_requests": 2,
    "account_age_months": 12, "failed_payments": 0, "region": "europe",
    "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1
  }'
```

Ответ:

```json
{"predicted_class": 0, "probabilities": [0.81, 0.19]}
```

## Тесты

```bash
python -m pytest -v
```