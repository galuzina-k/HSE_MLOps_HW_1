# HSE FTIAD MLOps HW 1
Выполнила: **Белоновская Кристина Константиновна**

## Структура проекта
```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Базовый класс модели
│   │   ├── implementations.py  # Реализации моделей
│   │   └── registry.py      # Реестр моделей
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── api_schemas.py   # Pydantic схемы
│   ├── storage/
│   │   ├── __init__.py
│   │   └── model_storage.py # Хранилище моделей
│   └── utils/
│       ├── __init__.py
│       └── logger.py        # Настройка логирования
├── dashboard/
│   └── streamlit_app.py     # Streamlit дашборд
├── tests/
│   ├── test_api.py          # Тесты API
│   └── test_models.py       # Тесты моделей
├── pyproject.toml           # Конфигурация Poetry
├── poetry.lock              # Зафиксированные зависимости
└── README.md
```


## Установка
1. Установите зависимости с помощью Poetry:
```bash
poetry install
```

2. Активируйте виртуальное окружение:
```bash
poetry shell
```

## Запуск

### 1. Запуск API сервера

```bash
uvicorn app.main:app --reload
```

### 2. Запуск Streamlit дашборда

```bash
poetry shell
streamlit run dashboard/streamlit_app.py
```

* API: http://localhost:8000
* Swagger: http://localhost:8000/docs
* Dashboard: http://localhost:8501

## Доступные эндпоинты

### `GET /`
Корневой эндпоинт API.

### `GET /health`
Проверка статуса API.

### `GET /models/types`
Список доступных типов моделей с описанием гиперпараметров.

### `POST /models/train`
Обучение модели. Если модель с таким именем уже существует, она будет удалена и обучена заново.

**Запрос:**
```json
{
  "model_type": "logistic_regression",
  "model_name": "my_classifier",
  "hyperparameters": {"C": 1.0, "max_iter": 100},
  "X_train": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  "y_train": [0, 1, 0]
}
```

### `POST /models/predict`
Получение предсказаний от обученной модели.

**Запрос:**
```json
{
  "model_name": "my_classifier",
  "X": [[1.5, 2.5], [3.5, 4.5]]
}
```

### `GET /models`
Список всех обученных моделей.

### `GET /models/{model_name}`
Информация о конкретной модели (тип, гиперпараметры).

### `DELETE /models/{model_name}`
Удаление обученной модели.

## Доступные модели

### 1. Linear Regression
- **Описание**: Линейная регрессия для задач регрессии
- **Гиперпараметры**:
  - `fit_intercept` (bool): использовать ли beta_0 (по умолчанию: true)

### 2. Logistic Regression
- **Описание**: Логистическая регрессия для бинарной классификации
- **Гиперпараметры**:
  - `C` (float): обратная сила регуляризации (по умолчанию: 1.0)
  - `max_iter` (int): максимальное количество итераций оптимизации (по умолчанию: 100)

### 3. Random Forest
- **Описание**: Случайный лес для задач классификации
- **Гиперпараметры**:
  - `n_estimators` (int): количество деревьев (по умолчанию: 100)
  - `max_depth` (int): максимальная глубина деревьев (по умолчанию: None)
  - `random_state` (int): seed для воспроизводимости (по умолчанию: 42)

## Логирование
Все операции логируются в:
- **Консоль**: вывод в stdout

Логи включают:
- Обучение моделей
- Предсказания
- Удаление моделей
- Ошибки и предупреждения

## Проверки
### Проверка стиля кода
```bash
poetry run ruff check .
```

### Тесты
```bash
poetry run pytest tests/ -v
```
