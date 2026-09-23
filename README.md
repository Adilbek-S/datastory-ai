# DataStory AI

Веб-приложение для интеллектуального анализа данных и автоматического формирования интерактивных аналитических дашбордов. Финальный проект курса LLM Engineering.

Пользователь загружает Excel, CSV или изображение с таблицей (и, по желанию, PDF с описанием показателей). Приложение анализирует структуру данных, предлагает визуализации, выполняет расчёты и формирует выводы.

> **Статус:** минимальный каркас (MVP). Работают загрузка XLSX/CSV с профилированием и подтверждением структуры, рабочее хранилище датасетов, базовые показатели, графики Plotly, LangGraph-воркфлоу (без LLM), извлечение текста из PDF, MCP-сервер и тесты. LLM-выводы, OCR изображений и индексация в ChromaDB — следующие этапы.

## Стек

Python 3.11+ · Streamlit · Pandas · Plotly · LangGraph · OpenAI GPT-4o-mini · text-embedding-3-small · ChromaDB · MCP SDK (FastMCP) · PyMuPDF · Pydantic · LangSmith · Pytest

## Запуск

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # затем впишите OPENAI_API_KEY
streamlit run app.py
```

Приложение откроется на http://localhost:8501.

## Загрузка и профилирование данных

Страница «Анализ данных» ведёт пользователя по шагам: **загрузка → предпросмотр → профиль → качество данных → подтверждение структуры → результаты**.

- **Форматы:** XLSX (с выбором листа) и CSV (кодировки UTF-8/UTF-8-BOM/CP1251/UTF-16, разделители `, ; Tab |`).
- **Профиль** (`DatasetProfile`): названия и типы колонок, число строк, пропуски, дубликаты, списки числовых / категориальных / временных колонок, описание. Числа, записанные текстом («1 234,5»), и даты (`2026-03`, `31.12.2026`) распознаются автоматически.
- **Ручная настройка типов:** если тип определён неверно, его можно выбрать в таблице; профиль и данные пересчитываются. Колонку также можно отметить как содержащую персональные данные.
- **Качество данных** (`DataQualityIssue`): дубликаты, пропуски (info / warning / error по доле), пустые и константные колонки, колонки без названия, нераспознанные числа и даты.
- **Ошибки** показываются по-русски и не роняют приложение: пустой или повреждённый файл, неподдерживаемый формат, файл без данных, некорректные значения.
- **Подтверждение:** после подтверждения таблица (с выбранными типами) сохраняется в рабочее хранилище `data/workspace/<dataset_id>/` (parquet + JSON). Другие компоненты находят датасет по ID через `DatasetStore` / `DatasetReference`.
- **Без LLM:** строки, пропуски, дубликаты и статистика считаются Pandas.

### Данные для LLM

В LLM никогда не передаётся таблица целиком. `build_llm_profile(profile)` строит краткий профиль только из `DatasetProfile`: названия колонок, типы, число строк, несколько примеров значений и основные статистики. Персональные данные исключаются автоматически: колонки, похожие на e-mail, телефон, ФИО, адрес, карту и т.п. (по названию и значениям) или отмеченные пользователем, передаются только названием и типом; свободный текст без примеров; значения, похожие на e-mail/телефон/карту, маскируются; имя файла не передаётся.

```python
from datastory.file_processing.loader import read_table
from datastory.profiler.profiler import build_profile
from datastory.profiler.llm_profile import build_llm_profile
from datastory.storage.store import DatasetStore

df = read_table(open("data/demo/transactions_2026.xlsx", "rb").read(), "transactions_2026.xlsx")
profile, typed = build_profile(df, "transactions_2026.xlsx")
ref = DatasetStore().save(typed, profile)          # DatasetReference с dataset_id
print(build_llm_profile(profile).to_prompt())      # компактный текст для промпта
```

## Тесты

```bash
pytest
```

## Демо-данные

В `data/demo/` лежат полностью синтетические данные вымышленной платёжной системы DemoPay KZ (реальные данные и организации не используются):

| Файл | Содержимое |
|---|---|
| `transactions_2026.xlsx` | статистика январь — июнь 2026: Month, Transactions, Successful, Failed, Amount_KZT, Channel (Mobile / Web / API), 18 строк; в марте успешность транзакций заметно снижена |
| `business_metrics.pdf` | описание показателей (Success Rate, Transaction Volume, Average Transaction Amount), каналов и контекстного события в марте |
| `transactions_screenshot.png` | изображение части таблицы (февраль — март) для распознавания Vision-моделью |
| `expected_metrics.json` | контрольные показатели, посчитанные обычным Python, — для автотестов |

Файлы воспроизводятся командой (фиксированный seed, результат побайтно одинаков при каждом запуске):

```bash
python scripts/generate_demo_data.py            # или --out <каталог>
```

Тест `test_committed_demo_files_are_up_to_date` следит, чтобы закоммиченные файлы совпадали с выводом генератора.

## MCP-сервер

```bash
python -m datastory.mcp_server.server
```

Сервер работает по stdio и предоставляет инструменты `profile_file`, `list_datasets` и `get_dataset_profile` (возвращают только краткий профиль, без строк таблицы).

## Структура

```
app.py                      точка входа Streamlit
datastory/
  config.py                 настройки из .env (pydantic-settings)
  models.py                 Pydantic-модели
  ui/                       тема, компоненты, страницы (views/)
  errors.py                 исключения с сообщениями для пользователя
  file_processing/          загрузка XLSX/CSV, листы, текст из PDF
  profiler/                 типы колонок, качество данных, ПДн, профиль, профиль для LLM
  storage/                  рабочее хранилище датасетов (DatasetStore)
  rag/                      RAG Engine (ChromaDB)
  workflow/                 LangGraph Workflow
  mcp_server/               MCP Server (FastMCP)
  analytics/                Analytics Engine
  visualization/            Visualization Engine
  insights/                 Insight Generator
  evaluation/               Evaluation Pipeline
scripts/                    generate_demo_data.py — генератор демо-данных
data/demo/                  синтетические демо-данные
tests/                      Pytest
```

## Зависимости

`requirements.in` — прямые зависимости, `requirements.txt` — зафиксированные версии. Пакет `mcp` закреплён на ветке 1.x (`mcp<2`), потому что в 2.x `FastMCP` переименован в `MCPServer`.
