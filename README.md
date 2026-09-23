# DataStory AI

Веб-приложение для интеллектуального анализа данных и автоматического формирования интерактивных аналитических дашбордов. Финальный проект курса LLM Engineering.

Пользователь загружает Excel, CSV или изображение с таблицей (и, по желанию, PDF с описанием показателей). Приложение анализирует структуру данных, предлагает визуализации, выполняет расчёты и формирует выводы.

> **Статус:** минимальный каркас (MVP). Работают загрузка XLSX/CSV с профилированием и подтверждением структуры, рабочее хранилище датасетов, базовые показатели, графики Plotly, RAG на ChromaDB (PDF-документы и метаданные датасетов), LangGraph-воркфлоу (без LLM), MCP-сервер и тесты. LLM-выводы и OCR изображений — следующие этапы.

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

## RAG и база знаний

Раздел «База знаний» работает с двумя категориями информации. RAG ищет **по смыслу** и возвращает источники; **точные числа он не считает и не хранит** — расчёты выполняет Pandas.

**1. Метаданные датасетов (data-aware RAG).** После подтверждения датасета автоматически индексируется его семантическое описание: название, описание, колонки и типы, безопасные примеры значений (без персональных данных), период данных, предполагаемые измерения и показатели. Строки таблицы не индексируются: датасет из 1000 строк — это 1 запись про датасет плюс по записи на колонку. Поиск сопоставляет фразу с колонкой («Количество операций» → `Transactions`, «Объём транзакций» → `Amount_KZT`). Если кандидаты почти равны по близости (например, слово «количество»), интерфейс просит пользователя подтвердить выбор; найденное соответствие — предположение системы, пока пользователь его не подтвердил.

**2. PDF-документы (document-aware RAG).** PyMuPDF извлекает текст, номера страниц и названия разделов (по размеру шрифта и нумерации). Chunking идёт по разделам; длинные разделы делятся по предложениям (до 900 символов) с перекрытием 120 символов. Для каждого фрагмента хранятся `document_id`, `filename`, `page`, `section`, `chunk_id` и `dataset_id` (если документ привязан к датасету). Поиск — Top-3 с источниками.

**Защита от повторной индексации.** `document_id` — хеш содержимого файла: тот же PDF под другим именем не индексируется заново, эмбеддинги при этом не запрашиваются. Эмбеддинги считаются до записи, поэтому при сбое API в индексе не остаётся половины документа.

**Факт, вычисление, предположение.** У каждого утверждения есть тип (`datastory/rag/evidence.py`): *факт из документа* (только с источником и дословной цитатой, найденной в фрагменте), *результат вычисления* (только с указанной формулой) и *предположение*. `validate_evidence` понижает «факт» без источника, с выдуманной цитатой или с причинным утверждением, которого нет в цитате, до предположения. Так модель не выдаст догадку за документально подтверждённый факт.

Эмбеддинги — `text-embedding-3-small`, хранилище — ChromaDB (`data/chroma`). Если ключ OpenAI не задан, включается **офлайн-режим**: детерминированный лексический поиск (по словам, не семантический), о чём интерфейс предупреждает. Векторы разных провайдеров лежат в разных коллекциях и не смешиваются. Режим задаёт `EMBEDDING_PROVIDER` (`auto` | `openai` | `offline`).

```python
from datastory.rag.api import (
    index_document, index_dataset_profile, search_business_context,
    search_dataset_metadata, get_source_reference, resolve_column,
)

index_document(open("data/demo/business_metrics.pdf", "rb").read(), "business_metrics.pdf")
result = search_business_context("формула Success Rate")        # Top-3
top = result.hits[0]
print(top.source.citation)                                        # business_metrics.pdf, стр. 1, раздел «4. Формула Success Rate»
print(get_source_reference(top.source.chunk_id))
```

## Тесты

```bash
pytest                                   # без сети и без ключа OpenAI (эмбеддинги — офлайн-провайдер)
RUN_LIVE_TESTS=1 pytest tests/test_rag_live.py   # живая проверка на text-embedding-3-small (ключ из .env)
```

`tests/conftest.py` принудительно отключает ключ и боевые каталоги, поэтому обычный прогон не тратит запросы OpenAI и не пишет в `data/chroma`.

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

Сервер работает по stdio и предоставляет инструменты `profile_file`, `list_datasets` и `get_dataset_profile` (возвращают только краткий профиль, без строк таблицы), а также `search_business_context`, `find_columns` и `get_source` для RAG.

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
  rag/                      RAG: pdf_parser, embeddings, knowledge_base (ChromaDB), dataset_descriptions, evidence, api
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
