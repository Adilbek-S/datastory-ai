# DataStory AI

Веб-приложение для интеллектуального анализа данных и автоматического формирования интерактивных аналитических дашбордов. Финальный проект курса LLM Engineering.

Пользователь загружает Excel, CSV или изображение с таблицей (и, по желанию, PDF с описанием показателей). Приложение анализирует структуру данных, предлагает визуализации, выполняет расчёты и формирует выводы.

> **Статус:** минимальный каркас (MVP). Работают загрузка CSV/Excel, профилирование, базовые показатели, графики Plotly, LangGraph-воркфлоу (без LLM), извлечение текста из PDF, MCP-сервер и тесты. LLM-выводы, OCR изображений и индексация в ChromaDB — следующие этапы.

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

### Тесты

```bash
pytest
```

### Демо-данные

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

### MCP-сервер

```bash
python -m datastory.mcp_server.server
```

Сервер работает по stdio и предоставляет инструмент `profile_file`.

## Структура

```
app.py                      точка входа Streamlit
datastory/
  config.py                 настройки из .env (pydantic-settings)
  models.py                 Pydantic-модели
  ui/                       тема, компоненты, страницы (views/)
  file_processing/          загрузка CSV/Excel, текст из PDF
  profiler/                 Dataset Profiler
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
