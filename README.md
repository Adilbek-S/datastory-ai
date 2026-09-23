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
tests/                      Pytest
```

## Зависимости

`requirements.in` — прямые зависимости, `requirements.txt` — зафиксированные версии. Пакет `mcp` закреплён на ветке 1.x (`mcp<2`), потому что в 2.x `FastMCP` переименован в `MCPServer`.
