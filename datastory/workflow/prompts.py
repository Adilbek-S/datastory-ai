"""Промпты этапов воркфлоу. Таблица целиком в LLM не передаётся: только сводка датасета (DatasetSummary) без персональных данных."""
from __future__ import annotations

from datastory.analytics.models import DatasetSummary
from datastory.skills import Skill
from datastory.workflow.context import BusinessContext
from datastory.workflow.methodology import methodology_prompt
from datastory.workflow.models import MAX_CHARTS, AnalysisCandidate

def intent_system(candidates: list[AnalysisCandidate]) -> str:
    """Список поддерживаемых анализов зависит от набора данных: платёжные показатели или суммы числовых колонок."""
    return (
        "Ты определяешь, какие из поддерживаемых анализов запросил пользователь. Поддерживаются только:\n"
        + "\n".join(f"- {c.id}: {c.label}" for c in candidates)
        + "\nВ analyses верни идентификаторы запрошенных анализов из этого списка. Всё, что запрошено сверх списка "
        "(прогноз, причины, другие показатели), перечисли в unsupported. Если запрос общий («проанализируй данные»), "
        "верни все анализы."
    )



PLAN_TASK = (
    f"Ты составляешь план анализа данных: не более {MAX_CHARTS} графиков, по одному на каждый выбранный анализ. "
    "Для каждого графика укажи: анализ (analysis), название на русском, тип графика, колонку оси X и одно предложение обоснования. "
    "Колонку оси X выбирай только из перечисленных кандидатов; анализы вне списка выбранных не добавляй."
)

INSIGHT_TASK = (
    "Ты пишешь аналитический вывод по одному графику и возвращаешь структурный ответ: название (title), краткий вывод (summary, "
    "1–3 предложения), важность (severity), id использованных числовых доказательств (evidence_ids), ограничение интерпретации "
    "(limitation). Если в выводе использован фрагмент документа, укажи его chunk_id и дословную цитату."
)


CUSTOM_PLAN_TASK = (
    f"Ты составляешь план анализа по запросу пользователя в поле steps: не более {MAX_CHARTS} шагов, обычно один. "
    "Шаг — показатель (metric строго из списка доступных id), колонки группировки (group_by из списка измерений: не более двух, "
    "дата первой), фильтры (filters) и тип графика. В схеме ответа есть ещё поле charts — оно НЕ для этого сценария (это отдельный "
    "устаревший каталог платёжных анализов, который здесь не применяется): charts всегда оставляй пустым [], каждый график описывай "
    "только в steps, даже если по смыслу он похож на один из типовых анализов. Правила:\n"
    "1. «Динамика», «изменение», «рост», «тренд», «по месяцам» — группировка по дате; сравнение категорий во времени — group_by [дата, категория].\n"
    "2. «Сравни», «по регионам/каналам/категориям» — группировка по названной категории.\n"
    "3. Если показатель не назван, используй основной показатель набора (указан в запросе).\n"
    "4. Если названы конкретные значения категории или периода («Web и Mobile», «только Almaty», «за март») — обязательно добавь filters "
    "по этой колонке с этими значениями (значения как в списке значений); для сравнения самих значений group_by — эта же колонка.\n"
    "5. Если группировка не названа, а в наборе есть дата — динамика по дате. Если группировка названа неопределённо («по группам», "
    "«по разрезам») — верни один шаг с пустым group_by и ambiguous=true, не размножай шаги по всем измерениям.\n"
    "6. Показатель, которого нет в списке доступных id, НЕ подменяй похожим по смыслу. Если пользователь просит EBITDA, прибыль, "
    "Return Rate, конверсию и т. п., а такого id нет, — не строй шаг, а перечисли запрошенное в unsupported. Даже если термин описан "
    "в документах, но подходящей колонки в данных нет, показателя тоже нет.\n"
    "7. Прогноз, будущие периоды и любые действия, кроме построения графиков по имеющимся данным, — только unsupported, без шагов.\n"
    "8. Не добавляй шаги, которых пользователь не просил.\n"
    "9. Разные показатели не взаимозаменяемы: выручка (Revenue) — не прибыль, маржа, EBITDA, себестоимость, расходы; отмены — не возвраты. "
    "Если запрошенного показателя нет в списке, шаг не строится, даже если в списке есть показатель из той же области.\n"
    "10. Ответ обязан содержать либо шаги, либо unsupported: пустой ответ недопустим."
)


def plan_system(skill: Skill, custom: bool = False) -> str:
    """Системный промпт этапа build_analysis_plan: формат ответа + разделы Skill для этого этапа."""
    return "\n\n".join([CUSTOM_PLAN_TASK if custom else PLAN_TASK, methodology_prompt(skill, "plan")])


def insight_system(skill: Skill) -> str:
    """Системный промпт этапа generate_insights: формат ответа + разделы Skill для этого этапа."""
    return "\n\n".join([INSIGHT_TASK, methodology_prompt(skill, "insights")])


def summary_text(summary: DatasetSummary) -> str:
    lines = [f"Датасет: {summary.row_count} строк, {summary.column_count} колонок."]
    for column in summary.columns:
        extra = ""
        if column.statistics and not column.is_sensitive:
            extra = f"; {column.statistics}"
        lines.append(f"- {column.name}: {column.kind}, уникальных значений {column.unique_count}{extra}")
    return "\n".join(lines)


def intent_prompt(request: str, summary: DatasetSummary) -> str:
    return f"Запрос пользователя: «{request}»\n\n{summary_text(summary)}"


def custom_plan_prompt(summary: DatasetSummary, context: BusinessContext, request: str, show_documents: bool = True) -> str:
    """Данные для планировщика по запросу: показатели, измерения со значениями, найденные определения. Строк таблицы нет.

    show_documents=False — контекст RAG не передаётся вовсе (раздел с документами не добавляется, даже как «ничего не найдено»).
    """
    from datastory.analytics.metrics import get_metric

    metrics = "\n".join(
        f"- {m} | {get_metric(m).label} | {get_metric(m).formula} | {get_metric(m).unit or 'без единицы'}" for m in summary.available_metrics
    )
    dimensions = []
    for column in summary.columns:
        if column.name not in summary.dimensions:
            continue
        stats = column.statistics or {}
        if column.kind == "datetime":
            dimensions.append(f"- {column.name}: дата/период, от {stats.get('min')} до {stats.get('max')}")
        else:
            values = ", ".join((stats.get("top_values") or {}))
            dimensions.append(f"- {column.name}: категория, значения: {values}")
    documents = "\n".join(f"[{h.source.citation}] {h.text}" for h in context.query_hits) or "по запросу ничего не найдено"
    from datastory.workflow.catalog import measure_metrics, primary_measure

    default_metric = primary_measure(measure_metrics(summary)) if measure_metrics(summary) else None
    return "\n\n".join([
        f"Запрос пользователя: «{request.strip()}»",
        f"Основной показатель набора (если в запросе не назван): {default_metric}.",
        f"Набор данных: {summary.row_count} строк, колонки: {', '.join(summary.column_names)}.",
        "Доступные показатели (id | название | формула | единица):\n" + metrics,
        "Измерения:\n" + "\n".join(dimensions),
        *(["Фрагменты документов, найденные по запросу (определения терминов):\n" + documents] if show_documents else []),
    ])


def plan_prompt(
    summary: DatasetSummary, candidates: list[AnalysisCandidate], selected: list[str], context: BusinessContext, request: str
) -> str:
    rows = []
    for c in candidates:
        if c.id not in selected:
            continue
        if c.available:
            rows.append(f"- {c.id} ({c.label}): доступен; кандидаты для оси X: {', '.join(c.dimension_candidates)}")
        else:
            rows.append(f"- {c.id} ({c.label}): НЕДОСТУПЕН — {'; '.join(c.reasons)} (в план не включать)")
    documented = [m for m in context.definitions] or ["нет"]
    parts = [
        summary_text(summary),
        "Выбранные анализы:\n" + "\n".join(rows),
        f"Показатели с бизнес-определением в документации: {', '.join(documented)}.",
    ]
    if request.strip():
        parts.insert(0, f"Запрос пользователя: «{request.strip()}»")
    return "\n\n".join(parts)
