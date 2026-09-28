"""Pydantic-модели воркфлоу: план анализа, показатели, дашборд, результат, решение пользователя.

MetricDefinition здесь — то, что видит и подтверждает пользователь (название, формула, откуда определение).
Это не то же самое, что datastory.analytics.metrics.MetricDefinition — правило расчёта на стороне MCP-сервера.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from datastory.analytics.models import DatasetSummary, FilterCondition, MetricResult
from datastory.models import KPI, ChartSpec, Insight

MAX_CHARTS = 4  # для MVP в плане не больше четырёх графиков
AUTO_GOAL = (
    "Найди наиболее значимые тенденции, сравнения и аномалии в наборе данных "
    "и предложи не более четырёх полезных визуализаций"
)  # цель пользователя для кнопки «Предложить анализ автоматически»

AnalysisId = Literal[
    "count_dynamics", "volume_dynamics", "success_dynamics", "channel_distribution",  # показатели платёжной системы
    "measure_dynamics", "measure_comparison",  # универсальные: сумма числовой колонки (продажи и т. п.)
    "custom",  # шаг, составленный по запросу пользователя (показатель, группировка, фильтры)
]
ChartType = Literal["line", "bar", "pie"]
MappingStatus = Literal["confident", "confirmed", "ambiguous"]
Phase = Literal["awaiting_approval", "completed", "empty", "cancelled", "failed"]


# --------------------------------------------------------------------------- намерение и доступность
class AnalysisIntent(BaseModel):
    """Что пользователь хочет увидеть (из запроса или по умолчанию — все поддерживаемые анализы)."""

    analyses: list[AnalysisId]
    source: Literal["llm", "default"] = "default"
    unsupported: list[str] = Field(default_factory=list, description="Запрошенное, чего MVP не умеет")
    comment: str = ""


class AnalysisCandidate(BaseModel):
    """Один из поддерживаемых анализов и его доступность для этого датасета."""

    id: AnalysisId
    label: str
    metric: str
    available: bool
    reasons: list[str] = Field(default_factory=list, description="Почему недоступен")
    dimension_candidates: list[str] = Field(default_factory=list, description="Колонки, подходящие для оси X")


# --------------------------------------------------------------------------- план
class MetricDefinition(BaseModel):
    metric: str
    label: str
    unit: str
    formula: str
    source_columns: list[str]
    documented: bool = Field(description="Есть ли определение в загруженной документации")
    definition_source: str | None = None
    definition_quote: str | None = None
    formula_confirmed_by_user: bool = False


class ColumnMapping(BaseModel):
    role: Literal["time", "category"]
    column: str | None = None
    candidates: list[str] = Field(default_factory=list)
    status: MappingStatus
    basis: str = ""


class AnalysisStep(BaseModel):
    """Шаг плана анализа: один график с показателем, группировкой и обоснованием (карточка на экране плана)."""

    chart_id: str
    analysis: AnalysisId
    title: str
    metric: str
    chart_type: ChartType
    x_column: str | None = None
    mapping: ColumnMapping
    rationale: str = ""
    group_by: list[str] = Field(default_factory=list, description="Все колонки группировки (первая — ось X); пусто — по x_column")
    filters: list[FilterCondition] = Field(default_factory=list, description="Условия отбора строк (только у шагов по запросу)")

    @property
    def grouping(self) -> list[str]:
        return self.group_by or ([self.x_column] if self.x_column else [])


PlannedChart = AnalysisStep  # прежнее название


class ExcludedItem(BaseModel):
    analysis: str
    label: str
    reason: str


class Clarification(BaseModel):
    chart_id: str
    question: str
    candidates: list[str]


class AnalysisPlan(BaseModel):
    dataset_id: str
    goal: str = ""
    planner: Literal["llm", "rules"] = "rules"
    charts: list[AnalysisStep] = Field(default_factory=list, max_length=MAX_CHARTS)
    metrics: list[MetricDefinition] = Field(default_factory=list)
    excluded: list[ExcludedItem] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    documentation_gaps: list[str] = Field(default_factory=list, description="Показатели без бизнес-определения")
    notes: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list, description="Запрошенное, чего система не строит (нет показателя, колонки, значения)")
    available_analyses: list[AnalysisId] = Field(default_factory=list)

    def metric_definition(self, metric: str) -> MetricDefinition | None:
        return next((m for m in self.metrics if m.metric == metric), None)


# --------------------------------------------------------------------------- подтверждение пользователем
class ApprovalRequest(BaseModel):
    """Значение interrupt: что показать пользователю. Хранится в чекпоинте, поэтому переживает перезапуск Streamlit."""

    plan: AnalysisPlan
    notice: str = ""
    round: int = 1


class ApprovalDecision(BaseModel):
    """Ответ пользователя (значение resume)."""

    action: Literal["approve", "revise", "cancel"]
    approved_chart_ids: list[str] = Field(default_factory=list)
    confirmed_metrics: list[str] = Field(
        default_factory=list, description="Показатели без документации, правило расчёта которых подтвердил пользователь"
    )
    column_choices: dict[str, str] = Field(default_factory=dict, description="chart_id -> выбранная колонка")
    selected_analyses: list[AnalysisId] = Field(default_factory=list, description="Новый выбор анализов при action=revise")


# --------------------------------------------------------------------------- результат
class InsightCheck(BaseModel):
    chart_id: str
    passed: bool
    attempts: int = 1
    violations: list[str] = Field(default_factory=list)
    fallback: bool = False  # после нарушений вывод заменён детерминированным


class DashboardSection(BaseModel):
    chart: ChartSpec
    insight: Insight | None = None
    metric_label: str = ""
    formula: str = ""


class DashboardSpec(BaseModel):
    title: str
    dataset_id: str
    filename: str
    kpis: list[KPI] = Field(default_factory=list)
    sections: list[DashboardSection] = Field(default_factory=list)
    documentation_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MethodologyInfo(BaseModel):
    """Какая методика (Skill) применялась и на каких LLM-этапах её инструкции попали в промпт."""

    skill: str
    description: str = ""
    stages: list[str] = Field(default_factory=list, description="plan | insights")
    rules: list[str] = Field(default_factory=list, description="Идентификаторы правил, объявленных в Skill")


class ContextSource(BaseModel):
    """Фрагмент базы знаний (RAG), найденный при анализе: определение показателя или контекст."""

    citation: str
    text: str
    kind: Literal["definition", "event"]
    metric_label: str | None = None
    used_in: list[str] = Field(default_factory=list, description="Названия выводов, которые ссылаются на этот источник")


class AnalysisResult(BaseModel):
    dataset_id: str
    goal: str = ""
    summary: DatasetSummary
    plan: AnalysisPlan | None = None
    kpis: list[KPI] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    insight_checks: list[InsightCheck] = Field(default_factory=list)
    dashboard: DashboardSpec | None = None
    methodology: MethodologyInfo | None = None
    context_sources: list[ContextSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- схемы ответов LLM
class IntentDraft(BaseModel):
    """Ответ LLM на этапе analyze_intent."""

    analyses: list[AnalysisId] = Field(description="Какие из поддерживаемых анализов запросил пользователь")
    unsupported: list[str] = Field(description="Запрошенное, чего нет среди поддерживаемых анализов (пустой список, если всё поддерживается)")
    comment: str = Field(description="Одно предложение: как понят запрос")


class ChartDraft(BaseModel):
    analysis: AnalysisId
    title: str = Field(description="Название графика на русском")
    chart_type: ChartType
    x_column: str = Field(description="Колонка датасета для оси X: дата/период или категория")
    rationale: str = Field(description="Одно предложение: зачем этот график")


class FilterDraft(BaseModel):
    column: str = Field(description="Колонка-измерение из списка")
    op: Literal["eq", "in", "ne", "not_in"]
    values: list[str] = Field(description="Значения как в данных: одно для eq/ne, несколько для in/not_in; период — «2026-03»")


class StepDraft(BaseModel):
    """Шаг плана по запросу пользователя (набор данных без колонок платёжной системы)."""

    title: str = Field(description="Название графика на русском")
    metric: str = Field(description="Идентификатор показателя строго из списка доступных")
    requested_term: str = Field(default="", description="Фраза из запроса, которой пользователь назвал показатель («cancellation rate», «средний чек»); пусто, если показатель не назван")
    group_by: list[str] = Field(description="Колонки группировки из списка измерений: не более двух, дата первой; пусто, если запрос неоднозначен")
    filters: list[FilterDraft] = Field(description="Условия отбора из запроса; пустой список, если фильтров нет")
    chart_type: ChartType
    rationale: str = Field(description="Одно предложение: зачем этот график")


class PlanDraft(BaseModel):
    """Ответ LLM на этапе build_analysis_plan (проверяется в validate_analysis_plan)."""

    goal: str = Field(description="Цель анализа одной фразой")
    charts: list[ChartDraft] = Field(
        default_factory=list,
        description="ТОЛЬКО для устаревшего каталога платёжных анализов, когда явно выбраны его пункты; иначе всегда [] — обычный план идёт в steps",
    )
    steps: list[StepDraft] = Field(default_factory=list, description="План анализа по запросу пользователя: сюда, а не в charts, идёт каждый график")
    unsupported: list[str] = Field(
        default_factory=list, description="Запрошенные показатели и действия, которых нет среди доступных (не придумывай их): по одной фразе на каждое"
    )
    ambiguous: bool = Field(default=False, description="Запрос допускает несколько разных толкований группировки, выбрать без уточнения нельзя")
