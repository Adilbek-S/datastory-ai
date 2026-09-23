"""LangGraph Workflow: profile -> analytics -> visualization -> insights.

Узлы пока детерминированные (без LLM). На следующих этапах внутрь узлов
добавятся вызовы GPT-4o-mini и RAG.
"""
from __future__ import annotations

from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from datastory.analytics.engine import basic_kpis
from datastory.insights.generator import generate_insights
from datastory.models import AnalysisResult, ChartSpec, DatasetProfile, Insight, KPI
from datastory.profiler.profiler import build_profile
from datastory.storage.store import DatasetStore
from datastory.visualization.engine import suggest_charts


class WorkflowState(TypedDict, total=False):
    df: pd.DataFrame
    profile: DatasetProfile
    kpis: list[KPI]
    charts: list[ChartSpec]
    insights: list[Insight]


def _profile_node(state: WorkflowState) -> WorkflowState:
    if "profile" in state:  # профиль уже подтверждён пользователем и сохранён в хранилище
        return {}
    profile, typed = build_profile(state["df"], "dataset")
    return {"profile": profile, "df": typed}


def _analytics_node(state: WorkflowState) -> WorkflowState:
    return {"kpis": basic_kpis(state["profile"])}


def _visualization_node(state: WorkflowState) -> WorkflowState:
    return {"charts": suggest_charts(state["profile"])}


def _insights_node(state: WorkflowState) -> WorkflowState:
    return {"insights": generate_insights(state["profile"])}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("profile", _profile_node)
    graph.add_node("analytics", _analytics_node)
    graph.add_node("visualization", _visualization_node)
    graph.add_node("insights", _insights_node)
    graph.add_edge(START, "profile")
    graph.add_edge("profile", "analytics")
    graph.add_edge("analytics", "visualization")
    graph.add_edge("visualization", "insights")
    graph.add_edge("insights", END)
    return graph.compile()


def _to_result(state: WorkflowState) -> AnalysisResult:
    return AnalysisResult(
        profile=state["profile"],
        kpis=state["kpis"],
        charts=state["charts"],
        insights=state["insights"],
    )


def run_analysis(df: pd.DataFrame, profile: DatasetProfile | None = None) -> AnalysisResult:
    """Запускает граф. Если профиль не передан, он строится автоматически."""
    initial: WorkflowState = {"df": df}
    if profile is not None:
        initial["profile"] = profile
    return _to_result(build_graph().invoke(initial))


def run_analysis_for_dataset(dataset_id: str, store: DatasetStore | None = None) -> AnalysisResult:
    """Находит подтверждённый датасет по ID в рабочем хранилище и анализирует его."""
    store = store or DatasetStore()
    return run_analysis(store.load_dataframe(dataset_id), store.load_profile(dataset_id))
