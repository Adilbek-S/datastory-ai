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
from datastory.profiler.profiler import profile_dataframe
from datastory.visualization.engine import suggest_charts


class WorkflowState(TypedDict, total=False):
    df: pd.DataFrame
    profile: DatasetProfile
    kpis: list[KPI]
    charts: list[ChartSpec]
    insights: list[Insight]


def _profile_node(state: WorkflowState) -> WorkflowState:
    return {"profile": profile_dataframe(state["df"])}


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


def run_analysis(df: pd.DataFrame) -> AnalysisResult:
    state = build_graph().invoke({"df": df})
    return AnalysisResult(
        profile=state["profile"],
        kpis=state["kpis"],
        charts=state["charts"],
        insights=state["insights"],
    )
