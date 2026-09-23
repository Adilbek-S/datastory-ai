"""Светлая тема в стиле аналитического SaaS: CSS и переиспользуемые компоненты."""
from __future__ import annotations

from html import escape

import streamlit as st

from datastory.models import KPI

CSS = """
<style>
:root { --ds-primary:#4F46E5; --ds-border:#E5E7EB; --ds-muted:#6B7280; }
.block-container { padding-top: 4rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--ds-border); }
.ds-hero { background: linear-gradient(135deg,#EEF2FF 0%,#F5F3FF 60%,#FFFFFF 100%);
  border:1px solid #E0E7FF; border-radius:20px; padding:2.2rem 2.4rem; margin-bottom:1.5rem; }
.ds-hero h1 { margin:0 0 .4rem 0; font-size:2.2rem; }
.ds-hero p { margin:0; color:#4B5563; font-size:1.05rem; max-width:720px; }
.ds-badge { display:inline-block; padding:.15rem .65rem; border-radius:999px; background:#EEF2FF;
  color:var(--ds-primary); font-size:.78rem; font-weight:600; margin-bottom:.8rem; }
.ds-card { background:#FFFFFF; border:1px solid var(--ds-border); border-radius:16px;
  padding:1.1rem 1.25rem; box-shadow:0 1px 2px rgba(16,24,40,.04); height:100%; }
.ds-card .label { color:var(--ds-muted); font-size:.82rem; font-weight:500; }
.ds-card .value { font-size:1.85rem; font-weight:700; margin:.15rem 0; }
.ds-card .hint { color:var(--ds-muted); font-size:.78rem; }
.ds-card h4 { margin:.2rem 0 .4rem 0; font-size:1.02rem; }
.ds-card p { margin:0; color:#4B5563; font-size:.92rem; }
.ds-step { color:var(--ds-primary); font-weight:700; font-size:.8rem; }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badge: str = "") -> None:
    badge_html = f'<span class="ds-badge">{escape(badge)}</span><br>' if badge else ""
    st.markdown(
        f'<div class="ds-hero">{badge_html}<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_card(kpi: KPI) -> str:
    hint = f'<div class="hint">{escape(kpi.hint)}</div>' if kpi.hint else ""
    return (
        f'<div class="ds-card"><div class="label">{escape(kpi.label)}</div>'
        f'<div class="value">{escape(kpi.value)}</div>{hint}</div>'
    )


def kpi_row(kpis: list[KPI]) -> None:
    for col, kpi in zip(st.columns(len(kpis)), kpis):
        col.markdown(kpi_card(kpi), unsafe_allow_html=True)


def info_card(title: str, text: str, step: str = "") -> str:
    step_html = f'<div class="ds-step">{escape(step)}</div>' if step else ""
    return f'<div class="ds-card">{step_html}<h4>{escape(title)}</h4><p>{escape(text)}</p></div>'
