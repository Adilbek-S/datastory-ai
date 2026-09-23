"""Навигация между страницами Streamlit."""
import streamlit as st

from datastory.ui.views import about, analysis, home, knowledge

HOME = st.Page(home.render, title="Главная", icon=":material/home:", url_path="home", default=True)
ANALYSIS = st.Page(analysis.render, title="Анализ данных", icon=":material/monitoring:", url_path="analysis")
KNOWLEDGE = st.Page(knowledge.render, title="База знаний", icon=":material/menu_book:", url_path="knowledge")
ABOUT = st.Page(about.render, title="О проекте", icon=":material/info:", url_path="about")

PAGES = [HOME, ANALYSIS, KNOWLEDGE, ABOUT]
