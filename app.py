"""Точка входа DataStory AI:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="DataStory AI", page_icon=":material/insights:", layout="wide")

from datastory.ui.navigation import PAGES  # noqa: E402 — после set_page_config
from datastory.ui.theme import apply_theme  # noqa: E402

apply_theme()
st.sidebar.markdown("### 📊 DataStory AI")
st.navigation(PAGES).run()
