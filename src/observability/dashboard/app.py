"""Modular RAG Dashboard – multi-page Streamlit application.

Entry-point: ``streamlit run src/observability/dashboard/app.py``
"""

from __future__ import annotations

import streamlit as st
from src.observability.dashboard.i18n import init_language, render_language_selector, t


def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


def _page_query() -> None:
    from src.observability.dashboard.pages.query_page import render
    render()


def main() -> None:
    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="📊",
        layout="wide",
    )

    init_language()
    render_language_selector()

    pages = [
        st.Page(_page_overview, title=t("nav.overview"), icon="📊", default=True),
        st.Page(_page_query, title=t("nav.query"), icon="🔎"),
        st.Page(_page_data_browser, title=t("nav.data_browser"), icon="🔍"),
        st.Page(_page_ingestion_manager, title=t("nav.ingestion_manager"), icon="📥"),
        st.Page(_page_ingestion_traces, title=t("nav.ingestion_traces"), icon="🔬"),
        st.Page(_page_query_traces, title=t("nav.query_traces"), icon="📡"),
        st.Page(_page_evaluation_panel, title=t("nav.evaluation_panel"), icon="📏"),
    ]

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    main()
