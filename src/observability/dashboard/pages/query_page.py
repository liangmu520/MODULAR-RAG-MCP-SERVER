"""Knowledge Query page - search and get AI-generated answers from the knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import streamlit as st

from src.observability.dashboard.i18n import t


def _build_query_engine(settings: Any, collection: str):
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.query_engine.reranker import create_core_reranker
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.core.settings import resolve_path

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    dense_retriever = create_dense_retriever(
        settings=settings, embedding_client=embedding_client, vector_store=vector_store
    )
    bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse_retriever = create_sparse_retriever(
        settings=settings, bm25_indexer=bm25_indexer, vector_store=vector_store
    )
    sparse_retriever.default_collection = collection
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    reranker = create_core_reranker(settings=settings)
    return hybrid_search, reranker


def _do_search(query: str, collection: str, top_k: int, use_rerank: bool) -> List[Any]:
    from src.core.settings import load_settings
    settings = load_settings()
    hybrid_search, reranker = _build_query_engine(settings, collection)
    result = hybrid_search.search(query=query, top_k=top_k)
    results = result if isinstance(result, list) else result.results
    if use_rerank and reranker.is_enabled and results:
        rerank_result = reranker.rerank(query=query, results=results, top_k=top_k)
        results = rerank_result.results
    return results


def _generate_answer(query: str, results: List[Any]) -> str:
    """Call LLM to generate an answer based on retrieved chunks."""
    from src.core.settings import load_settings
    from openai import OpenAI

    settings = load_settings()

    # Build context from top results
    context_parts = []
    for i, r in enumerate(results[:5]):
        text = getattr(r, "text", "")
        meta = getattr(r, "metadata", {}) or {}
        source = Path(meta.get("source_path", meta.get("source", ""))).name
        context_parts.append(f"[{i+1}] 来源: {source}\n{text}")
    context = "\n\n".join(context_parts)

    prompt = f"""你是一个知识库问答助手。请根据以下检索到的文档内容，回答用户的问题。
回答要准确、简洁，并基于提供的文档内容。如果文档中没有相关信息，请如实说明。

检索到的文档内容：
{context}

用户问题：{query}

请给出回答："""

    # Use OpenAI-compatible client with settings
    api_key = getattr(settings.llm, "api_key", None) or "sk-placeholder"
    base_url = getattr(settings.llm, "base_url", None) or "https://api.openai.com/v1"
    model = settings.llm.model

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )
    return response.choices[0].message.content


def render() -> None:
    st.header(t("query.title"))

    # ── Input form ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        query = st.text_input(
            t("query.input_label"),
            placeholder=t("query.input_placeholder"),
            key="query_input",
        )
    with col2:
        collection = st.text_input(t("query.collection"), value="default", key="query_collection")
    with col3:
        top_k = st.number_input(t("query.top_k"), min_value=1, max_value=50, value=5, key="query_top_k")

    col_btn, col_rerank, col_ai = st.columns([1, 1, 2])
    with col_btn:
        search_clicked = st.button(t("query.btn"), type="primary", key="query_btn")
    with col_rerank:
        use_rerank = st.checkbox(t("query.rerank"), value=False, key="query_rerank")
    with col_ai:
        enable_answer = st.checkbox(t("query.enable_answer"), value=True, key="query_enable_answer")

    # ── Search ─────────────────────────────────────────────────────
    if search_clicked and query.strip():
        with st.spinner(t("query.searching")):
            try:
                results = _do_search(query.strip(), collection.strip() or "default", int(top_k), use_rerank)
            except Exception as exc:
                st.error(t("query.error", exc))
                return

        if not results:
            st.info(t("query.no_results"))
            return

        # ── AI Answer ───────────────────────────────────────────────
        if enable_answer:
            st.subheader(t("query.answer_section"))
            with st.spinner(t("query.answer_generating")):
                try:
                    answer = _generate_answer(query.strip(), results)
                    st.markdown(answer)
                except Exception as exc:
                    st.warning(t("query.answer_error", exc))

            st.divider()

        # ── Source references ───────────────────────────────────────
        st.subheader(t("query.ref_section") + f" ({len(results)})")

        for i, r in enumerate(results):
            score = getattr(r, "score", 0)
            text = getattr(r, "text", "")
            meta = getattr(r, "metadata", {}) or {}
            chunk_id = getattr(r, "chunk_id", "")
            source = meta.get("source_path", meta.get("source", ""))

            badge = "🟢" if score >= 0.7 else ("🟡" if score >= 0.4 else "🔴")

            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 3, 1])
                with c1:
                    st.metric(t("query.score"), f"{score:.4f}")
                with c2:
                    st.caption(f"{t('query.source')}: `{Path(source).name if source else '-'}`")
                    st.caption(f"{t('query.chunk_id')}: `{chunk_id[-20:] if chunk_id else '-'}`")
                with c3:
                    st.markdown(f"### {badge} #{i+1}")

                st.text_area(
                    t("query.content"),
                    value=text,
                    height=max(80, min(len(text) // 3, 300)),
                    disabled=True,
                    key=f"qr_text_{i}",
                    label_visibility="collapsed",
                )
                with st.expander("Metadata", expanded=False):
                    st.json(meta)
