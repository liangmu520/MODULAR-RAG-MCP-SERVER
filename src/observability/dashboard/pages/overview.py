"""Overview page - system configuration and data statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml

from src.observability.dashboard.i18n import t
from src.observability.dashboard.services.config_service import ConfigService
from src.core.settings import DEFAULT_SETTINGS_PATH, resolve_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml() -> Dict[str, Any]:
    path = resolve_path(DEFAULT_SETTINGS_PATH)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(data: Dict[str, Any]) -> None:
    path = resolve_path(DEFAULT_SETTINGS_PATH)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _safe_collection_stats() -> Dict[str, Any]:
    try:
        from src.core.settings import load_settings
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = load_settings()
        persist_dir = str(resolve_path(settings.vector_store.persist_directory))
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        stats: Dict[str, Any] = {}
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            collection = client.get_collection(name)
            stats[name] = {"chunk_count": collection.count()}
        return stats
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_llm_editor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    llm = cfg.get("llm", {})
    st.markdown("**LLM**")
    c1, c2 = st.columns(2)
    with c1:
        provider = st.selectbox("Provider", ["openai", "azure", "ollama", "deepseek"],
                                index=["openai", "azure", "ollama", "deepseek"].index(llm.get("provider", "openai")),
                                key="llm_provider")
        model = st.text_input("Model", value=llm.get("model", ""), key="llm_model")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0,
                                      value=float(llm.get("temperature", 0.0)), step=0.1, key="llm_temp")
    with c2:
        api_key = st.text_input("API Key", value=llm.get("api_key", ""), type="password", key="llm_api_key")
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=128000,
                                     value=int(llm.get("max_tokens", 4096)), key="llm_max_tokens")
        deployment = st.text_input("Deployment Name", value=llm.get("deployment_name", ""), key="llm_deployment")

    c3, c4 = st.columns(2)
    with c3:
        base_url = st.text_input("Base URL（第三方API填这里）", value=llm.get("base_url", ""),
                                 key="llm_base_url", placeholder="https://api.example.com/v1")
    with c4:
        azure_endpoint = st.text_input("Azure Endpoint", value=llm.get("azure_endpoint", ""), key="llm_azure_ep")

    st.caption("💡 使用第三方 OpenAI 兼容 API 时，填写 Base URL 并清空 Azure Endpoint")

    return {**llm, "provider": provider, "model": model, "temperature": temperature,
            "max_tokens": max_tokens, "api_key": api_key, "deployment_name": deployment,
            "base_url": base_url or None, "azure_endpoint": azure_endpoint}


def _render_embedding_editor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    emb = cfg.get("embedding", {})
    st.markdown("**Embedding**")
    c1, c2 = st.columns(2)
    with c1:
        provider = st.selectbox("Provider", ["openai", "azure", "ollama"],
                                index=["openai", "azure", "ollama"].index(emb.get("provider", "openai")),
                                key="emb_provider")
        model = st.text_input("Model", value=emb.get("model", ""), key="emb_model")
        dimensions = st.number_input("Dimensions", min_value=1, max_value=4096,
                                     value=int(emb.get("dimensions", 1536)), key="emb_dim")
    with c2:
        api_key = st.text_input("API Key", value=emb.get("api_key", ""), type="password", key="emb_api_key")
        deployment = st.text_input("Deployment Name", value=emb.get("deployment_name", ""), key="emb_deployment")

    c3, c4 = st.columns(2)
    with c3:
        base_url = st.text_input("Base URL（第三方API填这里）", value=emb.get("base_url", ""),
                                 key="emb_base_url", placeholder="https://api.example.com/v1")
    with c4:
        azure_endpoint = st.text_input("Azure Endpoint", value=emb.get("azure_endpoint", ""), key="emb_azure_ep")

    st.caption("💡 使用第三方 OpenAI 兼容 API 时，填写 Base URL 并清空 Azure Endpoint")

    return {**emb, "provider": provider, "model": model, "dimensions": dimensions,
            "api_key": api_key, "deployment_name": deployment,
            "base_url": base_url or None, "azure_endpoint": azure_endpoint}


def _render_retrieval_editor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ret = cfg.get("retrieval", {})
    rerank = cfg.get("rerank", {})
    st.markdown("**检索 & 重排**")
    c1, c2, c3 = st.columns(3)
    with c1:
        dense_top_k = st.number_input("Dense Top-K", min_value=1, max_value=200,
                                      value=int(ret.get("dense_top_k", 20)), key="ret_dense")
        sparse_top_k = st.number_input("Sparse Top-K", min_value=1, max_value=200,
                                       value=int(ret.get("sparse_top_k", 20)), key="ret_sparse")
    with c2:
        fusion_top_k = st.number_input("Fusion Top-K", min_value=1, max_value=100,
                                       value=int(ret.get("fusion_top_k", 10)), key="ret_fusion")
        rrf_k = st.number_input("RRF K", min_value=1, max_value=200,
                                value=int(ret.get("rrf_k", 60)), key="ret_rrf")
    with c3:
        rerank_enabled = st.checkbox("启用 Rerank", value=bool(rerank.get("enabled", False)), key="rerank_enabled")
        rerank_provider = st.selectbox("Rerank Provider", ["none", "cross_encoder", "llm"],
                                       index=["none", "cross_encoder", "llm"].index(rerank.get("provider", "none")),
                                       key="rerank_provider")
        rerank_top_k = st.number_input("Rerank Top-K", min_value=1, max_value=50,
                                       value=int(rerank.get("top_k", 5)), key="rerank_top_k")

    new_ret = {**ret, "dense_top_k": dense_top_k, "sparse_top_k": sparse_top_k,
               "fusion_top_k": fusion_top_k, "rrf_k": rrf_k}
    new_rerank = {**rerank, "enabled": rerank_enabled, "provider": rerank_provider, "top_k": rerank_top_k}
    return new_ret, new_rerank


def _render_ingestion_editor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ing = cfg.get("ingestion", {})
    st.markdown("**摄取参数**")
    c1, c2 = st.columns(2)
    with c1:
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=10000,
                                     value=int(ing.get("chunk_size", 1000)), key="ing_chunk_size")
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=2000,
                                        value=int(ing.get("chunk_overlap", 200)), key="ing_overlap")
    with c2:
        splitter = st.selectbox("Splitter", ["recursive", "semantic", "fixed_length"],
                                index=["recursive", "semantic", "fixed_length"].index(ing.get("splitter", "recursive")),
                                key="ing_splitter")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=1000,
                                     value=int(ing.get("batch_size", 100)), key="ing_batch")

    return {**ing, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
            "splitter": splitter, "batch_size": batch_size}


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.header(t("overview.title"))

    # ── Editable component configuration ──────────────────────────
    st.subheader(t("overview.component_config"))

    try:
        cfg = _load_yaml()
    except Exception as exc:
        st.error(t("overview.config_error", exc))
        return

    with st.form("config_form"):
        tab_llm, tab_emb, tab_ret, tab_ing = st.tabs(["LLM", "Embedding", "检索 & 重排", "摄取参数"])

        with tab_llm:
            new_llm = _render_llm_editor(cfg)
        with tab_emb:
            new_emb = _render_embedding_editor(cfg)
        with tab_ret:
            new_ret, new_rerank = _render_retrieval_editor(cfg)
        with tab_ing:
            new_ing = _render_ingestion_editor(cfg)

        submitted = st.form_submit_button("💾 保存配置", type="primary")

    if submitted:
        try:
            cfg["llm"] = new_llm
            cfg["embedding"] = new_emb
            cfg["retrieval"] = new_ret
            cfg["rerank"] = new_rerank
            cfg["ingestion"] = new_ing
            _save_yaml(cfg)
            st.success("✅ 配置已保存到 config/settings.yaml")
            st.rerun()
        except Exception as exc:
            st.error(f"保存失败：{exc}")

    # ── Collection statistics ──────────────────────────────────────
    st.subheader(t("overview.collection_stats"))

    stats = _safe_collection_stats()
    if stats:
        stat_cols = st.columns(min(len(stats), 4))
        for idx, (name, info) in enumerate(sorted(stats.items())):
            with stat_cols[idx % len(stat_cols)]:
                count = info.get("chunk_count", "?")
                st.metric(label=name, value=count)
                if count == 0 or count == "?":
                    st.caption(t("overview.empty_collection"))
    else:
        st.warning(t("overview.no_collections"))

    # ── Trace file statistics ──────────────────────────────────────
    st.subheader(t("overview.trace_stats"))

    traces_path = resolve_path("logs/traces.jsonl")
    if traces_path.exists():
        line_count = sum(1 for _ in traces_path.open(encoding="utf-8"))
        if line_count > 0:
            st.metric(t("overview.total_traces"), line_count)
        else:
            st.info(t("overview.no_traces"))
    else:
        st.info(t("overview.no_traces"))
