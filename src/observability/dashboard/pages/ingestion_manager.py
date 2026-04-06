"""Ingestion Manager page – upload files, trigger ingestion, delete documents.

Layout:
1. File uploader + collection selector
2. Ingest button → progress bar (using on_progress callback)
3. Document list with delete buttons
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.i18n import t
from src.observability.dashboard.services.data_service import DataService


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """Save the uploaded file to a temp location and run the pipeline."""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()

    # Write uploaded file to a temp location
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": t("ingestion.stage_integrity"),
        "load": t("ingestion.stage_load"),
        "split": t("ingestion.stage_split"),
        "transform": t("ingestion.stage_transform"),
        "embed": t("ingestion.stage_embed"),
        "upsert": t("ingestion.stage_upsert"),
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection)
        pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        progress_bar.progress(1.0, text=t("ingestion.complete"))
        status_text.success(t("ingestion.success", uploaded_file.name, collection))
    except Exception as exc:
        status_text.error(t("ingestion.failed", exc))
    finally:
        TraceCollector().collect(trace)
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header(t("ingestion.title"))

    # ── Upload section ─────────────────────────────────────────────
    st.subheader(t("ingestion.upload_section"))

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            t("ingestion.select_file"),
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
        )
    with col2:
        collection = st.text_input(t("ingestion.collection"), value="default", key="ingest_collection")

    if uploaded is not None:
        if st.button(t("ingestion.start_btn"), key="btn_ingest"):
            progress_bar = st.progress(0, text=t("ingestion.preparing"))
            status_text = st.empty()
            _run_ingestion(uploaded, collection.strip() or "default", progress_bar, status_text)

    st.divider()

    # ── Document management section ────────────────────────────────
    st.subheader(t("ingestion.manage_section"))

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(t("ingestion.load_error", exc))
        return

    if not docs:
        st.info(t("ingestion.no_docs"))
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** — "
                f"{t('ingestion.collection_label')}: `{doc.get('collection', '—')}` | "
                f"{t('ingestion.chunks_label')}: {doc['chunk_count']} | "
                f"{t('ingestion.images_label')}: {doc['image_count']}"
            )
        with col_btn:
            if st.button(t("ingestion.delete_btn"), key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(t("ingestion.delete_success", result.chunks_deleted, result.images_deleted))
                        st.rerun()
                    else:
                        st.warning(t("ingestion.delete_partial", result.errors))
                except Exception as exc:
                    st.error(t("ingestion.delete_failed", exc))
