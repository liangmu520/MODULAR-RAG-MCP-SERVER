"""Data Browser page - browse ingested documents, chunks, and images."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.i18n import t
from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """Render the Data Browser page."""
    st.header(t("data_browser.title"))

    try:
        svc = DataService()
    except Exception as exc:
        st.error(t("data_browser.init_error", exc))
        return

    # Collection selector
    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")
    collection = st.selectbox(
        t("data_browser.collection"),
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    # Danger zone
    st.divider()
    with st.expander(t("data_browser.danger_zone"), expanded=False):
        st.warning(t("data_browser.danger_warning"))
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button(t("data_browser.clear_all"), type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error(t("data_browser.confirm_clear"))
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button(t("data_browser.yes_delete"), key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(t("data_browser.cleared_errors", len(result["errors"]), "; ".join(result["errors"])))
                    else:
                        st.success(t("data_browser.cleared_success", result["collections_deleted"]))
                    st.rerun()
            with c2:
                if st.button(t("data_browser.cancel"), key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    # Document list
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(t("data_browser.load_error", exc))
        return

    if not docs:
        st.info(t("data_browser.no_docs"))
        return

    st.subheader(t("data_browser.documents", len(docs)))

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = (
            f"📑 {source_name}  —  "
            f"{doc['chunk_count']} {t('data_browser.chunks')} · "
            f"{doc['image_count']} {t('data_browser.images')}"
        )
        with st.expander(label, expanded=(len(docs) == 1)):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(t("data_browser.chunks"), doc["chunk_count"])
            col_b.metric(t("data_browser.images"), doc["image_count"])
            col_c.metric(t("data_browser.collection"), doc.get("collection", "-"))
            st.caption(
                f"{t('data_browser.source')} {doc['source_path']}  ·  "
                f"{t('data_browser.hash')} `{doc['source_hash'][:16]}...`  ·  "
                f"{t('data_browser.processed')} {doc.get('processed_at', '-')}"
            )

            st.divider()

            # Chunk cards
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(t("data_browser.chunk_section", len(chunks)))
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    with st.container(border=True):
                        st.markdown(f"**Chunk {cidx + 1}** · `{chunk_id[-16:]}` · {len(text)} chars")
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            t("data_browser.content"),
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        with st.expander(t("data_browser.metadata"), expanded=False):
                            st.json(meta)
            else:
                st.caption(t("data_browser.no_chunks"))

            # Image preview
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(t("data_browser.image_section", len(images)))
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(t("data_browser.image_missing", img["image_id"]))
