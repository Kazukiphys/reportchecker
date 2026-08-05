from __future__ import annotations

import base64
from io import BytesIO
import pickle
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

from plagiarism_checker import (
    Document,
    PairResult,
    analyze_documents,
    build_pair_annotated_pdfs,
    extract_text_from_pdf,
    is_pdf_annotation_available,
)


st.set_page_config(page_title="レポート一致チェック", page_icon="📄", layout="wide")


@dataclass
class AnalysisSession:
    created_at: datetime
    documents: List[Document]
    pdf_bytes_list: List[bytes]
    failed_files: List[str]
    results: List[PairResult]
    annotated_cache: Dict[int, Tuple[bytes, bytes]] = field(default_factory=dict)


SESSION_TTL_HOURS = 12
_ANALYSIS_STORE: Dict[str, AnalysisSession] = {}
_ANALYSIS_LOCK = threading.Lock()
_ANALYSIS_CACHE_DIR = Path(".streamlit") / "analysis_cache"


def _analysis_cache_path(analysis_id: str) -> Path:
    return _ANALYSIS_CACHE_DIR / f"{analysis_id}.pkl"


def _cleanup_expired_analysis_sessions() -> None:
    threshold = datetime.now() - timedelta(hours=SESSION_TTL_HOURS)
    with _ANALYSIS_LOCK:
        expired = [key for key, value in _ANALYSIS_STORE.items() if value.created_at < threshold]
        for key in expired:
            del _ANALYSIS_STORE[key]


def _store_analysis_session(session: AnalysisSession) -> str:
    analysis_id = uuid.uuid4().hex
    _cleanup_expired_analysis_sessions()
    with _ANALYSIS_LOCK:
        _ANALYSIS_STORE[analysis_id] = session
    _ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _analysis_cache_path(analysis_id).open("wb") as fp:
        pickle.dump(session, fp)
    return analysis_id


def _get_analysis_session(analysis_id: str | None) -> AnalysisSession | None:
    if not analysis_id:
        return None
    _cleanup_expired_analysis_sessions()
    with _ANALYSIS_LOCK:
        session = _ANALYSIS_STORE.get(analysis_id)
    if session is not None:
        return session

    cache_path = _analysis_cache_path(analysis_id)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as fp:
            session = pickle.load(fp)
    except Exception:
        return None

    if isinstance(session, AnalysisSession):
        with _ANALYSIS_LOCK:
            _ANALYSIS_STORE[analysis_id] = session
        return session

    return None


def _clear_detail_query() -> None:
    try:
        st.query_params.clear()
    except Exception:
        st.query_params["analysis"] = ""
        st.query_params["pair"] = ""
    st.session_state.pop("detail_analysis_id", None)
    st.session_state.pop("detail_pair_idx", None)


def _set_detail_query(analysis_id: str, pair_idx: int) -> None:
    st.query_params["analysis"] = analysis_id
    st.query_params["pair"] = str(pair_idx)


def _activate_detail_view(analysis_id: str, pair_idx: int) -> None:
    st.session_state["detail_analysis_id"] = analysis_id
    st.session_state["detail_pair_idx"] = pair_idx
    # Streamlit Cloudではquery_params遷移が不安定なケースがあるため、
    # 詳細遷移はsession_stateを一次ソースとして扱う。


def _render_pdf_panel(pdf_bytes: bytes, title: str, height: int = 820) -> None:
    st.caption(title)
    # PDFブラウザ埋め込みは環境依存で空表示になるため、
    # プレビューをこのパネル内へ直接描画する。
    st.download_button(
        "PDFをダウンロード",
        data=pdf_bytes,
        file_name="annotated_preview.pdf",
        mime="application/pdf",
        key=f"download_{title}_{len(pdf_bytes)}",
    )

    if pdfium is None:
        st.info("プレビュー画像を生成できないため、PDFダウンロードで確認してください。")
        return

    try:
        pdf = pdfium.PdfDocument(BytesIO(pdf_bytes))
        preview_pages = min(80, len(pdf))
        if preview_pages <= 0:
            return

        st.caption("PDFプレビュー（パネル内スクロール）")
        image_blocks: list[str] = []
        for page_idx in range(preview_pages):
            page = pdf[page_idx]
            bitmap = page.render(scale=1.5, rotation=0)
            pil_image = bitmap.to_pil()
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="PNG")
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode("ascii")
            image_blocks.append(
                "<div style='margin:0 0 12px 0;'>"
                f"<div style='font-size:12px;color:#666;margin:0 0 4px 0;'>ページ {page_idx + 1}</div>"
                f"<img src='data:image/png;base64,{img_b64}' style='width:100%;height:auto;border:1px solid #eee;border-radius:6px;'/>"
                "</div>"
            )

        scroller_html = (
            "<div style='height:"
            f"{height}px;"
            "overflow-y:auto;padding:8px;border:1px solid #ddd;border-radius:8px;background:#fff;'>"
            + "".join(image_blocks)
            + "</div>"
        )
        components.html(scroller_html, height=height + 12)

        if len(pdf) > preview_pages:
            st.info(f"ページ数が多いため先頭{preview_pages}ページのみ表示しています。必要ならPDFをダウンロードして確認してください。")
    except Exception:
        st.info("プレビュー生成に失敗しました。PDFダウンロードで確認してください。")


def _render_detail_view() -> bool:
    pair_value = st.session_state.get("detail_pair_idx")
    if pair_value is None:
        pair_value = st.query_params.get("pair")

    analysis_value = (
        st.session_state.get("detail_analysis_id")
        or st.session_state.get("analysis_id")
        or st.query_params.get("analysis")
    )

    if analysis_value is None or pair_value is None or pair_value == "":
        return False

    session = st.session_state.get("analysis_session")
    if session is None:
        session = _get_analysis_session(analysis_value)

    if session is None:
        st.warning("比較結果の保存先が見つかりません。もう一度比較を実行してください。")
        if st.button("比較画面へ戻る"):
            _clear_detail_query()
            st.rerun()
        return True

    try:
        pair_idx = int(pair_value)
    except ValueError:
        st.error("詳細指定が不正です。")
        if st.button("比較画面へ戻る"):
            _clear_detail_query()
            st.rerun()
        return True

    if pair_idx < 0 or pair_idx >= len(session.results):
        st.error("指定した詳細が見つかりません。")
        if st.button("比較画面へ戻る"):
            _clear_detail_query()
            st.rerun()
        return True

    pair = session.results[pair_idx]
    docs = session.documents
    pdf_bytes = session.pdf_bytes_list

    st.title("一致箇所の詳細ビュー")
    st.caption(f"{docs[pair.idx_a].name} × {docs[pair.idx_b].name} / 一致率: {pair.similarity * 100:.2f}%")

    if st.button("比較結果へ戻る"):
        _clear_detail_query()
        st.rerun()

    if pair_idx not in session.annotated_cache:
        src_a = pdf_bytes[pair.idx_a]
        src_b = pdf_bytes[pair.idx_b]
        if is_pdf_annotation_available():
            ann_a, ann_b = build_pair_annotated_pdfs(pair, src_a, src_b)
        else:
            ann_a, ann_b = src_a, src_b
        session.annotated_cache[pair_idx] = (ann_a, ann_b)

    ann_a, ann_b = session.annotated_cache[pair_idx]

    if not is_pdf_annotation_available():
        st.warning("PDF注釈ライブラリ未導入のため，ハイライトなしで表示しています。")

    col_l, col_r = st.columns(2)
    with col_l:
        _render_pdf_panel(ann_a, f"左: {docs[pair.idx_a].name}")
    with col_r:
        _render_pdf_panel(ann_b, f"右: {docs[pair.idx_b].name}")

    st.subheader("一致候補テキスト")
    st.markdown("**同一文候補（ほぼ一致）**")
    if pair.exact_matches:
        for i, text in enumerate(pair.exact_matches[:12], start=1):
            st.markdown(f"{i}. {text}")
    else:
        st.write("該当なし")

    st.markdown("**類似文候補（文単位）**")
    if pair.near_matches:
        for i, m in enumerate(pair.near_matches[:10], start=1):
            st.markdown(f"{i}. 類似度: {m.score * 100:.2f}%")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("左PDF")
                st.write(m.sentence_a)
            with c2:
                st.caption("右PDF")
                st.write(m.sentence_b)
    else:
        st.write("該当なし")

    return True


if _render_detail_view():
    st.stop()


st.title("レポート一致チェック（校内利用向け）")
st.caption("複数PDFを比較して，一致率の高い組み合わせと一致箇所を確認できます。")

with st.expander("使い方", expanded=False):
    st.markdown(
        """
1. 生徒レポートのPDFをまとめてアップロードします。
2. 一致率のしきい値を調整して「比較を実行」を押します。
3. 一覧から一致率が高いペアを開くと，同一文・類似文の候補を確認できます。

注意:
- OCRを有効にすると，画像だけのPDF（スキャンPDF）も比較対象にできます。
- この結果はあくまで判定補助です。最終判断は教員が行ってください。
        """
    )

uploaded_files = st.file_uploader(
    "PDFファイルを選択（複数可）",
    type=["pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)
with col1:
    suspicious_threshold = st.slider(
        "一致率しきい値（これ以上を表示）",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.01,
    )
with col2:
    near_match_threshold = st.slider(
        "類似文しきい値（文単位）",
        min_value=0.50,
        max_value=0.98,
        value=0.72,
        step=0.01,
    )

run = st.button("比較を実行", type="primary")

if run:
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("比較には2件以上のPDFが必要です。")
        st.stop()

    documents: list[Document] = []
    pdf_bytes_list: list[bytes] = []
    failed_files: list[str] = []

    with st.spinner("PDFからテキストを抽出しています..."):
        for f in uploaded_files:
            try:
                file_bytes = f.read()
                text, ocr_used = extract_text_from_pdf(file_bytes, use_ocr_fallback=True)
                if not text.strip():
                    failed_files.append(f.name)
                    continue
                documents.append(Document(name=f.name, text=text, ocr_used=ocr_used))
                pdf_bytes_list.append(file_bytes)
            except Exception:
                failed_files.append(f.name)

    if len(documents) < 2:
        st.error("有効なテキストを抽出できたPDFが2件未満です。")
        if failed_files:
            st.warning("抽出失敗: " + ", ".join(failed_files))
        st.stop()

    with st.spinner("レポート同士を比較しています..."):
        results = analyze_documents(
            documents,
            suspicious_threshold=suspicious_threshold,
            near_match_threshold=near_match_threshold,
        )

    analysis_session = AnalysisSession(
        created_at=datetime.now(),
        documents=documents,
        pdf_bytes_list=pdf_bytes_list,
        failed_files=failed_files,
        results=results,
    )
    analysis_id = _store_analysis_session(analysis_session)

    st.session_state["analysis_id"] = analysis_id
    st.session_state["analysis_session"] = analysis_session
    st.session_state.pop("detail_pair_idx", None)
    st.session_state.pop("detail_analysis_id", None)


analysis_session = st.session_state.get("analysis_session")
if analysis_session is None:
    analysis_id_from_state = st.session_state.get("analysis_id")
    if analysis_id_from_state:
        restored = _get_analysis_session(analysis_id_from_state)
        if restored is not None:
            st.session_state["analysis_session"] = restored
            analysis_session = restored

if analysis_session is not None:
    documents = analysis_session.documents
    failed_files = analysis_session.failed_files
    results = analysis_session.results
    analysis_id = st.session_state.get("analysis_id")

    st.subheader("比較結果")
    st.write(f"比較対象: {len(documents)}件 / ペア数: {len(documents) * (len(documents) - 1) // 2}件")

    ocr_count = sum(1 for d in documents if d.ocr_used)
    st.write(f"OCR利用: {ocr_count}件")

    with st.expander("抽出したファイル情報", expanded=False):
        student_df = pd.DataFrame(
            [
                {
                    "レポート": d.name,
                    "OCR利用": "はい" if d.ocr_used else "いいえ",
                }
                for d in documents
            ]
        )
        st.dataframe(student_df, use_container_width=True, hide_index=True)

    if failed_files:
        st.warning("次のPDFは抽出に失敗したか，テキストが空でした: " + ", ".join(failed_files))

    if not results:
        st.success("しきい値以上の疑わしいペアは見つかりませんでした。")
        st.stop()

    table = pd.DataFrame(
        [
            {
                "レポートA": r.name_a,
                "レポートB": r.name_b,
                "一致率": round(r.similarity * 100, 2),
                "同一文候補数": len(r.exact_matches),
                "類似文候補数": len(r.near_matches),
            }
            for r in results
        ]
    )

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "結果をCSVで保存",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="plagiarism_check_results.csv",
        mime="text/csv",
    )

    st.subheader("詳細（一致箇所）")
    st.caption("各行の「詳細を開く」を押すと，別ビューで左右PDFと一致箇所のマーカーを表示します。")
    for idx, r in enumerate(results):
        row_col1, row_col2 = st.columns([5, 1])
        with row_col1:
            st.write(f"{idx + 1}. {r.name_a} × {r.name_b} / 一致率: {r.similarity * 100:.2f}%")
        with row_col2:
            if st.button("詳細を開く", key=f"open_detail_{idx}"):
                if analysis_id:
                    _activate_detail_view(analysis_id, idx)
                    st.rerun()
