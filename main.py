from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
import csv
import threading
import uuid
from typing import Dict, List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from plagiarism_checker import (
    Document,
    PairResult,
    analyze_documents,
    build_pair_annotated_pdfs,
    extract_text_from_pdf,
    is_ocr_available,
    is_pdf_annotation_available,
)


@dataclass
class AnalysisSession:
    created_at: datetime
    documents: List[Document]
    pdf_bytes_list: List[bytes]
    failed_files: List[str]
    results: List[PairResult]
    annotated_cache: Dict[int, Tuple[bytes, bytes]] = field(default_factory=dict)


APP_NAME = "レポート一致チェック"
SESSION_TTL_HOURS = 12

app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_store: Dict[str, AnalysisSession] = {}
_store_lock = threading.Lock()


def _cleanup_expired() -> None:
    threshold = datetime.now() - timedelta(hours=SESSION_TTL_HOURS)
    with _store_lock:
        expired = [key for key, value in _store.items() if value.created_at < threshold]
        for key in expired:
            del _store[key]


def _get_session_or_404(analysis_id: str) -> AnalysisSession:
    _cleanup_expired()
    with _store_lock:
        session = _store.get(analysis_id)
    if not session:
        raise HTTPException(status_code=404, detail="分析セッションが見つかりません。再度比較を実行してください。")
    return session


def _build_csv(rows: List[dict]) -> str:
    if not rows:
        return ""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _result_rows(results: List[PairResult]) -> List[dict]:
    rows = []
    for i, r in enumerate(results):
        rows.append(
            {
                "idx": i,
                "レポートA": r.name_a,
                "レポートB": r.name_b,
                "一致率": round(r.similarity * 100, 2),
                "同一文候補数": len(r.exact_matches),
                "類似文候補数": len(r.near_matches),
            }
        )
    return rows


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": APP_NAME,
            "ocr_available": is_ocr_available(),
        },
    )


@app.post("/analyze")
async def analyze(
    request: Request,
    files: List[UploadFile] = File(...),
    suspicious_threshold: float = Form(0.35),
    near_match_threshold: float = Form(0.72),
) -> Response:
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="比較には2件以上のPDFが必要です。")

    documents: List[Document] = []
    pdf_bytes_list: List[bytes] = []
    failed_files: List[str] = []

    valid_uploads: List[Tuple[str, bytes]] = []
    for upload in files:
        try:
            if not upload.filename or not upload.filename.lower().endswith(".pdf"):
                failed_files.append(upload.filename or "unknown")
                continue
            file_bytes = await upload.read()
            valid_uploads.append((upload.filename, file_bytes))
        except Exception:
            failed_files.append(upload.filename or "unknown")

    for filename, file_bytes in valid_uploads:
        try:
            text, ocr_used = extract_text_from_pdf(file_bytes, use_ocr_fallback=True)
            if not text.strip():
                failed_files.append(filename)
                continue

            documents.append(
                Document(
                    name=filename,
                    text=text,
                    ocr_used=ocr_used,
                )
            )
            pdf_bytes_list.append(file_bytes)
        except Exception:
            failed_files.append(filename)

    if len(documents) < 2:
        raise HTTPException(status_code=400, detail="有効なテキストを抽出できたPDFが2件未満です。")

    results = analyze_documents(
        documents,
        suspicious_threshold=suspicious_threshold,
        near_match_threshold=near_match_threshold,
    )

    analysis_id = uuid.uuid4().hex
    session = AnalysisSession(
        created_at=datetime.now(),
        documents=documents,
        pdf_bytes_list=pdf_bytes_list,
        failed_files=failed_files,
        results=results,
    )

    _cleanup_expired()
    with _store_lock:
        _store[analysis_id] = session

    return RedirectResponse(url=f"/analysis/{analysis_id}", status_code=303)


@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
def analysis_result(request: Request, analysis_id: str) -> HTMLResponse:
    session = _get_session_or_404(analysis_id)

    rows = _result_rows(session.results)
    ocr_count = sum(1 for d in session.documents if d.ocr_used)

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "title": APP_NAME,
            "analysis_id": analysis_id,
            "rows": rows,
            "documents": session.documents,
            "failed_files": session.failed_files,
            "ocr_count": ocr_count,
            "pair_count": len(session.documents) * (len(session.documents) - 1) // 2,
            "results_count": len(session.results),
        },
    )


@app.get("/analysis/{analysis_id}/pair/{pair_idx}", response_class=HTMLResponse)
def pair_detail(request: Request, analysis_id: str, pair_idx: int) -> HTMLResponse:
    session = _get_session_or_404(analysis_id)

    if pair_idx < 0 or pair_idx >= len(session.results):
        raise HTTPException(status_code=404, detail="ペア詳細が見つかりません。")

    pair = session.results[pair_idx]
    doc_a = session.documents[pair.idx_a]
    doc_b = session.documents[pair.idx_b]

    return templates.TemplateResponse(
        request=request,
        name="detail.html",
        context={
            "title": APP_NAME,
            "analysis_id": analysis_id,
            "pair_idx": pair_idx,
            "pair": pair,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "pdf_annotation_available": is_pdf_annotation_available(),
        },
    )


@app.get("/analysis/{analysis_id}/pair/{pair_idx}/pdf/{side}")
def pair_pdf(analysis_id: str, pair_idx: int, side: str) -> Response:
    session = _get_session_or_404(analysis_id)

    if pair_idx < 0 or pair_idx >= len(session.results):
        raise HTTPException(status_code=404, detail="ペア詳細が見つかりません。")

    if side not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="sideはleftまたはrightを指定してください。")

    pair = session.results[pair_idx]
    if pair_idx not in session.annotated_cache:
        src_a = session.pdf_bytes_list[pair.idx_a]
        src_b = session.pdf_bytes_list[pair.idx_b]
        if is_pdf_annotation_available():
            ann_a, ann_b = build_pair_annotated_pdfs(pair, src_a, src_b)
        else:
            ann_a, ann_b = src_a, src_b
        session.annotated_cache[pair_idx] = (ann_a, ann_b)

    ann_a, ann_b = session.annotated_cache[pair_idx]
    payload = ann_a if side == "left" else ann_b
    return Response(content=payload, media_type="application/pdf")


@app.get("/analysis/{analysis_id}/results.csv")
def export_csv(analysis_id: str) -> Response:
    session = _get_session_or_404(analysis_id)
    rows = _result_rows(session.results)
    csv_text = _build_csv(rows)
    headers = {"Content-Disposition": "attachment; filename=plagiarism_check_results.csv"}
    return Response(content=csv_text.encode("utf-8-sig"), media_type="text/csv", headers=headers)
