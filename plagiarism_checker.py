from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
import re
import unicodedata
from typing import Iterable, List, Tuple

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


@dataclass
class Document:
    name: str
    text: str
    ocr_used: bool = False


@dataclass
class SentenceMatch:
    sentence_a: str
    sentence_b: str
    score: float


@dataclass
class PairResult:
    idx_a: int
    idx_b: int
    name_a: str
    name_b: str
    similarity: float
    exact_matches: List[str]
    near_matches: List[SentenceMatch]


def _extract_text_from_pdf_native(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file in memory using PDF text layer."""
    reader = PdfReader(BytesIO(file_bytes))
    chunks: List[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _ocr_pdf(file_bytes: bytes, language_hint: str = "japanese") -> str:
    """Run OCR for each PDF page by rendering page images with pypdfium2."""
    if pdfium is None or RapidOCR is None:
        return ""

    pdf = pdfium.PdfDocument(BytesIO(file_bytes))
    engine = RapidOCR()
    chunks: List[str] = []

    for page in pdf:
        bitmap = page.render(scale=2.0, rotation=0)
        pil_img = bitmap.to_pil()
        img = np.array(pil_img.convert("RGB"))
        # rapidocr returns tuple(result, elapse). result can be None.
        result, _ = engine(img)
        if not result:
            continue
        page_text = "\n".join([line[1] for line in result if len(line) >= 2 and line[1]])
        if page_text.strip():
            chunks.append(page_text)

    return "\n".join(chunks)


def is_ocr_available() -> bool:
    return pdfium is not None and RapidOCR is not None


def is_pdf_annotation_available() -> bool:
    return fitz is not None


def _normalize_marker_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _marker_fragments(marker: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[。．.!?！？、，,;；:：\n]+", marker) if p and p.strip()]
    fragments: List[str] = []
    seen: set[str] = set()

    for part in parts:
        normalized = _normalize_marker_text(part)
        if len(normalized) < 6 or normalized in seen:
            continue
        seen.add(normalized)
        fragments.append(normalized)

    if not fragments:
        normalized = _normalize_marker_text(marker)
        if len(normalized) >= 6:
            fragments.append(normalized)

    # Short contiguous fallbacks help when the PDF text layer splits lines or words.
    normalized = _normalize_marker_text(marker)
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) >= 8:
        fallback_lengths = (18, 12, 8)
        for length in fallback_lengths:
            if len(compact) < length:
                continue
            start_positions = [0, max(0, (len(compact) - length) // 2), len(compact) - length]
            for start in start_positions:
                fragment = compact[start : start + length]
                if fragment and fragment not in seen:
                    seen.add(fragment)
                    fragments.append(fragment)

    return fragments


def _shared_fragments(text_a: str, text_b: str, min_len: int = 10, max_items: int = 5) -> List[str]:
    a = re.sub(r"\s+", "", _normalize_marker_text(text_a))
    b = re.sub(r"\s+", "", _normalize_marker_text(text_b))
    if len(a) < min_len or len(b) < min_len:
        return []

    matcher = SequenceMatcher(None, a, b)
    blocks = [blk for blk in matcher.get_matching_blocks() if blk.size >= min_len]
    blocks.sort(key=lambda blk: blk.size, reverse=True)

    fragments: List[str] = []
    seen: set[str] = set()
    for blk in blocks:
        fragment = a[blk.a : blk.a + blk.size]
        if fragment in seen:
            continue
        seen.add(fragment)
        fragments.append(fragment)
        if len(fragments) >= max_items:
            break
    return fragments


def _annotate_pdf_bytes(
    file_bytes: bytes,
    markers: List[str],
    color: Tuple[float, float, float],
    max_markers: int = 60,
    marked_regions: set[tuple[int, float, float, float, float]] | None = None,
) -> bytes:
    if fitz is None or not markers:
        return file_bytes

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if marked_regions is None:
        marked_regions = set()
    unique: List[str] = []
    seen: set[str] = set()
    for marker in markers:
        normalized = _normalize_marker_text(marker)
        if len(normalized) < 10:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
        if len(unique) >= max_markers:
            break

    for marker in unique:
        fragments = _marker_fragments(marker)
        for page_index, page in enumerate(doc):
            for fragment in fragments:
                areas = page.search_for(fragment)
                if not areas and len(fragment) > 24:
                    areas = page.search_for(fragment[:24])
                if not areas and len(fragment) > 16:
                    areas = page.search_for(fragment[:16])
                if not areas:
                    continue

                try:
                    filtered_areas = []
                    for area in areas:
                        key = (
                            page_index,
                            round(area.x0, 1),
                            round(area.y0, 1),
                            round(area.x1, 1),
                            round(area.y1, 1),
                        )
                        if key in marked_regions:
                            continue
                        marked_regions.add(key)
                        filtered_areas.append(area)

                    if not filtered_areas:
                        continue

                    annot = page.add_highlight_annot(filtered_areas)
                    annot.set_colors(stroke=color)
                    try:
                        annot.set_opacity(0.3)
                    except Exception:
                        pass
                    annot.update()
                except Exception:
                    pass

                border_color = tuple(max(0.0, c - 0.45) for c in color)
                for area in filtered_areas:
                    highlight = fitz.Rect(area.x0 - 1.2, area.y0 - 0.6, area.x1 + 1.2, area.y1 + 0.6)
                    page.draw_rect(
                        highlight,
                        color=border_color,
                        fill=None,
                        width=1.6,
                        overlay=True,
                    )
                    page.draw_rect(
                        highlight,
                        color=color,
                        fill=color,
                        width=1.2,
                        overlay=True,
                        fill_opacity=0.22,
                        stroke_opacity=1.0,
                    )

    output = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return output


def build_pair_annotated_pdfs(pair: PairResult, pdf_a: bytes, pdf_b: bytes) -> Tuple[bytes, bytes]:
    exact_shared: List[str] = [s for s in pair.exact_matches]
    near_shared: List[str] = []
    for m in pair.near_matches:
        near_shared.extend(_shared_fragments(m.sentence_a, m.sentence_b))

    combined_shared: List[str] = []
    seen: set[str] = set()
    for marker in exact_shared + near_shared:
        key = _normalize_marker_text(marker)
        if len(key) < 6 or key in seen:
            continue
        seen.add(key)
        combined_shared.append(marker)

    marked_a: set[tuple[int, float, float, float, float]] = set()
    ann_a = _annotate_pdf_bytes(pdf_a, combined_shared, color=(1.0, 0.88, 0.15), marked_regions=marked_a)

    marked_b: set[tuple[int, float, float, float, float]] = set()
    ann_b = _annotate_pdf_bytes(pdf_b, combined_shared, color=(1.0, 0.88, 0.15), marked_regions=marked_b)

    return ann_a, ann_b


def extract_text_from_pdf(
    file_bytes: bytes,
    use_ocr_fallback: bool = True,
    min_text_chars_for_no_ocr: int = 80,
) -> Tuple[str, bool]:
    """Extract text from PDF and optionally fallback to OCR for image PDFs.

    Returns:
        (text, ocr_used)
    """
    native_text = _extract_text_from_pdf_native(file_bytes)
    cleaned_native = native_text.strip()

    if not use_ocr_fallback:
        return cleaned_native, False

    if len(cleaned_native) >= min_text_chars_for_no_ocr:
        return cleaned_native, False

    ocr_text = _ocr_pdf(file_bytes)
    combined = "\n".join([t for t in [cleaned_native, ocr_text.strip()] if t])
    return combined.strip(), bool(ocr_text.strip())




def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    normalized_newline = re.sub(r"\r\n?|\u2028", "\n", text)
    # Flatten PDF layout noise while preserving bullet/list content.
    raw_lines = [line.strip() for line in normalized_newline.split("\n")]
    bullet_prefix = re.compile(
        r"^(?:[\-\*•●◦▪□■◆◇▶▷►]|\(?\d{1,3}[\.)]|[①-⑳]|[A-Za-z]\.|[（(]\d+[）)]|[ア-ンｱ-ﾝ][\.)、])\s*"
    )

    merged: List[str] = []
    current = ""

    for line in raw_lines:
        if not line:
            if current:
                merged.append(current.strip())
                current = ""
            continue

        cleaned = bullet_prefix.sub("", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue

        if current:
            current = f"{current} {cleaned}".strip()
        else:
            current = cleaned

        if re.search(r"[。．.!?！？]$", cleaned):
            merged.append(current.strip())
            current = ""

    if current:
        merged.append(current.strip())

    rough: List[str] = []
    for block in merged:
        rough.extend(re.split(r"(?<=[。．.!?！？])\s+", block))

    return [s.strip() for s in rough if s and len(s.strip()) >= 8]


def sentence_key(sentence: str) -> str:
    s = normalize_text(sentence)
    # Keep alphanumerics and Japanese scripts, drop punctuation to improve exact-hit recall.
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龯ー]", "", s)


def compute_similarity_matrix(docs: List[Document]) -> np.ndarray:
    if not docs:
        return np.zeros((0, 0), dtype=float)

    corpus = [normalize_text(doc.text) for doc in docs]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    matrix = cosine_similarity(tfidf)
    return matrix


def find_exact_sentence_matches(
    doc_a: Document,
    doc_b: Document,
    max_items: int = 20,
) -> List[str]:
    sent_a = split_sentences(doc_a.text)
    sent_b = split_sentences(doc_b.text)

    map_a: Dict[str, str] = {}
    for s in sent_a:
        k = sentence_key(s)
        if len(k) >= 18 and k not in map_a:
            map_a[k] = s

    map_b: Dict[str, str] = {}
    for s in sent_b:
        k = sentence_key(s)
        if len(k) >= 18 and k not in map_b:
            map_b[k] = s

    common_keys = set(map_a.keys()) & set(map_b.keys())
    common = sorted((map_a[k] for k in common_keys), key=len, reverse=True)
    return common[:max_items]


def _tfidf_sentence_pairs(
    sent_a: List[str],
    sent_b: List[str],
    threshold: float,
    max_items: int,
) -> Iterable[SentenceMatch]:
    if not sent_a or not sent_b:
        return []

    combined = [normalize_text(s) for s in sent_a + sent_b]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    tfidf = vec.fit_transform(combined)

    a_vec = tfidf[: len(sent_a)]
    b_vec = tfidf[len(sent_a) :]
    sim = cosine_similarity(a_vec, b_vec)

    pairs: List[SentenceMatch] = []
    for i in range(sim.shape[0]):
        j = int(sim[i].argmax())
        score = float(sim[i, j])
        if score >= threshold:
            pairs.append(SentenceMatch(sentence_a=sent_a[i], sentence_b=sent_b[j], score=score))

    pairs.sort(key=lambda x: x.score, reverse=True)

    unique: List[SentenceMatch] = []
    seen_b: set[str] = set()
    for p in pairs:
        key_b = sentence_key(p.sentence_b)
        if key_b in seen_b:
            continue
        seen_b.add(key_b)
        unique.append(p)
        if len(unique) >= max_items:
            break

    return unique


def find_near_sentence_matches(
    doc_a: Document,
    doc_b: Document,
    threshold: float = 0.72,
    max_items: int = 10,
) -> List[SentenceMatch]:
    sent_a = [s for s in split_sentences(doc_a.text) if len(sentence_key(s)) >= 20]
    sent_b = [s for s in split_sentences(doc_b.text) if len(sentence_key(s)) >= 20]
    return list(_tfidf_sentence_pairs(sent_a, sent_b, threshold=threshold, max_items=max_items))


def analyze_documents(
    docs: List[Document],
    suspicious_threshold: float = 0.35,
    near_match_threshold: float = 0.72,
) -> List[PairResult]:
    matrix = compute_similarity_matrix(docs)
    results: List[PairResult] = []

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = float(matrix[i, j])
            if sim < suspicious_threshold:
                continue

            exact = find_exact_sentence_matches(docs[i], docs[j])
            near = find_near_sentence_matches(docs[i], docs[j], threshold=near_match_threshold)

            results.append(
                PairResult(
                    idx_a=i,
                    idx_b=j,
                    name_a=docs[i].name,
                    name_b=docs[j].name,
                    similarity=sim,
                    exact_matches=exact,
                    near_matches=near,
                )
            )

    results.sort(key=lambda x: x.similarity, reverse=True)
    return results
