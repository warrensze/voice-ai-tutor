# Configure offline-only mode BEFORE any other imports
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from pathlib import Path
import argparse
import re
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from local_providers import create_embedding_model
from settings_store import (
    PROJECT_ROOT,
    RAG_SOURCE_MODES,
    SOURCE_ROLES,
    UserSettings,
    load_user_settings,
)

# Data locations
ASSETS_DIR = PROJECT_ROOT / "assets"
DB_LOCATION = str(PROJECT_ROOT / "chrome_langchain_db")
EMBEDDING_MODEL = "mxbai-embed-large"
COLLECTION_NAME = "langchain"
USER_VECTOR_DIR = PROJECT_ROOT / "data" / "vector_stores"
LEGACY_PDF_PATH = ASSETS_DIR / "Grade9GTjoyluckclub.pdf"
EMBED_CHUNK_SIZE = 500
EMBED_CHUNK_OVERLAP = 100
UPSERT_BATCH_SIZE = 1000
OCR_TEXT_SUFFIX = ".ocr.txt"
DEFAULT_MATH_COURSE = "algebra_ii"
DEFAULT_SOURCE_ROLE = "textbook"

COURSE_SOURCE_FILE_ALIASES = {
    ("math", "algebra_ii"): (
        {"source_file": "Algebra-2-Book.pdf", "source_role": "textbook"},
    ),
}

SUBJECT_PDF_PATTERNS = {
    "history": ["*history*.pdf", "*world*.pdf"],
    "chemistry": ["*chem*.pdf", "*chemistry*.pdf"],
    "math": ["*math*.pdf", "*algebra*.pdf", "*geometry*.pdf", "*calculus*.pdf"],
    "english": [
        "*english*.pdf",
        "*literature*.pdf",
        "*grammar*.pdf",
        "*writing*.pdf",
        "*joyluckclub*.pdf",
    ],
}


def _safe_key(value: str) -> str:
    """Return a filesystem-safe key for provider/model-specific vector stores."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "default"


def _settings_or_default(settings: UserSettings | None = None) -> UserSettings:
    return settings if settings is not None else load_user_settings()


def clean_course(value: str | None, *, subject: str | None = "math") -> str:
    cleaned = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "algebra_2": "algebra_ii",
        "algebra_ii": "algebra_ii",
        "alg_2": "algebra_ii",
        "alg_ii": "algebra_ii",
        "pre_calc": "precalculus",
        "precalculus": "precalculus",
        "pre_calculus": "precalculus",
    }
    cleaned = aliases.get(cleaned, cleaned)
    if subject == "math" and cleaned in {"algebra_ii", "precalculus"}:
        return cleaned
    return ""


def clean_source_role(value: str | None, default: str = DEFAULT_SOURCE_ROLE) -> str:
    cleaned = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "core": "textbook",
        "book": "textbook",
        "text": "textbook",
        "practice": "workbook",
        "practice_book": "workbook",
        "exercise": "workbook",
        "exercises": "workbook",
        "test": "exam",
    }
    cleaned = aliases.get(cleaned, cleaned)
    return cleaned if cleaned in SOURCE_ROLES else default


def clean_source_mode(value: str | None) -> str:
    cleaned = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return cleaned if cleaned in RAG_SOURCE_MODES else "auto"


def infer_course_from_filename(file_path: Path, subject: str) -> str:
    filename = file_path.name.lower()
    if subject != "math":
        return ""
    if any(token in filename for token in ("precalc", "pre-calc", "pre_cal", "precalculus")):
        return "precalculus"
    if any(token in filename for token in ("algebra-2", "algebra_2", "algebra 2", "algebra-ii", "algebra_ii", "algebra ii")):
        return "algebra_ii"
    return DEFAULT_MATH_COURSE


def infer_source_role_from_filename(file_path: Path) -> str:
    filename = file_path.name.lower()
    if any(token in filename for token in ("exam", "test", "assessment")):
        return "exam"
    if any(token in filename for token in ("workbook", "practice", "exercise")):
        return "workbook"
    if any(token in filename for token in ("formula", "reference", "table")):
        return "reference"
    if any(token in filename for token in ("note", "notes")):
        return "notes"
    return DEFAULT_SOURCE_ROLE


def source_role_label(value: str | None) -> str:
    role = clean_source_role(value, default="other")
    return role.replace("_", " ")


def course_label(value: str | None) -> str:
    course = clean_course(value)
    labels = {
        "algebra_ii": "Algebra II",
        "precalculus": "Precalculus",
    }
    return labels.get(course, "")


def embedding_model_name(settings: UserSettings | None = None) -> str:
    """Return the active embedding model name."""
    active = _settings_or_default(settings)
    if active.embedding_provider == "ollama":
        return active.ollama_embedding_model
    return active.llamacpp_embedding_model


def get_vector_db_location(settings: UserSettings | None = None) -> str:
    """Return a provider/model-specific Chroma directory."""
    active = _settings_or_default(settings)
    if (
        active.embedding_provider == "ollama"
        and active.ollama_embedding_model == EMBEDDING_MODEL
    ):
        return DB_LOCATION

    key = _safe_key(f"{active.embedding_provider}_{embedding_model_name(active)}")
    return str(USER_VECTOR_DIR / key)


def _stamp_embedding_metadata(
    metadata: dict, settings: UserSettings | None = None
) -> dict:
    active = _settings_or_default(settings)
    metadata["embedding_provider"] = active.embedding_provider
    metadata["embedding_model"] = embedding_model_name(active)
    return metadata


def discover_subject_pdfs() -> list[tuple[str, Path]]:
    """Discover subject PDFs from the assets directory using filename patterns."""
    discovered: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()

    if not ASSETS_DIR.exists():
        return discovered

    for subject, patterns in SUBJECT_PDF_PATTERNS.items():
        for pattern in patterns:
            for file_path in sorted(ASSETS_DIR.glob(pattern)):
                if not file_path.is_file():
                    continue
                resolved = str(file_path.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                discovered.append((subject, file_path))

    if not discovered and LEGACY_PDF_PATH.exists():
        discovered.append(("english", LEGACY_PDF_PATH))

    return discovered


def infer_subject_from_filename(file_path: Path) -> str:
    """Infer a subject from a filename when no explicit pattern matched."""
    filename = file_path.name.lower()
    for subject, patterns in SUBJECT_PDF_PATTERNS.items():
        for pattern in patterns:
            token = pattern.replace("*", "").replace(".pdf", "")
            token = token.strip().lower()
            if token and token in filename:
                return subject
    return "english"


def discover_all_pdfs_with_subjects() -> list[tuple[str, Path]]:
    """Discover all PDFs in assets, assigning subjects via patterns then fallback."""
    discovered = discover_subject_pdfs()
    seen_paths = {str(path.resolve()) for _, path in discovered}

    if not ASSETS_DIR.exists():
        return discovered

    for file_path in sorted(ASSETS_DIR.glob("*.pdf")):
        if not file_path.is_file():
            continue
        resolved = str(file_path.resolve())
        if resolved in seen_paths:
            continue

        subject = infer_subject_from_filename(file_path)
        discovered.append((subject, file_path))
        seen_paths.add(resolved)

    return discovered


def get_ocr_text_path(pdf_path: Path) -> Path:
    """Resolve the expected OCR sidecar text path for a PDF."""
    return pdf_path.with_name(f"{pdf_path.stem}{OCR_TEXT_SUFFIX}")


def build_ingest_key(
    subject: str,
    pdf_path: Path,
    ocr_text_path: Path | None = None,
    *,
    course: str = "",
    source_role: str = "",
) -> str:
    """Build a stable ingestion key so each file version is indexed once."""
    stats = pdf_path.stat()
    key = f"{subject}:{pdf_path.name}:{stats.st_size}:{stats.st_mtime_ns}"
    if ocr_text_path and ocr_text_path.exists():
        ocr_stats = ocr_text_path.stat()
        key = f"{key}:ocr:{ocr_stats.st_size}:{ocr_stats.st_mtime_ns}"
    return key


def ingest_key_exists(vector_store: Chroma, ingest_key: str) -> bool:
    """Check whether this file version has already been indexed."""
    try:
        rows = vector_store.get(where={"ingest_key": ingest_key}, limit=1, include=[])
        return bool(rows.get("ids"))
    except Exception:
        return False


def load_and_split_pdf(
    pdf_path: str,
    *,
    subject: str,
    source_file: str,
    ingest_key: str,
    course: str = "",
    source_role: str = DEFAULT_SOURCE_ROLE,
    title: str = "",
) -> list[Document]:
    """Load a PDF and split it into chunks with subject metadata."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split the PDF in to chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=EMBED_CHUNK_SIZE,
        chunk_overlap=EMBED_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " "],
    )

    # Preserve page numbers
    final_chunks = []
    for page_idx, page in enumerate(pages):
        chunks = text_splitter.split_text(page.page_content)
        for chunk in chunks:
            metadata = {**page.metadata}
            metadata["subject"] = subject
            metadata["course"] = course
            metadata["course_label"] = course_label(course)
            metadata["source_role"] = source_role
            metadata["title"] = title or Path(source_file).stem
            metadata["source_file"] = source_file
            metadata["ingest_key"] = ingest_key
            # Always use page_idx (0-based) directly for proper filtering
            metadata["page"] = page_idx
            # Add human-readable page label (1-based)
            metadata["page_label"] = str(page_idx + 1)

            final_chunks.append(Document(page_content=chunk, metadata=metadata))

    return final_chunks


def load_and_split_ocr_text(
    text_path: Path,
    *,
    subject: str,
    source_file: str,
    ingest_key: str,
    course: str = "",
    source_role: str = DEFAULT_SOURCE_ROLE,
    title: str = "",
) -> list[Document]:
    """Load OCR text sidecar output and split it into vector chunks."""
    raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
    if not raw_text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=EMBED_CHUNK_SIZE,
        chunk_overlap=EMBED_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " "],
    )

    page_pattern = re.compile(r"=== PAGE (\d+) ===")
    segments = page_pattern.split(raw_text)
    final_chunks: list[Document] = []

    if len(segments) <= 1:
        chunks = text_splitter.split_text(raw_text)
        for chunk in chunks:
            final_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "subject": subject,
                        "course": course,
                        "course_label": course_label(course),
                        "source_role": source_role,
                        "title": title or Path(source_file).stem,
                        "source_file": source_file,
                        "ingest_key": ingest_key,
                        "ocr_text_file": text_path.name,
                    },
                )
            )
        return final_chunks

    # segments format after split: [prefix, page_num, content, page_num, content, ...]
    for index in range(1, len(segments), 2):
        page_number = segments[index]
        page_content = segments[index + 1] if index + 1 < len(segments) else ""
        if not page_content.strip():
            continue

        chunks = text_splitter.split_text(page_content)
        for chunk in chunks:
            final_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "subject": subject,
                        "course": course,
                        "course_label": course_label(course),
                        "source_role": source_role,
                        "title": title or Path(source_file).stem,
                        "source_file": source_file,
                        "ingest_key": ingest_key,
                        "ocr_text_file": text_path.name,
                        "page_label": page_number,
                        "page": int(page_number) - 1,
                    },
                )
            )

    return final_chunks


def build_page_filter(start_page: int | None = None, end_page: int | None = None):
    """Build an optional page filter for vector similarity search.

    Page numbers are 1-based for user input, but stored as 0-based in metadata.
    """
    if start_page is None and end_page is None:
        return None

    if start_page is None:
        start_page = end_page
    if end_page is None:
        end_page = start_page
    if start_page is None or end_page is None:
        return None
    if start_page < 1 or end_page < 1:
        raise ValueError("Page numbers must start at 1.")
    if start_page > end_page:
        raise ValueError("start_page must be less than or equal to end_page.")

    # Convert 1-based user page numbers to 0-based storage format
    start_page_zero_based = start_page - 1
    end_page_zero_based = end_page - 1

    return {
        "$and": [
            {"page": {"$gte": start_page_zero_based}},
            {"page": {"$lte": end_page_zero_based}},
        ]
    }


def build_subject_filter(subject: str | None = None):
    """Build an optional subject filter for vector similarity search."""
    if subject is None:
        return None

    cleaned = subject.strip().lower()
    if not cleaned:
        return None

    return {"subject": {"$eq": cleaned}}


def _as_and_clauses(filter_spec) -> list[dict]:
    if not filter_spec:
        return []
    if isinstance(filter_spec, dict) and set(filter_spec.keys()) == {"$and"}:
        return list(filter_spec["$and"])
    return [filter_spec]


def _or_filters(filters: list[dict]):
    cleaned = [item for item in filters if item]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return {"$or": cleaned}


def combine_filters(page_filter, subject_filter, study_filter=None):
    """Combine page, subject, and study-set filters into one expression."""
    clauses = []
    clauses.extend(_as_and_clauses(page_filter))
    clauses.extend(_as_and_clauses(subject_filter))
    clauses.extend(_as_and_clauses(study_filter))
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def build_study_filter(
    *,
    subject: str | None = None,
    course: str | None = None,
    source_mode: str | None = None,
):
    """Build optional course/source-role filters for focused study sets."""
    cleaned_subject = str(subject or "").strip().lower()
    cleaned_course = clean_course(course, subject=cleaned_subject)
    cleaned_mode = clean_source_mode(source_mode)
    role = cleaned_mode if cleaned_mode in {"textbook", "workbook"} else ""

    clauses = []
    aliases = list(COURSE_SOURCE_FILE_ALIASES.get((cleaned_subject, cleaned_course), ()))

    if cleaned_course:
        course_alias_filters = [
            {"source_file": {"$eq": item["source_file"]}}
            for item in aliases
        ]
        clauses.append(
            _or_filters(
                [{"course": {"$eq": cleaned_course}}, *course_alias_filters]
            )
        )

    if role:
        role_alias_filters = [
            {"source_file": {"$eq": item["source_file"]}}
            for item in aliases
            if item.get("source_role") == role
        ]
        clauses.append(
            _or_filters(
                [{"source_role": {"$eq": role}}, *role_alias_filters]
            )
        )

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def ingest_subject_documents(
    vector_store: Chroma, settings: UserSettings | None = None
):
    """Ingest all discovered subject PDFs that are not yet indexed."""
    subject_pdfs = discover_all_pdfs_with_subjects()
    if not subject_pdfs:
        print(
            f"Warning: no PDF files were found in '{ASSETS_DIR}'. "
            "Add subject PDFs to enable retrieval."
        )
        return

    for subject, pdf_path in subject_pdfs:
        ocr_text_path = get_ocr_text_path(pdf_path)
        course = infer_course_from_filename(pdf_path, subject)
        source_role = infer_source_role_from_filename(pdf_path)
        ingest_key = build_ingest_key(
            subject,
            pdf_path,
            ocr_text_path if ocr_text_path.exists() else None,
            course=course,
            source_role=source_role,
        )
        if ingest_key_exists(vector_store, ingest_key):
            continue

        chunks = load_and_split_pdf(
            str(pdf_path),
            subject=subject,
            source_file=pdf_path.name,
            ingest_key=ingest_key,
            course=course,
            source_role=source_role,
            title=pdf_path.stem,
        )
        if not chunks:
            if ocr_text_path.exists():
                chunks = load_and_split_ocr_text(
                    ocr_text_path,
                    subject=subject,
                    source_file=pdf_path.name,
                    ingest_key=ingest_key,
                    course=course,
                    source_role=source_role,
                    title=pdf_path.stem,
                )
                if chunks:
                    print(
                        f"[Vector] Using OCR text from {ocr_text_path.name} "
                        f"for {pdf_path.name} ({subject})."
                    )

            if not chunks:
                print(
                    f"[Vector] Skipped {pdf_path.name} ({subject}): no extractable text found."
                )
                print(
                    "[Vector] Run OCR first: "
                    f'python src/util/pdf_ocr.py --input "{pdf_path.as_posix()}"'
                )
                continue

        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[start : start + UPSERT_BATCH_SIZE]
            for doc in batch:
                _stamp_embedding_metadata(doc.metadata, settings)
            # Log page metadata for first chunk to verify storage
            if start == 0 and batch:
                first_chunk = batch[0]
                page_info = f"page={first_chunk.metadata.get('page')}, page_label={first_chunk.metadata.get('page_label')}"
                print(
                    f"[Vector]   Indexing {len(batch)} chunks - sample metadata: {page_info}"
                )
            vector_store.add_documents(batch)
        print(
            f"[Vector] Indexed {len(chunks)} chunks from {pdf_path.name} ({subject})."
        )


vector_store = None
_vector_stores_by_key: dict[str, Chroma] = {}


def get_vector_store(settings: UserSettings | None = None) -> Chroma:
    """Create or load vector store and ensure subject documents are indexed."""
    global vector_store
    active = _settings_or_default(settings)
    db_location = get_vector_db_location(active)
    cache_key = f"{active.embedding_provider}:{embedding_model_name(active)}:{db_location}"
    cached = _vector_stores_by_key.get(cache_key)
    if cached is not None:
        return cached

    embeddings = create_embedding_model(active)
    selected_store = Chroma(
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    # Validate and repair page metadata if needed
    _validate_and_repair_page_metadata(selected_store, db_location)

    ingest_subject_documents(selected_store, active)
    _vector_stores_by_key[cache_key] = selected_store
    vector_store = selected_store
    return selected_store


def _validate_and_repair_page_metadata(vector_store: Chroma, db_location: str):
    """Check if existing chunks have proper page metadata; repair if missing."""
    try:
        # Get a sample of existing records to check metadata
        sample = vector_store.get(limit=5, include=["metadatas"])
        if not sample.get("metadatas"):
            return  # Empty database, nothing to validate

        # Check if any records lack proper page field
        has_missing_pages = False
        for metadata in sample["metadatas"]:
            if metadata and "page" not in metadata:
                has_missing_pages = True
                break

        if has_missing_pages:
            print(
                "[Vector] Detected missing page metadata in vector database. "
                "Clearing old data to rebuild with proper page storage."
            )
            # Clear the database - it will be rebuilt with proper metadata
            try:
                db_path = Path(db_location)
                if db_path.exists():
                    shutil.rmtree(db_path)
                    print(
                        "[Vector] Old vector database cleared. Will rebuild on next ingestion."
                    )
            except Exception as e:
                print(f"[Vector] Warning: Could not clear old database: {e}")
    except Exception:
        # Silently ignore validation errors
        pass


def search_documents(
    query: str,
    *,
    subject: str | None = None,
    course: str | None = None,
    source_mode: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    k: int = 5,
    settings: UserSettings | None = None,
):
    """Search documents with combined subject and page filters.

    Args:
        query: Search query text
        subject: Optional subject filter ('history', 'chemistry', 'math', 'english')
        course: Optional course filter for subjects with multiple study sets
        source_mode: Optional source scope ('auto', 'textbook', 'workbook', 'all')
        start_page: Optional start page (1-based, inclusive)
        end_page: Optional end page (1-based, inclusive)
        k: Number of results to return

    Returns:
        List of matching documents with filters applied, using fallback strategy if needed
    """
    global vector_store
    active_store = get_vector_store(settings)

    # Build filters
    page_filter = build_page_filter(start_page, end_page)
    subject_filter = build_subject_filter(subject)
    study_filter = build_study_filter(
        subject=subject,
        course=course,
        source_mode=source_mode,
    )
    combined_filter = combine_filters(page_filter, subject_filter, study_filter)

    # Track whether page filter was requested for fallback strategy
    page_filter_requested = start_page is not None or end_page is not None

    # Step 1: Try search with all filters applied
    search_kwargs = {"k": k}
    if combined_filter is not None:
        search_kwargs["filter"] = combined_filter
        print(f"[Vector] Applying filter: {combined_filter}")
    else:
        print(f"[Vector] No filters applied")

    # Execute search with filters
    results = active_store.similarity_search(query, **search_kwargs)

    # Log results with metadata
    print(f"[Vector] Found {len(results)} results")
    for i, doc in enumerate(results, 1):
        page_label = doc.metadata.get("page_label", "unknown")
        page_num = doc.metadata.get("page", "unknown")
        subject_result = doc.metadata.get("subject", "unknown")
        course_result = doc.metadata.get("course") or infer_course_from_filename(
            Path(str(doc.metadata.get("source_file") or "")),
            str(subject_result),
        )
        role_result = doc.metadata.get("source_role") or infer_source_role_from_filename(
            Path(str(doc.metadata.get("source_file") or ""))
        )
        print(
            f"[Vector]   Result {i}: page_label={page_label} (page={page_num}), "
            f"subject={subject_result}, course={course_result}, role={role_result}"
        )

    # Step 2: Fallback strategy - if page filter was requested but returned no results,
    # retry with higher k and page filter to ensure we find content from those pages
    if page_filter_requested and len(results) == 0:
        print(
            f"[Vector] No results with combined filters. "
            f"Retrying with page filter and higher k={k * 3}..."
        )
        retry_kwargs = {"k": k * 3}
        retry_filter = combine_filters(page_filter, subject_filter, study_filter)
        if retry_filter is not None:
            retry_kwargs["filter"] = retry_filter
        results = active_store.similarity_search(query, **retry_kwargs)

        print(f"[Vector] Retry found {len(results)} results with page filter")
        for i, doc in enumerate(results, 1):
            page_label = doc.metadata.get("page_label", "unknown")
            page_num = doc.metadata.get("page", "unknown")
            subject_result = doc.metadata.get("subject", "unknown")
            print(
                f"[Vector]   Retry Result {i}: page_label={page_label} (page={page_num}), subject={subject_result}"
            )

        # Step 3: If still no results with page filter, return top k results from all pages
        # to avoid "pages don't exist" when they do
        if len(results) == 0:
            print(
                f"[Vector] Still no results with page filter. "
                f"Expanding search to all pages to find relevant content..."
            )
            expand_kwargs = {"k": k}
            expand_filter = combine_filters(None, subject_filter, study_filter)
            if expand_filter is not None:
                expand_kwargs["filter"] = expand_filter
            results = active_store.similarity_search(query, **expand_kwargs)

            print(
                f"[Vector] Expanded search found {len(results)} results "
                f"(note: these may be outside requested page range)"
            )
            for i, doc in enumerate(results, 1):
                page_label = doc.metadata.get("page_label", "unknown")
                page_num = doc.metadata.get("page", "unknown")
                subject_result = doc.metadata.get("subject", "unknown")
                print(
                    f"[Vector]   Expanded Result {i}: page_label={page_label} (page={page_num}), subject={subject_result}"
                )

    # Validate that filters were actually applied to results
    if combined_filter is not None:
        _validate_filter_applied(
            results, combined_filter, subject, start_page, end_page
        )

    return results


def _validate_filter_applied(results, filter_spec, subject, start_page, end_page):
    """Verify that filters were actually applied to search results."""
    if not results:
        return  # No results, nothing to validate

    # Check subject filter application
    if subject:
        for doc in results:
            doc_subject = doc.metadata.get("subject", "").lower()
            if doc_subject and doc_subject != subject.lower():
                print(
                    f"[Vector] WARNING: Subject filter not fully applied. "
                    f"Expected '{subject}' but got '{doc_subject}' in result."
                )
                break

    # Check page filter application
    if start_page is not None or end_page is not None:
        for doc in results:
            doc_page = doc.metadata.get("page")
            if doc_page is not None:
                page_num_1based = doc_page + 1

                if start_page is not None and page_num_1based < start_page:
                    print(
                        f"[Vector] WARNING: Page filter not fully applied. "
                        f"Result on page {page_num_1based} but start_page is {start_page}."
                    )
                    break

                if end_page is not None and page_num_1based > end_page:
                    print(
                        f"[Vector] WARNING: Page filter not fully applied. "
                        f"Result on page {page_num_1based} but end_page is {end_page}."
                    )
                    break


def get_retriever(
    k: int = 5,
    subject: str | None = None,
    course: str | None = None,
    source_mode: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    settings: UserSettings | None = None,
):
    """Create a retriever with optional subject and page range filters.

    Args:
        k: Number of documents to retrieve
        subject: Subject to filter by (e.g., 'history', 'chemistry', 'math', 'english')
        start_page: First page number (1-based, inclusive)
        end_page: Last page number (1-based, inclusive)

    Returns:
        A LangChain retriever with appropriate filters applied
    """
    global vector_store
    active_store = get_vector_store(settings)

    # Build filters
    page_filter = build_page_filter(start_page, end_page)
    subject_filter = build_subject_filter(subject)
    study_filter = build_study_filter(
        subject=subject,
        course=course,
        source_mode=source_mode,
    )
    combined_filter = combine_filters(page_filter, subject_filter, study_filter)

    search_kwargs = {"k": k}
    if combined_filter is not None:
        search_kwargs["filter"] = combined_filter

    return active_store.as_retriever(search_kwargs=search_kwargs)


def _get_collection_rows_fast(
    *, settings: UserSettings | None = None, db_location: str | None = None
) -> dict:
    """Read metadata directly from local Chroma DB without embedding calls."""
    db_path = Path(db_location or get_vector_db_location(settings))
    if not db_path.exists():
        return {"ids": [], "metadatas": []}

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(COLLECTION_NAME)
        return collection.get(include=["metadatas"])
    except Exception:
        return {"ids": [], "metadatas": []}


def get_ingestion_summary(
    *, ensure_indexed: bool = False, settings: UserSettings | None = None
) -> dict:
    """Return a summary of indexed chunks grouped by subject and source file.

    Set ensure_indexed=True to run embedding-backed ingestion before summarizing.
    """
    rows = None
    active = _settings_or_default(settings)
    if ensure_indexed:
        global vector_store
        active_store = get_vector_store(active)
        try:
            rows = active_store.get(include=["metadatas"])
        except Exception as error:
            return {
                "total_chunks": 0,
                "subjects": {},
                "discovered_files": [],
                "error": str(error),
            }

    discovered_files = [
        {
            "subject": subject,
            "course": infer_course_from_filename(path, subject),
            "course_label": course_label(infer_course_from_filename(path, subject)),
            "source_role": infer_source_role_from_filename(path),
            "source_file": path.name,
            "source_path": str(path),
            "title": path.stem,
            "file_type": path.suffix.removeprefix(".").lower() or "pdf",
            "has_ocr_text": get_ocr_text_path(path).exists(),
        }
        for subject, path in discover_all_pdfs_with_subjects()
    ]

    if rows is None:
        rows = _get_collection_rows_fast(settings=active)

    metadatas = rows.get("metadatas") or []
    ids = rows.get("ids") or []
    summary: dict[str, dict] = {}
    courses: dict[str, dict] = {}
    indexed_source_files: set[str] = set()

    for metadata in metadatas:
        record = metadata or {}
        subject = str(record.get("subject") or "unknown")
        source_file = str(record.get("source_file") or "unknown")
        course = str(record.get("course") or "").strip()
        if not course:
            course = infer_course_from_filename(Path(source_file), subject)
        source_role = str(record.get("source_role") or "").strip()
        if not source_role:
            source_role = infer_source_role_from_filename(Path(source_file))
        if source_file != "unknown":
            indexed_source_files.add(source_file)

        subject_bucket = summary.setdefault(subject, {"chunks": 0, "files": {}})
        subject_bucket["chunks"] += 1
        file_counts = subject_bucket["files"]
        file_counts[source_file] = file_counts.get(source_file, 0) + 1
        if course:
            course_key = f"{subject}:{course}"
            course_bucket = courses.setdefault(
                course_key,
                {
                    "subject": subject,
                    "course": course,
                    "course_label": course_label(course),
                    "chunks": 0,
                    "roles": {},
                },
            )
            course_bucket["chunks"] += 1
            roles = course_bucket["roles"]
            roles[source_role] = roles.get(source_role, 0) + 1

    pending_files = [
        file_info
        for file_info in discovered_files
        if file_info["source_file"] not in indexed_source_files
    ]
    builtin_sources = []
    pending_keys = {
        (item["subject"], item["source_file"]) for item in pending_files
    }
    for item in discovered_files:
        subject = item["subject"]
        source_file = item["source_file"]
        chunk_count = (
            summary.get(subject, {}).get("files", {}).get(source_file, 0)
        )
        status = (
            "pending"
            if (subject, source_file) in pending_keys or chunk_count == 0
            else "ready"
        )
        builtin_sources.append(
            {
                **item,
                "status": status,
                "chunk_count": chunk_count,
                "library_asset": False,
            }
        )

    return {
        "total_chunks": len(ids) or len(metadatas),
        "subjects": summary,
        "courses": list(courses.values()),
        "discovered_files": discovered_files,
        "pending_files": pending_files,
        "builtin_sources": builtin_sources,
        "embedding_provider": active.embedding_provider,
        "embedding_model": embedding_model_name(active),
        "db_location": get_vector_db_location(active),
    }


def index_documents(
    documents: list[Document], *, settings: UserSettings | None = None
) -> int:
    """Add already-extracted documents to the active vector store."""
    if not documents:
        return 0

    active = _settings_or_default(settings)
    active_store = get_vector_store(active)
    indexed = 0
    for start in range(0, len(documents), UPSERT_BATCH_SIZE):
        batch = documents[start : start + UPSERT_BATCH_SIZE]
        for doc in batch:
            _stamp_embedding_metadata(doc.metadata, active)
        active_store.add_documents(batch)
        indexed += len(batch)
    return indexed


def delete_documents_for_asset(
    asset_id: str, *, settings: UserSettings | None = None
) -> None:
    """Remove indexed chunks for one user-library asset when supported by Chroma."""
    if not asset_id:
        return
    active_store = get_vector_store(settings)
    try:
        active_store.delete(where={"asset_id": asset_id})
    except Exception as error:
        print(f"[Vector] Failed to delete chunks for asset {asset_id}: {error}")


def print_ingestion_summary():
    """Print a human-readable summary of indexed ingestion status."""
    summary = get_ingestion_summary()

    print("\n=== Vector Ingestion Summary ===")
    print(f"Total indexed chunks: {summary.get('total_chunks', 0)}")

    error = summary.get("error")
    if error:
        print(f"Error reading vector store: {error}")
        return

    discovered = summary.get("discovered_files", [])
    if discovered:
        print("\nDiscovered PDFs:")
        for item in discovered:
            print(f"- {item['subject']}: {item['source_file']}")
    else:
        print("\nDiscovered PDFs: none")

    subjects = summary.get("subjects", {})
    if not subjects:
        print("\nIndexed subjects: none")
        return

    print("\nIndexed subjects:")
    for subject in sorted(subjects):
        bucket = subjects[subject]
        print(f"- {subject}: {bucket['chunks']} chunks")
        files = bucket.get("files", {})
        for source_file in sorted(files):
            print(f"  * {source_file}: {files[source_file]} chunks")

    if "unknown" in subjects:
        print(
            "\nNote: 'unknown' chunks are legacy records created before subject metadata"
            " was added."
        )

    pending_files = summary.get("pending_files", [])
    if pending_files:
        print("\nDiscovered but not indexed yet:")
        for item in pending_files:
            print(f"- {item['subject']}: {item['source_file']}")
        print(
            "  These files may be image-only PDFs or still pending first successful ingestion."
        )


def diagnose_filters():
    """Test that subject and page filters are working correctly.

    This is a diagnostic function to verify filters are properly applied
    during vector database retrieval.
    """
    global vector_store

    print("\n=== Vector Filter Diagnostic ===\n")

    if vector_store is None:
        vector_store = get_vector_store()

    # Get summary of database
    summary = get_ingestion_summary()
    total_chunks = summary.get("total_chunks", 0)

    if total_chunks == 0:
        print("No chunks indexed. Cannot run filter diagnostics.")
        return

    print(f"Total chunks in database: {total_chunks}\n")

    # Test 1: Subject filter
    print("TEST 1: Subject filtering")
    for subject in ["history", "chemistry", "math", "english"]:
        results = search_documents("sample query", subject=subject, k=5)
        if results:
            # Verify all results match the subject filter
            all_match = all(
                doc.metadata.get("subject", "").lower() == subject.lower()
                for doc in results
            )
            status = "✓ PASS" if all_match else "✗ FAIL"
            print(
                f"  {status}: {subject.upper()} - found {len(results)} results, all match={all_match}"
            )
        else:
            print(f"  - {subject.upper()} - no results")

    # Test 2: Page filtering
    print("\nTEST 2: Page range filtering")

    # Get sample pages from database
    try:
        all_metadata = vector_store.get(include=["metadatas"], limit=100)
        pages_in_db = set()
        for metadata in all_metadata.get("metadatas", []):
            if metadata and "page" in metadata:
                pages_in_db.add(int(metadata["page"]) + 1)  # Convert to 1-based

        if pages_in_db:
            min_page = min(pages_in_db)
            max_page = max(pages_in_db)
            print(f"  Pages in database: {min_page} to {max_page}")

            # Test a specific page range
            test_start = min_page
            test_end = min(min_page + 2, max_page)

            results = search_documents(
                "sample query", start_page=test_start, end_page=test_end, k=10
            )

            if results:
                all_in_range = all(
                    test_start <= (doc.metadata.get("page", 0) + 1) <= test_end
                    for doc in results
                )
                status = "✓ PASS" if all_in_range else "✗ FAIL"
                print(
                    f"  {status}: Pages {test_start}-{test_end} - found {len(results)} results, all in range={all_in_range}"
                )
            else:
                print(f"  - Pages {test_start}-{test_end} - no results")
        else:
            print("  No page metadata found in database")
    except Exception as e:
        print(f"  Error testing page filtering: {e}")

    # Test 3: Combined filters
    print("\nTEST 3: Combined subject + page filtering")

    try:
        # Get a subject with pages
        summary = get_ingestion_summary()
        subjects = summary.get("subjects", {})

        if subjects:
            test_subject = list(subjects.keys())[0]
            results = search_documents(
                "sample query", subject=test_subject, start_page=1, end_page=5, k=10
            )

            if results:
                subject_match = all(
                    doc.metadata.get("subject", "").lower() == test_subject.lower()
                    for doc in results
                )
                page_match = all(
                    1 <= (doc.metadata.get("page", 0) + 1) <= 5
                    for doc in results
                    if "page" in doc.metadata
                )
                status = "✓ PASS" if (subject_match and page_match) else "✗ FAIL"
                print(
                    f"  {status}: {test_subject.upper()} pages 1-5 - {len(results)} results, subject ok={subject_match}, pages ok={page_match}"
                )
            else:
                print(f"  - No results for {test_subject.upper()} pages 1-5")
    except Exception as e:
        print(f"  Error testing combined filters: {e}")

    print("\n=== End of Diagnostic ===\n")


def parse_args():
    """Parse CLI args for status and ingestion actions."""
    parser = argparse.ArgumentParser(description="Vector ingestion utilities")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run embedding-backed ingestion before printing summary.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run filter diagnostics to verify subject and page filtering works.",
    )
    return parser.parse_args()


def main():
    """CLI entrypoint for vector indexing and status checks."""
    args = parse_args()
    if args.ingest:
        print("Running ingestion (embeddings enabled)...")
        get_vector_store()
    if args.diagnose:
        diagnose_filters()
    print_ingestion_summary()


if __name__ == "__main__":
    main()
