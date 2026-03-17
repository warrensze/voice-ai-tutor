from pathlib import Path
import argparse
import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

# Data locations
ASSETS_DIR = Path("./assets")
DB_LOCATION = "./chrome_langchain_db"
EMBEDDING_MODEL = "mxbai-embed-large"
COLLECTION_NAME = "langchain"
LEGACY_PDF_PATH = ASSETS_DIR / "Grade9GTjoyluckclub.pdf"
EMBED_CHUNK_SIZE = 500
EMBED_CHUNK_OVERLAP = 100
UPSERT_BATCH_SIZE = 1000
OCR_TEXT_SUFFIX = ".ocr.txt"

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
    subject: str, pdf_path: Path, ocr_text_path: Path | None = None
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
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        for chunk in chunks:
            metadata = {**page.metadata}
            metadata["subject"] = subject
            metadata["source_file"] = source_file
            metadata["ingest_key"] = ingest_key
            if "page" in metadata and "page_label" not in metadata:
                metadata["page_label"] = str(int(metadata["page"]) + 1)
            final_chunks.append(Document(page_content=chunk, metadata=metadata))

    # chunks = text_splitter.split_documents(documents)
    return final_chunks


def load_and_split_ocr_text(
    text_path: Path,
    *,
    subject: str,
    source_file: str,
    ingest_key: str,
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
    """Build an optional page filter for vector similarity search."""
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

    return {
        "$and": [
            {"page": {"$gte": start_page - 1}},
            {"page": {"$lte": end_page - 1}},
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


def combine_filters(page_filter, subject_filter):
    """Combine page and subject filters into one filter expression."""
    if page_filter and subject_filter:
        clauses = []
        if isinstance(page_filter, dict) and set(page_filter.keys()) == {"$and"}:
            clauses.extend(page_filter["$and"])
        else:
            clauses.append(page_filter)
        clauses.append(subject_filter)
        return {"$and": clauses}
    return page_filter or subject_filter


def ingest_subject_documents(vector_store: Chroma):
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
        ingest_key = build_ingest_key(
            subject,
            pdf_path,
            ocr_text_path if ocr_text_path.exists() else None,
        )
        if ingest_key_exists(vector_store, ingest_key):
            continue

        chunks = load_and_split_pdf(
            str(pdf_path),
            subject=subject,
            source_file=pdf_path.name,
            ingest_key=ingest_key,
        )
        if not chunks:
            if ocr_text_path.exists():
                chunks = load_and_split_ocr_text(
                    ocr_text_path,
                    subject=subject,
                    source_file=pdf_path.name,
                    ingest_key=ingest_key,
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
            vector_store.add_documents(chunks[start : start + UPSERT_BATCH_SIZE])
        print(
            f"[Vector] Indexed {len(chunks)} chunks from {pdf_path.name} ({subject})."
        )


def get_vector_store() -> Chroma:
    """Create or load vector store and ensure subject documents are indexed."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )
    ingest_subject_documents(vector_store)
    return vector_store


vector_store = None


def search_documents(
    query: str,
    *,
    subject: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    k: int = 5,
):
    global vector_store
    if vector_store is None:
        vector_store = get_vector_store()

    page_filter = build_page_filter(start_page, end_page)
    subject_filter = build_subject_filter(subject)
    combined_filter = combine_filters(page_filter, subject_filter)
    search_kwargs = {"k": k}
    if combined_filter is not None:
        search_kwargs["filter"] = combined_filter

    return vector_store.similarity_search(query, **search_kwargs)


def get_retriever(k: int = 5, subject: str | None = None):
    global vector_store
    if vector_store is None:
        vector_store = get_vector_store()

    search_kwargs = {"k": k}
    subject_filter = build_subject_filter(subject)
    if subject_filter is not None:
        search_kwargs["filter"] = subject_filter

    return vector_store.as_retriever(search_kwargs=search_kwargs)


def _get_collection_rows_fast() -> dict:
    """Read metadata directly from local Chroma DB without embedding calls."""
    db_path = Path(DB_LOCATION)
    if not db_path.exists():
        return {"ids": [], "metadatas": []}

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(COLLECTION_NAME)
        return collection.get(include=["metadatas"])
    except Exception:
        return {"ids": [], "metadatas": []}


def get_ingestion_summary(*, ensure_indexed: bool = False) -> dict:
    """Return a summary of indexed chunks grouped by subject and source file.

    Set ensure_indexed=True to run embedding-backed ingestion before summarizing.
    """
    rows = None
    if ensure_indexed:
        global vector_store
        if vector_store is None:
            vector_store = get_vector_store()
        try:
            rows = vector_store.get(include=["metadatas"])
        except Exception as error:
            return {
                "total_chunks": 0,
                "subjects": {},
                "discovered_files": [],
                "error": str(error),
            }

    discovered_files = [
        {"subject": subject, "source_file": path.name}
        for subject, path in discover_all_pdfs_with_subjects()
    ]

    if rows is None:
        rows = _get_collection_rows_fast()

    metadatas = rows.get("metadatas") or []
    ids = rows.get("ids") or []
    summary: dict[str, dict] = {}
    indexed_source_files: set[str] = set()

    for metadata in metadatas:
        record = metadata or {}
        subject = str(record.get("subject") or "unknown")
        source_file = str(record.get("source_file") or "unknown")
        if source_file != "unknown":
            indexed_source_files.add(source_file)

        subject_bucket = summary.setdefault(subject, {"chunks": 0, "files": {}})
        subject_bucket["chunks"] += 1
        file_counts = subject_bucket["files"]
        file_counts[source_file] = file_counts.get(source_file, 0) + 1

    pending_files = [
        file_info
        for file_info in discovered_files
        if file_info["source_file"] not in indexed_source_files
    ]

    return {
        "total_chunks": len(ids) or len(metadatas),
        "subjects": summary,
        "discovered_files": discovered_files,
        "pending_files": pending_files,
    }


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


def parse_args():
    """Parse CLI args for status and ingestion actions."""
    parser = argparse.ArgumentParser(description="Vector ingestion utilities")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run embedding-backed ingestion before printing summary.",
    )
    return parser.parse_args()


def main():
    """CLI entrypoint for vector indexing and status checks."""
    args = parse_args()
    if args.ingest:
        print("Running ingestion (embeddings enabled)...")
        get_vector_store()
    print_ingestion_summary()


if __name__ == "__main__":
    main()
