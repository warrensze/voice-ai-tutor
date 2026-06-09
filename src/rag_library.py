"""User-managed study asset library for local RAG ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings_store import DATA_DIR, SUPPORTED_SUBJECTS, UserSettings, load_user_settings
from vector import (
    EMBED_CHUNK_OVERLAP,
    EMBED_CHUNK_SIZE,
    delete_documents_for_asset,
    index_documents,
)

LIBRARY_DIR = DATA_DIR / "library"
LIBRARY_INDEX_PATH = LIBRARY_DIR / "library_index.json"
EXTRACTED_DIR = LIBRARY_DIR / "extracted"
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(name).name)
    return cleaned.strip("._") or "asset"


def _clean_subject(subject: str | None) -> str:
    value = str(subject or "").strip().lower()
    return value if value in SUPPORTED_SUBJECTS else "english"


def _infer_subject(filename: str) -> str:
    lower = filename.lower()
    subject_tokens = {
        "history": ("history", "world", "apwh"),
        "chemistry": ("chem", "chemistry"),
        "math": ("math", "algebra", "geometry", "calculus"),
        "english": ("english", "literature", "grammar", "novel", "writing"),
    }
    for subject, tokens in subject_tokens.items():
        if any(token in lower for token in tokens):
            return subject
    return "english"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LibraryManager:
    """Manage user-added local RAG assets and ingestion status."""

    def __init__(
        self,
        *,
        library_dir: str | Path | None = None,
        index_path: str | Path | None = None,
    ):
        self.library_dir = Path(library_dir) if library_dir else LIBRARY_DIR
        self.index_path = Path(index_path) if index_path else LIBRARY_INDEX_PATH
        self.extracted_dir = self.library_dir / "extracted"
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_index()

    def list_assets(self) -> list[dict[str, Any]]:
        payload = self._read_index()
        return list(payload.get("assets") or [])

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        for asset in self.list_assets():
            if asset.get("id") == asset_id:
                return asset
        return None

    def add_asset(
        self,
        source_path: str | Path,
        *,
        original_filename: str | None = None,
        subject: str | None = None,
        title: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        source = Path(source_path)
        filename = original_filename or source.name
        extension = source.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError("Supported study assets are PDF, EPUB, and TXT files.")

        file_hash = _file_sha256(source)
        duplicate = self._find_by_hash(file_hash)
        if duplicate is not None:
            duplicate["duplicate"] = True
            return duplicate

        asset_id = uuid.uuid4().hex
        safe_name = _safe_filename(filename)
        destination = self.library_dir / f"{asset_id}_{safe_name}"
        shutil.copy2(source, destination)

        asset = {
            "id": asset_id,
            "title": title.strip() if title else Path(filename).stem,
            "subject": _clean_subject(subject) if subject else _infer_subject(filename),
            "notes": notes.strip(),
            "source_path": str(destination),
            "source_file": filename,
            "file_type": extension.removeprefix("."),
            "file_hash": file_hash,
            "status": "queued",
            "chunk_count": 0,
            "error": "",
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "indexed_at": "",
            "duplicate": False,
        }
        self._upsert_asset(asset)
        return asset

    def update_asset(self, asset_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        for key in ("title", "notes"):
            if key in updates:
                asset[key] = str(updates.get(key) or "").strip()
        if "subject" in updates:
            asset["subject"] = _clean_subject(str(updates.get("subject") or ""))
            asset["status"] = "queued"
        asset["updated_at"] = _utc_now_iso()
        self._upsert_asset(asset)
        return asset

    def remove_asset(
        self, asset_id: str, *, settings: UserSettings | None = None
    ) -> dict[str, Any] | None:
        asset = self.get_asset(asset_id)
        if asset is None:
            return None
        delete_documents_for_asset(asset_id, settings=settings or load_user_settings())
        source_path = Path(str(asset.get("source_path") or ""))
        if source_path.exists() and source_path.is_file():
            source_path.unlink()
        extracted_path = self._extracted_text_path(asset_id)
        if extracted_path.exists():
            extracted_path.unlink()
        self._delete_asset(asset_id)
        return asset

    def reindex_asset(
        self, asset_id: str, *, settings: UserSettings | None = None
    ) -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        asset["status"] = "queued"
        asset["error"] = ""
        asset["updated_at"] = _utc_now_iso()
        self._upsert_asset(asset)
        return self.index_asset(asset_id, settings=settings)

    def index_asset(
        self, asset_id: str, *, settings: UserSettings | None = None
    ) -> dict[str, Any]:
        active_settings = settings or load_user_settings()
        asset = self.get_asset(asset_id)
        if asset is None:
            raise KeyError(asset_id)

        try:
            self._set_status(asset_id, "extracting")
            documents, preview = self._extract_documents(asset)
            self._write_preview(asset_id, preview)
            if not documents:
                status = "needs_ocr" if asset.get("file_type") == "pdf" else "failed"
                return self._set_status(
                    asset_id,
                    status,
                    error="No extractable text was found.",
                    chunk_count=0,
                )

            self._set_status(asset_id, "embedding")
            delete_documents_for_asset(asset_id, settings=active_settings)
            chunk_count = index_documents(documents, settings=active_settings)
            return self._set_status(
                asset_id,
                "ready",
                chunk_count=chunk_count,
                indexed_at=_utc_now_iso(),
            )
        except Exception as error:
            return self._set_status(asset_id, "failed", error=str(error))

    def preview_asset(self, asset_id: str, *, max_chars: int = 12000) -> str:
        path = self._extracted_text_path(asset_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]

    def _extract_documents(self, asset: dict[str, Any]) -> tuple[list[Document], str]:
        source_path = Path(str(asset.get("source_path") or ""))
        file_type = str(asset.get("file_type") or "").lower()
        if file_type == "pdf":
            return self._extract_pdf(asset, source_path)
        if file_type == "epub":
            return self._extract_epub(asset, source_path)
        if file_type == "txt":
            return self._extract_text(asset, source_path)
        return [], ""

    def _extract_pdf(
        self, asset: dict[str, Any], source_path: Path
    ) -> tuple[list[Document], str]:
        loader = PyPDFLoader(str(source_path))
        pages = loader.load()
        splitter = self._splitter()
        documents: list[Document] = []
        preview_parts = []
        for page_idx, page in enumerate(pages):
            page_text = page.page_content or ""
            if page_text.strip():
                preview_parts.append(page_text)
            for chunk in splitter.split_text(page_text):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=self._metadata(
                            asset,
                            page=page_idx,
                            page_label=str(page_idx + 1),
                        ),
                    )
                )

        if documents:
            return documents, "\n\n".join(preview_parts)

        sidecar = source_path.with_name(f"{source_path.stem}.ocr.txt")
        if sidecar.exists():
            return self._extract_text(asset, sidecar)
        return [], ""

    def _extract_epub(
        self, asset: dict[str, Any], source_path: Path
    ) -> tuple[list[Document], str]:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except Exception as error:
            raise RuntimeError(
                "EPUB support requires EbookLib and beautifulsoup4."
            ) from error

        book = epub.read_epub(str(source_path))
        splitter = self._splitter()
        documents: list[Document] = []
        preview_parts = []
        chapter_index = 0
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            chapter_index += 1
            html = item.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text("\n", strip=True)
            if not text:
                continue
            preview_parts.append(text)
            for chunk in splitter.split_text(text):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=self._metadata(
                            asset,
                            page=chapter_index - 1,
                            page_label=f"chapter {chapter_index}",
                        ),
                    )
                )
        return documents, "\n\n".join(preview_parts)

    def _extract_text(
        self, asset: dict[str, Any], source_path: Path
    ) -> tuple[list[Document], str]:
        raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
        if not raw_text.strip():
            return [], ""

        splitter = self._splitter()
        page_pattern = re.compile(r"=== PAGE (\d+) ===")
        segments = page_pattern.split(raw_text)
        documents: list[Document] = []

        if len(segments) <= 1:
            for chunk in splitter.split_text(raw_text):
                documents.append(
                    Document(page_content=chunk, metadata=self._metadata(asset))
                )
            return documents, raw_text

        for index in range(1, len(segments), 2):
            page_number = segments[index]
            page_content = segments[index + 1] if index + 1 < len(segments) else ""
            for chunk in splitter.split_text(page_content):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata=self._metadata(
                            asset,
                            page=int(page_number) - 1,
                            page_label=page_number,
                        ),
                    )
                )
        return documents, raw_text

    def _metadata(
        self,
        asset: dict[str, Any],
        *,
        page: int | None = None,
        page_label: str | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "asset_id": asset["id"],
            "library_asset": True,
            "subject": asset["subject"],
            "title": asset["title"],
            "source_file": asset["source_file"],
            "source_path": asset["source_path"],
            "file_type": asset["file_type"],
            "file_hash": asset["file_hash"],
            "ingest_key": f"library:{asset['id']}:{asset['file_hash']}",
        }
        if page is not None:
            metadata["page"] = page
        if page_label is not None:
            metadata["page_label"] = page_label
        return metadata

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=EMBED_CHUNK_SIZE,
            chunk_overlap=EMBED_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " "],
        )

    def _ensure_index(self) -> None:
        if self.index_path.exists():
            return
        self._write_index({"version": 1, "updated_at": _utc_now_iso(), "assets": []})

    def _read_index(self) -> dict[str, Any]:
        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = {"version": 1, "updated_at": _utc_now_iso(), "assets": []}
        if not isinstance(payload.get("assets"), list):
            payload["assets"] = []
        return payload

    def _write_index(self, payload: dict[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = _utc_now_iso()
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        tmp_path.replace(self.index_path)

    def _upsert_asset(self, asset: dict[str, Any]) -> None:
        payload = self._read_index()
        assets = [item for item in payload["assets"] if item.get("id") != asset["id"]]
        assets.append(asset)
        payload["assets"] = sorted(assets, key=lambda item: item.get("created_at", ""))
        self._write_index(payload)

    def _delete_asset(self, asset_id: str) -> None:
        payload = self._read_index()
        payload["assets"] = [
            item for item in payload["assets"] if item.get("id") != asset_id
        ]
        self._write_index(payload)

    def _find_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        for asset in self.list_assets():
            if asset.get("file_hash") == file_hash:
                return dict(asset)
        return None

    def _set_status(
        self,
        asset_id: str,
        status: str,
        *,
        error: str = "",
        chunk_count: int | None = None,
        indexed_at: str | None = None,
    ) -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        asset["status"] = status
        asset["error"] = error
        asset["updated_at"] = _utc_now_iso()
        asset["duplicate"] = False
        if chunk_count is not None:
            asset["chunk_count"] = chunk_count
        if indexed_at is not None:
            asset["indexed_at"] = indexed_at
        self._upsert_asset(asset)
        return asset

    def _extracted_text_path(self, asset_id: str) -> Path:
        return self.extracted_dir / f"{asset_id}.txt"

    def _write_preview(self, asset_id: str, text: str) -> None:
        path = self._extracted_text_path(asset_id)
        path.write_text(text[:200000], encoding="utf-8")
