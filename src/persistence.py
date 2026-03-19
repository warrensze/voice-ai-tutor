"""Persistence helpers for conversation continuity and question banking."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SUBJECTS = ("history", "chemistry", "math", "english")
QUESTION_HINTS = (
    "question",
    "questions",
    "quiz",
    "test",
    "exam",
    "practice",
    "multiple choice",
    "mcq",
    "true or false",
    "assessment",
    "practice test",
    "mock exam",
    "short answer",
    "long answer",
    "free response",
    "fill in the blank",
    "fill-in-the-blank",
    "essay",
)


def _utc_now_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_question(text: str) -> str:
    """Normalize question text for deduplication."""
    compact = re.sub(r"\s+", " ", text.strip().lower())
    compact = compact.rstrip("?.! ")
    return compact


def _strip_markdown_artifacts(text: str) -> str:
    """Remove lightweight markdown wrappers to improve parser tolerance."""
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"\[(.*?)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = cleaned.replace("`", "")
    return cleaned.strip()


def _clean_question_candidate(text: str) -> str:
    """Strip common list prefixes and collapse whitespace."""
    if not text:
        return ""

    cleaned = _strip_markdown_artifacts(text)
    cleaned = re.sub(
        r"^\s*(?:[-*]\s*)?(?:(?:question|q)\s*\d+[\).:-]|\d+[\).:-])\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in cleaned.splitlines()
        if line.strip()
    ]
    cleaned = "\n".join(lines).strip().rstrip(" -:")
    return cleaned


def _supports_assessment_storage(text: str, source_prompt: str = "") -> bool:
    """Return True when content clearly indicates generated assessment items."""
    combined = f"{source_prompt}\n{text}".strip()
    lower = combined.lower()
    if any(hint in lower for hint in QUESTION_HINTS):
        return True

    explicit_type_cues = (
        "multiple choice",
        "fill in the blank",
        "fill-in-the-blank",
        "essay:",
        "short answer:",
        "long answer:",
        "free response:",
    )
    if any(cue in lower for cue in explicit_type_cues):
        return True

    # Accept explicit answer-option formatting as a strong exam signal.
    if re.search(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\(?[a-h]\)?[\).:-]\s+\S+", text):
        return True

    return False


def _extract_multiple_choice_items(source: str) -> list[dict[str, str]]:
    """Extract multiple-choice items from common plain-text/markdown layouts."""
    candidates: list[dict[str, str]] = []
    current_stem = ""
    current_options: list[tuple[str, str]] = []

    question_prefix_pattern = re.compile(
        r"(?i)^(?:(?:question|q)\s*\d*|\d+)\s*[\).:-]\s*(.+)$"
    )
    option_pattern = re.compile(r"(?i)^\(?([a-h])\)?[\).:-]\s+(.+)$")

    def flush_current() -> None:
        nonlocal current_stem, current_options

        stem = _clean_question_candidate(current_stem)
        if stem and len(current_options) >= 2:
            option_lines = [
                f"{letter.upper()}) {_clean_question_candidate(option_text)}"
                for letter, option_text in current_options
            ]
            candidates.append(
                {
                    "question": f"{stem}\n" + "\n".join(option_lines),
                    "question_type": "multiple_choice",
                }
            )

        current_stem = ""
        current_options = []

    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        normalized_line = _strip_markdown_artifacts(stripped)
        normalized_line = re.sub(
            r"(?i)\s+(?=(?:question\s*\d*\s*[\).:-]))",
            "\n",
            normalized_line,
        )
        normalized_line = re.sub(
            r"(?i)(?<=[^\s\-*])\s+(?=(?:\(?[a-h]\)?[\).:-]\s+\S))",
            "\n",
            normalized_line,
        )

        for segment in normalized_line.splitlines():
            line = segment.strip()
            line = re.sub(r"^\s*[-*]\s+", "", line).strip()
            if not line:
                continue

            option_match = option_pattern.match(line)
            if option_match and current_stem:
                letter = option_match.group(1)
                option_text = option_match.group(2).strip()
                current_options.append((letter, option_text))
                continue

            question_match = question_prefix_pattern.match(line)
            if question_match:
                flush_current()
                current_stem = question_match.group(1).strip()
                continue

            # Some outputs put the question stem without numbering but still include options.
            if not current_stem and line.endswith("?"):
                current_stem = line
                continue

            # Support wrapped stem/option lines.
            if current_stem and current_options:
                letter, text = current_options[-1]
                current_options[-1] = (letter, f"{text} {line}".strip())
                continue

            if current_stem and not current_options:
                current_stem = f"{current_stem} {line}".strip()

    flush_current()
    return candidates


def _dedupe_assessment_items(
    items: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Deduplicate extracted items while preserving order."""
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in items:
        question = _clean_question_candidate(item.get("question", ""))
        question_type = str(item.get("question_type", "unknown")).strip().lower()
        if not question:
            continue

        normalized = _normalize_question(question)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        deduped.append(
            {
                "question": question,
                "question_type": question_type if question_type else "unknown",
            }
        )

    return deduped


def extract_assessment_items(
    text: str,
    *,
    source_prompt: str = "",
) -> list[dict[str, str]]:
    """Extract explicit tutor-generated assessment items with question type labels."""
    if not text or not text.strip():
        return []

    source = text.strip()
    if not _supports_assessment_storage(source, source_prompt):
        return []

    candidates: list[dict[str, str]] = []
    normalized_source = "\n".join(
        _strip_markdown_artifacts(line) for line in source.splitlines()
    )

    for candidate in _extract_multiple_choice_items(source):
        candidates.append(candidate)

    mcq_stems = {
        _normalize_question(
            _clean_question_candidate(item["question"].split("\n", 1)[0])
        )
        for item in candidates
        if item.get("question_type") == "multiple_choice"
    }

    fill_blank_pattern = re.compile(
        r"(?im)^\s*(?:question\s*\d+[:\-]\s*|\d+[\).:-]\s*)?"
        r"(?:fill\s*in\s*the\s*blank[:\-]?\s*)?"
        r"(.{5,320}(?:_{3,}|\bblank\b).*)$"
    )
    for match in fill_blank_pattern.finditer(normalized_source):
        question_text = _clean_question_candidate(match.group(1))
        if question_text:
            candidates.append(
                {
                    "question": question_text,
                    "question_type": "fill_in_the_blank",
                }
            )

    essay_label_pattern = re.compile(
        r"(?im)^\s*(?:question\s*\d+[:\-]\s*|\d+[\).:-]\s*)?"
        r"(?:essay|short answer|long answer|free response)\s*[:\-]\s*(.{8,360})$"
    )
    for match in essay_label_pattern.finditer(normalized_source):
        question_text = _clean_question_candidate(match.group(1))
        if question_text:
            candidates.append(
                {
                    "question": question_text,
                    "question_type": "essay",
                }
            )

    # Capture clearly formatted short-answer prompts when assessment context exists.
    short_answer_pattern = re.compile(
        r"(?im)^\s*(?:question\s*\d+[:\-]\s*|\d+[\).:-]\s*)(.{6,320}\?)\s*$"
    )
    for match in short_answer_pattern.finditer(normalized_source):
        question_text = _clean_question_candidate(match.group(1))
        if question_text:
            stem_key = _normalize_question(question_text)
            if stem_key in mcq_stems:
                continue
            candidates.append(
                {
                    "question": question_text,
                    "question_type": "short_answer",
                }
            )

    # Numbered exam prompts sometimes omit question marks.
    numbered_prompt_pattern = re.compile(
        r"(?im)^\s*(?:question\s*\d+[:\-]\s*|\d+[\).:-]\s*)(.{8,320})\s*$"
    )
    for match in numbered_prompt_pattern.finditer(normalized_source):
        question_text = _clean_question_candidate(match.group(1))
        if not question_text:
            continue

        # Skip answer-option lines or obvious non-question boilerplate.
        if re.match(r"(?i)^[a-h][\).:-]\s+", question_text):
            continue
        if any(
            marker in question_text.lower()
            for marker in (
                "here are",
                "practice questions",
                "quiz questions",
                "test questions",
                "multiple choice",
            )
        ):
            continue

        stem_key = _normalize_question(question_text)
        if stem_key in mcq_stems:
            continue

        lower_question = question_text.lower()
        question_type = (
            "essay"
            if re.match(
                r"^(explain|discuss|analyze|compare|evaluate|argue|describe|assess)\b",
                lower_question,
            )
            else "short_answer"
        )

        candidates.append(
            {
                "question": question_text,
                "question_type": question_type,
            }
        )

    # Essay-style prompts are sometimes imperative and may not end in '?'.
    essay_prompt_pattern = re.compile(
        r"(?im)^\s*(?:question\s*\d+[:\-]\s*|\d+[\).:-]\s*)?"
        r"((?:explain|discuss|analyze|compare|evaluate|argue|describe|assess)\b.{8,320})$"
    )
    for match in essay_prompt_pattern.finditer(normalized_source):
        question_text = _clean_question_candidate(match.group(1))
        if question_text:
            candidates.append(
                {
                    "question": question_text,
                    "question_type": "essay",
                }
            )

    return _dedupe_assessment_items(candidates)[:25]


def extract_assessment_questions(text: str) -> list[str]:
    """Backward-compatible wrapper returning only extracted question text."""
    return [item["question"] for item in extract_assessment_items(text)]


class TutorPersistence:
    """Persist conversation history and question banks across app restarts."""

    def __init__(
        self,
        *,
        subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
        data_dir: str | Path | None = None,
        max_messages_per_subject: int = 400,
    ):
        self.subjects = tuple(subjects)
        self.max_messages_per_subject = max(20, int(max_messages_per_subject))
        project_root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_dir) if data_dir else (project_root / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.question_bank_path = self.data_dir / "question_bank.json"
        self.conversation_state_path = self.data_dir / "conversation_state.json"
        self._lock = threading.Lock()

        self._ensure_file(self.question_bank_path, self._default_question_payload())
        self._ensure_file(
            self.conversation_state_path,
            self._default_conversation_payload(),
        )

    def _default_question_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "subjects": {subject: [] for subject in self.subjects},
        }

    def _default_conversation_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "current_subject": "english",
            "subjects": {subject: [] for subject in self.subjects},
        }

    def _ensure_file(self, path: Path, default_payload: dict[str, Any]) -> None:
        if path.exists():
            return
        self._write_json(path, default_payload)

    def _read_json(self, path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return fallback

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        tmp.replace(path)

    def load_conversation(self) -> tuple[dict[str, list[dict[str, Any]]], str]:
        """Load persisted conversation state by subject."""
        with self._lock:
            payload = self._read_json(
                self.conversation_state_path,
                self._default_conversation_payload(),
            )

        subjects_payload = payload.get("subjects") if isinstance(payload, dict) else {}
        if not isinstance(subjects_payload, dict):
            subjects_payload = {}

        normalized_subjects: dict[str, list[dict[str, Any]]] = {}
        for subject in self.subjects:
            messages = subjects_payload.get(subject, [])
            if isinstance(messages, list):
                normalized_subjects[subject] = [
                    msg for msg in messages if isinstance(msg, dict)
                ]
            else:
                normalized_subjects[subject] = []

        current_subject = str(payload.get("current_subject", "english")).lower()
        if current_subject not in self.subjects:
            current_subject = "english"

        return normalized_subjects, current_subject

    def set_current_subject(self, subject: str) -> None:
        """Persist the currently active subject."""
        if subject not in self.subjects:
            return

        with self._lock:
            payload = self._read_json(
                self.conversation_state_path,
                self._default_conversation_payload(),
            )
            subjects_payload = payload.get("subjects")
            if not isinstance(subjects_payload, dict):
                subjects_payload = {subject_name: [] for subject_name in self.subjects}
                payload["subjects"] = subjects_payload

            payload["current_subject"] = subject
            payload["updated_at"] = _utc_now_iso()
            self._write_json(self.conversation_state_path, payload)

    def append_turn(self, subject: str, user_text: str, assistant_text: str) -> None:
        """Append a completed user/assistant turn to persisted conversation."""
        if subject not in self.subjects:
            return

        user_text = str(user_text).strip()
        assistant_text = str(assistant_text).strip()
        if not user_text and not assistant_text:
            return

        now_iso = _utc_now_iso()

        with self._lock:
            payload = self._read_json(
                self.conversation_state_path,
                self._default_conversation_payload(),
            )
            subjects_payload = payload.get("subjects")
            if not isinstance(subjects_payload, dict):
                subjects_payload = {subject_name: [] for subject_name in self.subjects}

            subject_messages = subjects_payload.get(subject)
            if not isinstance(subject_messages, list):
                subject_messages = []

            if user_text:
                subject_messages.append(
                    {
                        "role": "human",
                        "content": user_text,
                        "timestamp": now_iso,
                    }
                )

            if assistant_text:
                subject_messages.append(
                    {
                        "role": "ai",
                        "content": assistant_text,
                        "timestamp": now_iso,
                    }
                )

            if len(subject_messages) > self.max_messages_per_subject:
                subject_messages = subject_messages[-self.max_messages_per_subject :]

            subjects_payload[subject] = subject_messages
            payload["subjects"] = subjects_payload
            payload["current_subject"] = subject
            payload["updated_at"] = now_iso

            self._write_json(self.conversation_state_path, payload)

    def load_question_bank(self) -> dict[str, list[dict[str, Any]]]:
        """Load persisted question bank grouped by subject."""
        with self._lock:
            payload = self._read_json(
                self.question_bank_path,
                self._default_question_payload(),
            )

        subjects_payload = payload.get("subjects") if isinstance(payload, dict) else {}
        if not isinstance(subjects_payload, dict):
            subjects_payload = {}

        questions_by_subject: dict[str, list[dict[str, Any]]] = {}
        for subject in self.subjects:
            entries = subjects_payload.get(subject, [])
            if isinstance(entries, list):
                questions_by_subject[subject] = [
                    entry for entry in entries if isinstance(entry, dict)
                ]
            else:
                questions_by_subject[subject] = []

        return questions_by_subject

    def upsert_questions(
        self,
        subject: str,
        questions: list[str] | list[dict[str, Any]],
        *,
        source_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Insert/update questions for a subject and return changed records."""
        if subject not in self.subjects:
            return []

        cleaned_questions: list[dict[str, str]] = []
        for question in questions:
            if isinstance(question, dict):
                raw_text = str(question.get("question", "")).strip()
                raw_type = str(question.get("question_type", "unknown")).strip().lower()
                if raw_text:
                    cleaned_questions.append(
                        {
                            "question": raw_text,
                            "question_type": raw_type if raw_type else "unknown",
                        }
                    )
                continue

            raw_text = str(question).strip()
            if raw_text:
                cleaned_questions.append(
                    {
                        "question": raw_text,
                        "question_type": "unknown",
                    }
                )

        if not cleaned_questions:
            return []

        now_iso = _utc_now_iso()
        source_prompt = str(source_prompt).strip()
        changed_records: list[dict[str, Any]] = []

        with self._lock:
            payload = self._read_json(
                self.question_bank_path,
                self._default_question_payload(),
            )

            subjects_payload = payload.get("subjects")
            if not isinstance(subjects_payload, dict):
                subjects_payload = {subject_name: [] for subject_name in self.subjects}

            entries = subjects_payload.get(subject)
            if not isinstance(entries, list):
                entries = []

            index = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = _normalize_question(str(entry.get("question", "")))
                if key:
                    index[key] = entry

            for question_data in cleaned_questions:
                question_text = question_data["question"]
                question_type = question_data.get("question_type", "unknown")
                key = _normalize_question(question_text)
                if not key:
                    continue

                existing = index.get(key)
                if existing:
                    existing["times_seen"] = int(existing.get("times_seen", 1)) + 1
                    existing["updated_at"] = now_iso
                    existing_type = (
                        str(existing.get("question_type", "")).strip().lower()
                    )
                    if question_type and existing_type in {"", "unknown"}:
                        existing["question_type"] = question_type
                    if source_prompt:
                        existing["last_source_prompt"] = source_prompt
                    changed_records.append(existing)
                    continue

                record = {
                    "id": str(uuid.uuid4()),
                    "question": question_text,
                    "question_type": question_type,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                    "times_seen": 1,
                    "last_source_prompt": source_prompt,
                }
                entries.append(record)
                index[key] = record
                changed_records.append(record)

            subjects_payload[subject] = entries
            payload["subjects"] = subjects_payload
            payload["updated_at"] = now_iso

            self._write_json(self.question_bank_path, payload)

        return changed_records
