"""Dedicated note-taker agent for persisting tutor-generated assessment items."""

from __future__ import annotations

import re
from typing import Any

from persistence import TutorPersistence, extract_assessment_items

_GENERATION_REQUEST_PATTERN = re.compile(
    r"(?i)(?:\b(?:create|generate|write|make|build|produce|give|provide|ask)\b.{0,40}"
    r"\b(?:question|questions|quiz|test|exam|mcq|multiple choice|prompt|practice)\b"
    r"|\b(?:quiz me|test me|ask me|come up with|another question|new set of questions)\b)"
)

_EXPLICIT_ASSESSMENT_STRUCTURE_PATTERN = re.compile(
    r"(?im)(?:\bquestion\s*\d*\b|^\s*\d+[\).:-]\s+|^\s*(?:\*\*)?\(?[a-h]\)?[\).:-]\s+\S|\bmultiple\s+choice\b)"
)

_CONVERSATIONAL_PROMPT_PATTERN = re.compile(
    r"(?i)(?:would you like|do you want|if you'd like|let me know|what would you like|"
    r"are you preparing|to get back on track|move on to|review the|continue with)"
)

_EXAM_STEM_START_PATTERN = re.compile(
    r"(?i)^(?:which|what|who|when|where|why|how|identify|explain|describe|analyze|"
    r"compare|evaluate|calculate|solve|determine|assess|select|choose|discuss)\b"
)

_META_STEM_PATTERN = re.compile(
    r"(?i)(?:specific topic|new topic|next topic|set of questions|review the questions)"
)


class QuestionNoteTakerAgent:
    """Extract and persist only exam/test style questions from tutor output."""

    def __init__(self, persistence: TutorPersistence):
        self._persistence = persistence

    def _is_generation_request(self, source_prompt: str) -> bool:
        prompt = str(source_prompt or "").strip()
        if not prompt:
            return False
        return bool(_GENERATION_REQUEST_PATTERN.search(prompt))

    def _has_explicit_assessment_structure(self, tutor_response: str) -> bool:
        response = str(tutor_response or "").strip()
        if not response:
            return False
        return bool(_EXPLICIT_ASSESSMENT_STRUCTURE_PATTERN.search(response))

    def _looks_like_exam_stem(self, stem: str) -> bool:
        normalized = " ".join(str(stem or "").split()).strip()
        if not normalized:
            return False

        lower = normalized.lower()
        if _CONVERSATIONAL_PROMPT_PATTERN.search(lower):
            return False
        if _META_STEM_PATTERN.search(lower):
            return False

        if re.match(
            r"(?i)^(?:are|is|do|did|can|could|would|should|will)\b", normalized
        ):
            if " you " in f" {lower} ":
                return False

        if _EXAM_STEM_START_PATTERN.match(normalized):
            return True

        if "which of the following" in lower:
            return True

        if normalized.endswith("?") and len(normalized.split()) >= 5:
            return True

        return False

    def extract_questions(
        self,
        tutor_response: str,
        *,
        source_prompt: str = "",
    ) -> list[dict[str, str]]:
        """Return filtered assessment items suitable for persistence."""
        response = str(tutor_response or "").strip()
        if not response:
            return []

        generation_requested = self._is_generation_request(source_prompt)
        explicit_structure = self._has_explicit_assessment_structure(response)
        if not generation_requested and not explicit_structure:
            return []

        candidates = extract_assessment_items(response, source_prompt=source_prompt)
        if not candidates:
            return []

        accepted: list[dict[str, str]] = []
        seen: set[str] = set()

        for item in candidates:
            question = str(item.get("question", "")).strip()
            if not question:
                continue

            question_type = str(item.get("question_type", "unknown")).strip().lower()
            if not question_type:
                question_type = "unknown"

            stem = question.split("\n", 1)[0].strip()
            stem_key = re.sub(r"\s+", " ", stem.lower()).strip().rstrip("?.! ")
            if not stem_key or stem_key in seen:
                continue

            if question_type == "multiple_choice":
                if not re.search(r"(?im)^\s*[a-h][\).:-]\s+\S", question):
                    continue
                if not self._looks_like_exam_stem(stem):
                    continue
                accepted.append({"question": question, "question_type": question_type})
                seen.add(stem_key)
                continue

            if question_type == "fill_in_the_blank":
                if (
                    not generation_requested
                    and "blank" not in question.lower()
                    and "___" not in question
                ):
                    continue
                if not self._looks_like_exam_stem(stem):
                    continue
                accepted.append({"question": question, "question_type": question_type})
                seen.add(stem_key)
                continue

            if question_type in {"short_answer", "essay", "unknown"}:
                if not generation_requested:
                    continue
                if not self._looks_like_exam_stem(stem):
                    continue
                accepted.append({"question": question, "question_type": question_type})
                seen.add(stem_key)

        return accepted

    def persist_from_response(
        self,
        subject: str,
        tutor_response: str,
        *,
        source_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Extract and upsert tutor-generated assessment items into storage."""
        extracted = self.extract_questions(
            tutor_response,
            source_prompt=source_prompt,
        )
        if not extracted:
            return []

        return self._persistence.upsert_questions(
            subject,
            extracted,
            source_prompt=source_prompt,
        )
