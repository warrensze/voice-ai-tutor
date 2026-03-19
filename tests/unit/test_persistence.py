"""Unit tests for persistence-backed conversation and question storage."""

import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from persistence import (
    TutorPersistence,
    extract_assessment_items,
    extract_assessment_questions,
)


class TestQuestionExtraction(unittest.TestCase):
    def test_extracts_numbered_questions_with_assessment_hints(self):
        text = (
            "Here are 3 practice questions for your quiz:\n"
            "1) What caused World War I?\n"
            "2) How did alliances escalate the conflict?\n"
            "3) Explain total war in one sentence?"
        )
        result = extract_assessment_questions(text)
        self.assertEqual(len(result), 3)
        self.assertIn("What caused World War I?", result)

    def test_ignores_single_conversational_question_without_hint(self):
        text = "Can you clarify what you mean by that?"
        result = extract_assessment_questions(text)
        self.assertEqual(result, [])

    def test_ignores_multiple_conversational_questions_without_hint(self):
        text = "What is stoichiometry? Why is molarity important?"
        result = extract_assessment_questions(text)
        self.assertEqual(result, [])

    def test_extracts_multiple_choice_block(self):
        text = (
            "Multiple choice quiz:\n"
            "1) Which gas is most abundant in Earth's atmosphere?\n"
            "A) Oxygen\n"
            "B) Nitrogen\n"
            "C) Carbon Dioxide\n"
            "D) Argon"
        )
        items = extract_assessment_items(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question_type"], "multiple_choice")
        self.assertIn("A) Oxygen", items[0]["question"])

    def test_extracts_fill_blank_and_essay_items(self):
        text = (
            "Practice test items:\n"
            "2) Fill in the blank: The powerhouse of the cell is the ________.\n"
            "3) Essay: Analyze the impact of industrialization on urban life."
        )
        items = extract_assessment_items(text)
        types = {item["question_type"] for item in items}
        self.assertIn("fill_in_the_blank", types)
        self.assertIn("essay", types)

    def test_extracts_numbered_prompts_when_source_prompt_requests_quiz(self):
        text = (
            "1) Explain how mitosis differs from meiosis.\n"
            "2) Describe two causes of inflation.\n"
            "3) Analyze the author's tone in this paragraph."
        )
        items = extract_assessment_items(
            text,
            source_prompt="Create three quiz questions for me.",
        )
        self.assertEqual(len(items), 3)
        self.assertIn("essay", {item["question_type"] for item in items})

    def test_extracts_markdown_question_labels_and_option_bullets(self):
        text = (
            "Here are two practice multiple choice questions:\n"
            "**Question 1:** Which organelle generates ATP\n"
            "- A) Nucleus\n"
            "- B) Mitochondria\n"
            "- C) Ribosome\n"
            "- D) Golgi apparatus\n\n"
            "**Question 2:** Which process moves water across a semipermeable membrane\n"
            "- A) Diffusion\n"
            "- B) Osmosis\n"
            "- C) Active transport\n"
            "- D) Endocytosis"
        )
        items = extract_assessment_items(text)

        self.assertEqual(len(items), 2)
        self.assertEqual(
            [item["question_type"] for item in items],
            ["multiple_choice", "multiple_choice"],
        )
        self.assertIn("A) Nucleus", items[0]["question"])
        self.assertIn("B) Osmosis", items[1]["question"])

    def test_extracts_mcq_numbered_stems_without_question_marks(self):
        text = (
            "1. Which statement best describes mitosis\n"
            "A. It creates gametes\n"
            "B. It creates two genetically identical daughter cells\n"
            "C. It halves chromosome number\n"
            "D. It occurs only in sex cells\n\n"
            "2. Which molecule stores genetic instructions\n"
            "A. ATP\n"
            "B. DNA\n"
            "C. RNA polymerase\n"
            "D. Hemoglobin"
        )
        items = extract_assessment_items(
            text,
            source_prompt="Give me two multiple choice questions.",
        )

        self.assertEqual(len(items), 2)
        self.assertTrue(
            all(item["question_type"] == "multiple_choice" for item in items)
        )
        self.assertIn("B) DNA", items[1]["question"])

    def test_extracts_inline_single_paragraph_mcq_format(self):
        text = (
            "Here is a challenging MCQ: **Question:** Which event most directly "
            "triggered World War I? **A)** The Russian Revolution **B)** The assassination "
            "of Archduke Franz Ferdinand **C)** The Treaty of Versailles **D)** The rise "
            "of fascism in Italy"
        )
        items = extract_assessment_items(text, source_prompt="Give me one MCQ")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question_type"], "multiple_choice")
        self.assertIn(
            "B) The assassination of Archduke Franz Ferdinand", items[0]["question"]
        )


class TestTutorPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.subjects = ("history", "chemistry", "math", "english")
        self.persistence = TutorPersistence(
            subjects=self.subjects,
            data_dir=self.tmp_dir.name,
            max_messages_per_subject=10,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_append_turn_and_load_conversation(self):
        self.persistence.append_turn(
            "history",
            "Can you quiz me on AP World?",
            "Sure. Here is question 1: What caused World War I?",
        )
        by_subject, current_subject = self.persistence.load_conversation()

        self.assertEqual(current_subject, "history")
        self.assertIn("history", by_subject)
        self.assertEqual(len(by_subject["history"]), 2)
        self.assertEqual(by_subject["history"][0]["role"], "human")
        self.assertEqual(by_subject["history"][1]["role"], "ai")

    def test_conversation_persists_across_instances(self):
        self.persistence.append_turn(
            "math",
            "Review derivatives with me.",
            "Let's begin with the power rule.",
        )

        restarted = TutorPersistence(
            subjects=self.subjects,
            data_dir=self.tmp_dir.name,
            max_messages_per_subject=10,
        )
        by_subject, current_subject = restarted.load_conversation()

        self.assertEqual(current_subject, "math")
        self.assertEqual(len(by_subject["math"]), 2)
        self.assertEqual(
            by_subject["math"][0]["content"], "Review derivatives with me."
        )
        self.assertEqual(
            by_subject["math"][1]["content"], "Let's begin with the power rule."
        )

    def test_upsert_questions_deduplicates_with_counter(self):
        first = self.persistence.upsert_questions(
            "history",
            [
                {
                    "question": "What caused World War I?",
                    "question_type": "short_answer",
                },
                {
                    "question": "Explain total war.",
                    "question_type": "essay",
                },
            ],
            source_prompt="Quiz me",
        )
        second = self.persistence.upsert_questions(
            "history",
            [
                {
                    "question": "What caused World War I?",
                    "question_type": "short_answer",
                }
            ],
            source_prompt="Re-ask",
        )

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)

        bank = self.persistence.load_question_bank()
        self.assertEqual(len(bank["history"]), 2)

        question = [
            q for q in bank["history"] if q["question"] == "What caused World War I?"
        ][0]
        self.assertEqual(question["times_seen"], 2)
        self.assertEqual(question["last_source_prompt"], "Re-ask")
        self.assertEqual(question["question_type"], "short_answer")

    def test_set_current_subject(self):
        self.persistence.set_current_subject("chemistry")
        _, current_subject = self.persistence.load_conversation()
        self.assertEqual(current_subject, "chemistry")


if __name__ == "__main__":
    unittest.main()
