"""Unit tests for note taker question filtering and persistence."""

import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from note_taker_agent import QuestionNoteTakerAgent
from persistence import TutorPersistence


class TestQuestionNoteTakerAgent(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.persistence = TutorPersistence(data_dir=self.tmp_dir.name)
        self.note_taker = QuestionNoteTakerAgent(self.persistence)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_persists_real_multiple_choice_question(self):
        response = (
            "Here is a challenging MCQ: **Question:** Which of the following best "
            "describes osmosis? **A)** Water moving from high to low solute concentration "
            "**B)** Water moving from low to high solute concentration "
            "**C)** Glucose moving through active transport "
            "**D)** Protein synthesis in ribosomes"
        )

        changed = self.note_taker.persist_from_response(
            "chemistry",
            response,
            source_prompt="Come up with another multiple choice question.",
        )

        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["question_type"], "multiple_choice")
        self.assertIn("A)", changed[0]["question"])

    def test_ignores_numbered_study_tips_not_questions(self):
        response = (
            "To approach multiple choice questions: \n"
            "1. Analyze the stimulus carefully.\n"
            "2. Read all options before selecting.\n"
            "3. Eliminate clearly incorrect choices."
        )

        changed = self.note_taker.persist_from_response(
            "history",
            response,
            source_prompt="How do I approach multiple choice questions?",
        )

        self.assertEqual(changed, [])

    def test_ignores_conversational_branching_questions(self):
        response = (
            "It seems we shifted topics. To get back on track, would you like to:\n"
            "1) Review the questions we discussed?\n"
            "2) Move on to a new set of questions?\n"
            "3) Discuss a specific topic in AP World History?"
        )

        changed = self.note_taker.persist_from_response(
            "history",
            response,
            source_prompt="Help me identify themes in Joy Luck Club",
        )

        self.assertEqual(changed, [])


if __name__ == "__main__":
    unittest.main()
