import unittest

from vector import build_study_filter, combine_filters


class TestVectorStudySetFilters(unittest.TestCase):
    def test_math_course_filter_includes_legacy_algebra_book(self):
        result = build_study_filter(
            subject="math",
            course="algebra_ii",
            source_mode="auto",
        )

        self.assertEqual(
            result,
            {
                "$or": [
                    {"course": {"$eq": "algebra_ii"}},
                    {"source_file": {"$eq": "Algebra-2-Book.pdf"}},
                ]
            },
        )

    def test_textbook_mode_keeps_matching_legacy_alias(self):
        result = build_study_filter(
            subject="math",
            course="algebra_ii",
            source_mode="textbook",
        )

        self.assertEqual(
            result,
            {
                "$and": [
                    {
                        "$or": [
                            {"course": {"$eq": "algebra_ii"}},
                            {"source_file": {"$eq": "Algebra-2-Book.pdf"}},
                        ]
                    },
                    {
                        "$or": [
                            {"source_role": {"$eq": "textbook"}},
                            {"source_file": {"$eq": "Algebra-2-Book.pdf"}},
                        ]
                    },
                ]
            },
        )

    def test_workbook_mode_excludes_textbook_alias_by_role(self):
        result = build_study_filter(
            subject="math",
            course="algebra_ii",
            source_mode="workbook",
        )

        self.assertEqual(
            result,
            {
                "$and": [
                    {
                        "$or": [
                            {"course": {"$eq": "algebra_ii"}},
                            {"source_file": {"$eq": "Algebra-2-Book.pdf"}},
                        ]
                    },
                    {"source_role": {"$eq": "workbook"}},
                ]
            },
        )

    def test_combines_study_filter_with_subject(self):
        subject_filter = {"subject": {"$eq": "math"}}
        study_filter = build_study_filter(
            subject="math",
            course="precalculus",
            source_mode="auto",
        )

        self.assertEqual(
            combine_filters(None, subject_filter, study_filter),
            {
                "$and": [
                    {"subject": {"$eq": "math"}},
                    {"course": {"$eq": "precalculus"}},
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
