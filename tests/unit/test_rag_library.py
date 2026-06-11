import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_library import LibraryManager
from settings_store import UserSettings


class TestLibraryManager(unittest.TestCase):
    def test_add_duplicate_index_preview_and_remove_text_asset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "algebra_notes.txt"
            source.write_text(
                "Quadratic equations can be solved by factoring or by using the quadratic formula.",
                encoding="utf-8",
            )
            manager = LibraryManager(
                library_dir=root / "library",
                index_path=root / "library" / "library_index.json",
            )

            asset = manager.add_asset(
                source,
                subject="math",
                title="Algebra Notes",
                course="algebra_ii",
                source_role="workbook",
                topic_tags="quadratics, factoring",
            )
            duplicate = manager.add_asset(source, subject="math", title="Duplicate")

            self.assertFalse(asset["duplicate"])
            self.assertEqual(asset["course"], "algebra_ii")
            self.assertEqual(asset["source_role"], "workbook")
            self.assertEqual(asset["topic_tags"], "quadratics, factoring")
            self.assertTrue(duplicate["duplicate"])
            self.assertEqual(duplicate["id"], asset["id"])

            captured_documents = []

            def fake_index(docs, settings=None):
                captured_documents.extend(docs)
                return len(docs)

            with (
                patch(
                    "rag_library.index_documents",
                    side_effect=fake_index,
                ) as mock_index,
                patch("rag_library.delete_documents_for_asset") as mock_delete,
            ):
                indexed = manager.index_asset(asset["id"], settings=UserSettings())
                self.assertEqual(indexed["status"], "ready")
                self.assertGreater(indexed["chunk_count"], 0)
                self.assertTrue(mock_index.called)
                self.assertEqual(
                    captured_documents[0].metadata["course"],
                    "algebra_ii",
                )
                self.assertEqual(
                    captured_documents[0].metadata["source_role"],
                    "workbook",
                )

                preview = manager.preview_asset(asset["id"])
                self.assertIn("Quadratic equations", preview)

                removed = manager.remove_asset(asset["id"], settings=UserSettings())
                self.assertIsNotNone(removed)
                self.assertTrue(mock_delete.called)
                self.assertEqual(manager.list_assets(), [])


if __name__ == "__main__":
    unittest.main()
