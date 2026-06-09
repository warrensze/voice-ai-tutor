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

            asset = manager.add_asset(source, subject="math", title="Algebra Notes")
            duplicate = manager.add_asset(source, subject="math", title="Duplicate")

            self.assertFalse(asset["duplicate"])
            self.assertTrue(duplicate["duplicate"])
            self.assertEqual(duplicate["id"], asset["id"])

            with (
                patch(
                    "rag_library.index_documents",
                    side_effect=lambda docs, settings=None: len(docs),
                ) as mock_index,
                patch("rag_library.delete_documents_for_asset") as mock_delete,
            ):
                indexed = manager.index_asset(asset["id"], settings=UserSettings())
                self.assertEqual(indexed["status"], "ready")
                self.assertGreater(indexed["chunk_count"], 0)
                self.assertTrue(mock_index.called)

                preview = manager.preview_asset(asset["id"])
                self.assertIn("Quadratic equations", preview)

                removed = manager.remove_asset(asset["id"], settings=UserSettings())
                self.assertIsNotNone(removed)
                self.assertTrue(mock_delete.called)
                self.assertEqual(manager.list_assets(), [])


if __name__ == "__main__":
    unittest.main()
