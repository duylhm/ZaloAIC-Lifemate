import tempfile
import unittest
import zipfile
from pathlib import Path
from lifemate.bootstrap import stage_source, find_data, safe_extract


class BootstrapTests(unittest.TestCase):
    def test_directory_and_zip_have_same_source(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source = base / "input" / "source"
            (source / "lifemate").mkdir(parents=True)
            (source / "source_revision.txt").write_text("v2")
            (source / "lifemate" / "runner.py").write_text("pass\n")
            zip_path = base / "source.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                for p in source.rglob("*"):
                    if p.is_file():
                        z.write(p, "nested/" + p.relative_to(source).as_posix())
            a = stage_source(source, base / "working")
            b = stage_source(zip_path, base / "working")
            self.assertEqual(a, b)

    def test_zip_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            zpath = base / "evil.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("../escape.txt", "unsafe")
            with self.assertRaises(ValueError):
                safe_extract(zpath, base / "out")
            self.assertFalse((base / "escape.txt").exists())

    def test_nested_dataset_root(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            train = base / "dataset" / "train"
            (train / "zaic_tools").mkdir(parents=True)
            (train / "train.jsonl").write_text("")
            (train / "zaic_tools" / "__init__.py").touch()
            self.assertEqual(find_data(base, base / "working"), train.parent)
