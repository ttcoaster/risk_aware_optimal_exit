import unittest,tempfile
from pathlib import Path
from src.evaluation.integrity_report import digest,require,audit
class TestIntegrityReport(unittest.TestCase):
    def test_hash_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'; p.write_bytes(b'a'); h=digest(p); p.write_bytes(b'b'); self.assertNotEqual(h,digest(p))
    def test_guard(self):
        with self.assertRaises(ValueError): require(False,'bad')
    def test_missing_source(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): audit(d,Path(d)/'missing',d)
