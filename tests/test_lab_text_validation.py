# -*- coding: utf-8 -*-
"""BUG-002 API 回归：临时数据库/输出目录，不读取或改写团队运行数据。

运行：python tests/test_lab_text_validation.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    with tempfile.TemporaryDirectory(prefix="lab-text-test-") as directory:
        for key, path in {
            "DB_PATH": "test.db", "UPLOAD_DIR": "uploads", "OUTPUTS_DIR": "outputs",
            "ANNOTATED_DIR": "annotated", "VIDEO_DIR": "videos",
            "SCREENSHOT_DIR": "screenshots",
        }.items():
            os.environ["LABSAFETY_" + key] = str(Path(directory) / path)

        from fastapi.testclient import TestClient
        from backend.main import app

        class LabTextValidationTests(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.client = TestClient(app)
                cls.client.__enter__()

            @classmethod
            def tearDownClass(cls):
                cls.client.__exit__(None, None, None)

            def assert_rejected(self, name, description, message):
                before = self.client.get("/api/settings").json()["labs"]
                response = self.client.post("/api/settings/lab", data={
                    "name": name, "description": description,
                })
                self.assertEqual(response.status_code, 400)
                self.assertIn(message, response.json()["error"])
                self.assertEqual(self.client.get("/api/settings").json()["labs"], before)

            def test_overlong_description(self):
                self.assert_rejected("超长说明测试", "中" * 501, "500")

            def test_overlong_name(self):
                self.assert_rejected("名" * 101, "", "100")

            def test_whitespace_name(self):
                self.assert_rejected("  \t  ", "说明", "不能为空")

            def test_replacement_characters(self):
                for name, description in (("实验室�", ""), ("编码测试", "�")):
                    with self.subTest(name=name):
                        self.assert_rejected(name, description, "编码替换字符")

            def test_control_characters(self):
                for name, description in (("实验\n室", ""), ("控制字符测试", "a\x00b"),
                                          ("控制字符测试", "a\x7fb")):
                    with self.subTest(name=name, description=repr(description)):
                        self.assert_rejected(name, description, "控制字符")

            def test_exact_boundaries_and_unicode_roundtrip(self):
                name = "边" * 99 + "🧪"
                description = "说" * 499 + "🧪"
                response = self.client.post("/api/settings/lab", data={
                    "name": name, "description": description,
                })
                self.assertEqual(response.status_code, 200)
                labs = self.client.get("/api/settings").json()["labs"]
                saved = next(lab for lab in labs if lab["name"] == name)
                self.assertEqual(saved["description"], description)

            def test_optional_description_and_duplicate(self):
                self.assertEqual(self.client.post("/api/settings/lab", data={
                    "name": "正常中文 English 🧪",
                }).status_code, 200)
                self.assert_rejected("正常中文 English 🧪", "", "已存在")

            def test_multiline_description_roundtrip(self):
                description = "第一行\r\n第二行\tEnglish 🧪"
                response = self.client.post("/api/settings/lab", data={
                    "name": "多行说明", "description": description,
                })
                self.assertEqual(response.status_code, 200)
                labs = self.client.get("/api/settings").json()["labs"]
                self.assertEqual(next(lab for lab in labs if lab["name"] == "多行说明")
                                 ["description"], description)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(LabTextValidationTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
