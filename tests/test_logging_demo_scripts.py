from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_logging_demo.py"
VERIFY_PATH = ROOT / "scripts" / "verify_logging_demo.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LoggingDemoScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_module("run_logging_demo_for_tests", RUNNER_PATH)
        cls.verify = load_module("verify_logging_demo_for_tests", VERIFY_PATH)

    def test_report_includes_jsonl_paths_outputs_and_final_status(self) -> None:
        state = self.runner.DemoState(log_dir=Path("logs/logging_demo"))
        state.status = "PASS"
        state.commands.append(
            self.runner.CommandResult(
                title="Run JSONL logging verification",
                command="python scripts/verify_logging_demo.py --clean",
                returncode=0,
                stdout='{"status":"PASS","counts":{"system_jsonl_rows":3}}',
                stderr="",
            )
        )
        state.verification = {
            "status": "PASS",
            "counts": {
                "system_jsonl_rows": 3,
                "business_audit_jsonl_rows": 1,
                "raw_payloads_jsonl_rows": 1,
            },
        }

        report = self.runner.render_report(state)

        self.assertIn("# JSONL Logging Demo Run", report)
        self.assertIn("Log dir", report)
        self.assertIn("python scripts/verify_logging_demo.py --clean", report)
        self.assertIn('"system_jsonl_rows": 3', report)
        self.assertIn('"business_audit_jsonl_rows": 1', report)
        self.assertIn('"raw_payloads_jsonl_rows": 1', report)
        self.assertIn("Final result: `PASS`", report)

    def test_read_jsonl_parses_records_and_ignores_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "one"}),
                        "",
                        json.dumps({"event": "two"}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                self.verify.read_jsonl(path),
                [{"event": "one"}, {"event": "two"}],
            )

    def test_verification_writes_and_reads_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.verify.run_verification(Path(tmp), clean=True)

            self.assertEqual(summary["status"], "PASS")
            self.assertTrue((Path(tmp) / "system.jsonl").is_file())
            self.assertTrue((Path(tmp) / "business_audit.jsonl").is_file())
            self.assertTrue((Path(tmp) / "raw_payloads.jsonl").is_file())
            self.assertGreaterEqual(summary["counts"]["system_jsonl_rows"], 1)
            self.assertEqual(summary["counts"]["business_audit_jsonl_rows"], 1)
            self.assertEqual(summary["counts"]["raw_payloads_jsonl_rows"], 1)
            self.assertTrue(summary["audit_reason_code"])

    def test_write_report_creates_markdown_file(self) -> None:
        state = self.runner.DemoState()
        with tempfile.TemporaryDirectory() as tmp:
            original = self.runner.REPORT_PATH
            try:
                self.runner.REPORT_PATH = Path(tmp) / "logging_demo_run.md"
                self.runner.write_report(state)
                self.assertTrue(self.runner.REPORT_PATH.is_file())
                self.assertIn("Final result", self.runner.REPORT_PATH.read_text(encoding="utf-8"))
            finally:
                self.runner.REPORT_PATH = original


if __name__ == "__main__":
    unittest.main()
