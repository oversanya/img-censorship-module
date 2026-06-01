import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from img_censor.cli import main


class CliTest(unittest.TestCase):
    def test_reads_prompt_from_stdin_when_prompt_arg_missing(self):
        stdout = io.StringIO()

        with patch("sys.stdin", io.StringIO("Сгенерируй фото машины\n")):
            with redirect_stdout(stdout):
                exit_code = main(["--config", "configs/local.yaml"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"verdict": "allow"', stdout.getvalue())

    def test_interactive_mode_blocks_prompt(self):
        stdout = io.StringIO()

        with patch("builtins.input", side_effect=["Нарисуй свастику", ""]):
            with redirect_stdout(stdout):
                exit_code = main(["--config", "configs/local.yaml", "--interactive"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"verdict": "block"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
