import json
import tempfile
import unittest
from pathlib import Path

from img_censor.review_queue import maybe_enqueue_review
from img_censor.schemas import GuardRequest, GuardResult, Verdict


class ReviewQueueTest(unittest.TestCase):
    def test_review_result_is_written_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "review.jsonl"
            result = GuardResult(
                verdict=Verdict.REVIEW,
                categories=["financial_misleading"],
                rationale="Needs manual review.",
                findings=[],
            )
            request = GuardRequest(prompt="проверь доходность", request_id="r-42", scenario="text2image")

            maybe_enqueue_review(result, request, str(queue_path))

            payload = json.loads(queue_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["request"]["request_id"], "r-42")
            self.assertEqual(payload["result"]["verdict"], "review")


if __name__ == "__main__":
    unittest.main()
