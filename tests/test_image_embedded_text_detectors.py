import unittest

from img_censor.detectors.ocr_text import OcrTextDetector
from img_censor.detectors.qr_code import QrCodeDetector
from img_censor.schemas import Verdict


class ImageEmbeddedTextDetectorsTest(unittest.TestCase):
    def test_ocr_text_is_mapped_to_policy_category(self):
        detector = OcrTextDetector({"enabled": True, "stages": ["input_image"]}, {})

        findings = detector._classify_text("input_image", "Нарисуй свастику")

        self.assertEqual(findings[0].detector, "ocr_text")
        self.assertEqual(findings[0].category, "extremism_terrorism_symbols")
        self.assertEqual(findings[0].verdict, Verdict.BLOCK)

    def test_qr_payload_is_blocked_as_payment_fraud_surface(self):
        detector = QrCodeDetector({"enabled": True, "stages": ["output_image"]}, {})

        finding = detector._classify_payload("output_image", "https://pay.example/transfer?amount=1000")

        self.assertEqual(finding.detector, "qr_code")
        self.assertEqual(finding.category, "fraudulent_qr_payment_push")
        self.assertEqual(finding.verdict, Verdict.BLOCK)


if __name__ == "__main__":
    unittest.main()
