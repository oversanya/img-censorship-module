from typing import List

from img_censor.detectors.base import Detector
from img_censor.detectors.heuristics import PromptKeywordGuard
from img_censor.io import load_image
from img_censor.schemas import Finding, GuardRequest, Verdict


class QrCodeDetector(Detector):
    name = "qr_code"

    def run(self, request: GuardRequest) -> List[Finding]:
        if not self.enabled:
            return []

        findings: List[Finding] = []
        for stage, image_path in (
            ("input_image", request.input_image),
            ("output_image", request.output_image),
        ):
            if not image_path or not self.should_run_on(stage):
                continue
            payloads = self._decode_qr_payloads(image_path)
            for payload in payloads:
                findings.append(self._classify_payload(stage, payload))
        return findings

    def _decode_qr_payloads(self, image_path: str) -> List[str]:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return []

        image = load_image(image_path)
        array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()

        payloads = []
        ok, decoded_info, _, _ = detector.detectAndDecodeMulti(array)
        if ok:
            payloads.extend([payload for payload in decoded_info if payload])
        else:
            payload, _, _ = detector.detectAndDecode(array)
            if payload:
                payloads.append(payload)
        return payloads

    def _classify_payload(self, stage: str, payload: str) -> Finding:
        keyword_guard = PromptKeywordGuard(
            {"enabled": True, "stages": ["prompt"], "block_threshold": 1.0},
            self.runtime_config,
        )
        keyword_findings = keyword_guard.run(GuardRequest(prompt=payload))
        if keyword_findings:
            category = keyword_findings[0].category
            raw = keyword_findings[0].raw
        else:
            category = "fraudulent_qr_payment_push"
            raw = {}

        return Finding(
            detector=self.name,
            stage=stage,
            category=category,
            score=1.0,
            verdict=Verdict.BLOCK,
            rationale="QR code detected in image; payload requires payment/fraud review.",
            raw={
                **raw,
                "qr_payload": payload[:1000],
            },
        )

