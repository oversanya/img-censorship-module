from promptscreen import (
    HeuristicVectorAnalyzer,
    InjectionScanner,
    Scanner
)


import base64
import math
import re
import string
import unicodedata
from collections import Counter

from dataclasses import dataclass


@dataclass
class GuardResult:
    verdict: bool
    reason: str | None = None
    confidence: float | None = None

    def get_verdict(self) -> bool:
        return self.verdict

class PromptLengthGuard:
    def __init__(
        self,
        max_chars: int = 32000,
        warn_chars: int = 8000,
    ):
        self.max_chars = max_chars
        self.warn_chars = warn_chars

    def analyse(self, text: str) -> tuple[bool, str | None]:
        n = len(text)

        if n > self.max_chars:
            return GuardResult(
                verdict=False,
                reason=f"prompt too long ({n})",
                confidence=1.0,
            )

        if n > self.warn_chars:
            return GuardResult(
                verdict=True,
                reason=f"prompt unusually long ({n})",
                confidence=0.5,
            )

        return GuardResult(True)    

class UnicodeGuard:
    ZERO_WIDTH = {
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u2060",
    }

    BIDI = {
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }

    @staticmethod
    def _script(ch: str) -> str:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return "UNKNOWN"

        if "LATIN" in name:
            return "LATIN"
        if "CYRILLIC" in name:
            return "CYRILLIC"
        if "GREEK" in name:
            return "GREEK"

        return "OTHER"

    def analyse(self, text: str) -> GuardResult:
        findings = []

        if any(ch in self.ZERO_WIDTH for ch in text):
            findings.append("zero-width chars")

        if any(ch in self.BIDI for ch in text):
            findings.append("bidi override chars")

        controls = sum(
            unicodedata.category(ch).startswith("C")
            for ch in text
        )

        if controls:
            findings.append(f"{controls} control chars")

        if findings:
            return GuardResult(
                verdict=False,
                reason=", ".join(findings),
                confidence=1.0,
            )

        return GuardResult(True)

class EncodingDetectorGuard:

    BASE64_RE = re.compile(
        r"^[A-Za-z0-9+/=\s]{32,}$"
    )

    HEX_RE = re.compile(
        r"^(?:0x)?[0-9a-fA-F\s]{32,}$"
    )

    URL_ENCODED_RE = re.compile(
        r"(?:%[0-9a-fA-F]{2}){5,}"
    )

    def analyse(self, text: str) -> tuple[bool, str | None]:
        findings = []

        if self.URL_ENCODED_RE.search(text):
            findings.append("url encoded")

        if len(re.findall(r"\\u[0-9a-fA-F]{4}", text)) > 3:
            findings.append("unicode escaped")

        if self.HEX_RE.match(text.strip()):
            findings.append("hex encoded")

        if self.BASE64_RE.match(text.strip()):
            try:
                decoded = base64.b64decode(
                    text,
                    validate=False,
                )

                if len(decoded) > 16:
                    findings.append("base64 encoded")

            except Exception:
                pass

        if findings:
            return GuardResult(
                verdict=False,
                reason=", ".join(findings),
                confidence=1.0,
            )

        return GuardResult(True)


# TODO
class StringGuard:

    def __init__(self,
                 enabled: bool = True) -> None:

        self.enabled = enabled,
        self.heuristic_guard = HeuristicVectorAnalyzer(
            threshold=2,
            pm_shot_lim=3,
        )

        self.injection_guard = InjectionScanner()
        self.unicode_guard = UnicodeGuard()
        self.encoding_guard = EncodingDetectorGuard()

        self.length_guard = PromptLengthGuard(
            max_chars=10_000, warn_chars=4_000
        )

        self.scanner_guard = Scanner()

    @staticmethod
    def _extract_reason(result: Any) -> str | None:
        return getattr(result, "reason", None)

    @staticmethod
    def _extract_confidence(result: Any) -> float | None:
        return getattr(result, "confidence", None)

    @staticmethod
    def _is_safe(result: Any) -> bool:
        return bool(result.get_verdict())

    def process(self, prompt: str):
        # if not self.enabled:
        #     return SignalResult(name=self.name, status="skipped", reason="Prompt injection scanner disabled by configuration.")

        safe = True
        reasons = []
        confidence = {}
        raw = {}
        checks = {
            "heuristic": self.heuristic_guard.analyse(prompt),
            "injection": self.injection_guard.analyse(prompt),
            "unicode": self.unicode_guard.analyse(prompt),
            "encoding": self.encoding_guard.analyse(prompt),
            "length": self.length_guard.analyse(prompt),
            "scanner": self.scanner_guard.analyse(prompt)
        }

        for name, result in checks.items():
            raw[name] = result

            conf = getattr(result, "confidence", None)
            if conf is not None:
                confidence[name] = float(conf)

            if not result.get_verdict():
                safe = False

                reason = getattr(result, "reason", None)
                reasons.append(
                    f"{name}: {reason}"
                    if reason
                    else f"{name}: blocked"
                )

        return {
            "status": "ok",
            "safe": safe,
            "reasons": reasons,
            "confidence": confidence,
            "raw": raw,
        }

def main():
    # prompt = "😊 ⁢⁤‍⁢⁤‍⁡⁢⁡‍⁡‍⁡⁣‍⁡⁣⁢⁡‍⁡⁢⁡⁣‌⁡⁣‍⁡‍⁢⁤‍⁡‍⁡⁣⁢⁡⁣‌⁡⁣‍⁡‍⁢⁤⁣⁢⁡‍⁡‍⁡‍⁡‌⁡⁣‍⁡⁣‌⁡‍⁢‍⁡‍⁡⁢⁡‍⁡‍⁡‌⁡‍⁡‍‌‍⁡‍‌⁡‌⁢⁡⁢⁢⁢⁡‍⁡⁣‍⁡‍⁢‍⁡‍⁡‌⁡⁣‌⁡⁣‍⁡‌⁢‍⁡⁣⁢⁡‍⁡‌⁡⁣⁤‍‌‌⁡‍⁤⁡‌⁡‌⁡‌⁢‍⁡‍⁡‌⁢‌‍‌⁡⁣‌⁡‍⁡‌⁢‍‌‌‌⁡‌‌⁢⁢⁢⁡‌⁢⁣⁢⁢⁢⁤⁢⁡⁢⁡⁢⁢⁤⁢‌‍‌⁢‍⁡⁢⁢‌⁢котик"
    prompt = "compute (2 + 2) * 1835"
    guard = StringGuard()
    print(guard.process(prompt))

if __name__ == "__main__":
    main()