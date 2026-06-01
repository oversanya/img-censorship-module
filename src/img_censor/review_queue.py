import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from img_censor.schemas import GuardRequest, GuardResult, Verdict


def maybe_enqueue_review(result: GuardResult, request: GuardRequest, queue_path: Optional[str]) -> None:
    if result.verdict != Verdict.REVIEW or not queue_path:
        return

    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": request.to_audit_dict(),
        "result": result.to_dict(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
