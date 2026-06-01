from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def detector_config(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    return (config.get("detectors") or {}).get(name, {})

