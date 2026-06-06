from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "logging_demo_run.md"
DEFAULT_LOG_DIR = ROOT / "logs" / "logging_demo"


@dataclass
class CommandResult:
    title: str
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class DemoState:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    log_dir: Path = DEFAULT_LOG_DIR
    python_version: str = sys.version.replace("\n", " ")
    commands: list[CommandResult] = field(default_factory=list)
    verification: dict | None = None
    status: str = "FAIL"
    error: str | None = None


class DemoFailure(RuntimeError):
    pass


def stringify_command(command: Sequence[str] | str) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def decode_output(raw: bytes) -> str:
    if not raw:
        return ""
    for encoding in ("utf-8", "utf-16", "utf-16le", "cp866", "cp1251"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return text
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def run_command(
    state: DemoState,
    title: str,
    command: Sequence[str] | str,
    timeout_seconds: int = 120,
    cwd: Path = ROOT,
) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout_seconds,
    )
    result = CommandResult(
        title=title,
        command=stringify_command(command),
        returncode=completed.returncode,
        stdout=decode_output(completed.stdout).strip(),
        stderr=decode_output(completed.stderr).strip(),
    )
    state.commands.append(result)
    return result


def check_python_dependencies(state: DemoState) -> None:
    result = run_command(
        state,
        "Check Python dependencies",
        [
            sys.executable,
            "-c",
            "import pydantic, PIL; print('required Python dependencies are installed')",
        ],
        timeout_seconds=30,
    )
    if result.returncode != 0:
        raise DemoFailure("Python dependencies are missing. Run: python -m pip install -r requirements.txt")


def run_verification_command(state: DemoState) -> None:
    result = run_command(
        state,
        "Run JSONL logging verification",
        [
            sys.executable,
            "scripts/verify_logging_demo.py",
            "--log-dir",
            str(state.log_dir),
            "--clean",
        ],
        timeout_seconds=120,
    )
    if result.stdout:
        try:
            state.verification = json.loads(result.stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise DemoFailure(f"Verification output was not valid JSON: {exc}") from exc
    if result.returncode != 0:
        message = "Verification failed."
        if state.verification and state.verification.get("error"):
            message = str(state.verification["error"])
        raise DemoFailure(message)
    if not state.verification or state.verification.get("status") != "PASS":
        raise DemoFailure("Verification did not report PASS.")
    state.status = "PASS"


def truncate(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated ..."


def render_report(state: DemoState) -> str:
    lines: list[str] = [
        "# JSONL Logging Demo Run",
        "",
        f"- Started at: `{state.started_at.isoformat()}`",
        f"- Status: `{state.status}`",
        f"- Log dir: `{state.log_dir}`",
        f"- Python: `{state.python_version}`",
        "",
    ]
    if state.error:
        lines.extend(["## Error", "", f"`{state.error}`", ""])

    lines.extend(["## Commands And Outputs", ""])
    for index, command in enumerate(state.commands, start=1):
        lines.extend(
            [
                f"### {index}. {command.title}",
                "",
                "Command:",
                "",
                "```text",
                command.command,
                "```",
                "",
                f"Exit code: `{command.returncode}`",
                "",
                "Stdout:",
                "",
                "```text",
                truncate(command.stdout) or "<empty>",
                "```",
                "",
                "Stderr:",
                "",
                "```text",
                truncate(command.stderr) or "<empty>",
                "```",
                "",
            ]
        )

    lines.extend(["## Verification Summary", ""])
    if state.verification is None:
        lines.append("No verification summary was produced.")
    else:
        lines.extend(
            [
                "```json",
                json.dumps(state.verification, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    lines.append("")
    lines.append(f"Final result: `{state.status}`")
    lines.append("")
    return "\n".join(lines)


def write_report(state: DemoState) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(state), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a JSONL logging demo.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    args = parser.parse_args()

    state = DemoState(log_dir=Path(args.log_dir))
    try:
        check_python_dependencies(state)
        run_verification_command(state)
    except DemoFailure as exc:
        state.status = "FAIL"
        state.error = str(exc)
    except Exception as exc:
        state.status = "FAIL"
        state.error = f"{type(exc).__name__}: {exc}"
    finally:
        write_report(state)

    print(f"Logging demo report: {REPORT_PATH}")
    print(f"Result: {state.status}")
    if state.error:
        print(f"Error: {state.error}")
    return 0 if state.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
