from __future__ import annotations

import sys
from pathlib import Path

from censor_guard.schemas import SignalResult

# Корень проекта, где лежит модуль robust_censor.py и веса robust_probe.npz.
# Кладём его в sys.path, чтобы `from robust_censor import RobustCensor` работал
# независимо от того, откуда запущен пайплайн (CLI, UI, тесты).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _install_autoattack_shim() -> None:
    """robustbench при импорте тянет `autoattack` (FAB/AutoAttack), но в PyPI он
    приезжает как `pyautoattack` (тот же код, другое имя пакета). load_model сам
    атаку не запускает, поэтому достаточно зарегистрировать алиас, чтобы пройти
    импорт `robustbench.__init__` → `robustbench.eval`. Если настоящий `autoattack`
    уже стоит — ничего не делаем."""
    import importlib.util

    if importlib.util.find_spec("autoattack") is not None:
        return
    try:
        import pyautoattack
        import pyautoattack.autoattack as _aa
        import pyautoattack.state as _state
    except Exception:
        return  # нет и pyautoattack — пусть импорт robustbench честно упадёт ниже
    if not hasattr(pyautoattack, "AutoAttack"):
        pyautoattack.AutoAttack = _aa.AutoAttack
    sys.modules.setdefault("autoattack", pyautoattack)
    sys.modules.setdefault("autoattack.state", _state)


class RobustGuardAdapter:
    """Робастный цензор (Linf-robust ConvNeXt + linear probe) как ДЕТЕКТОР
    adversarial-примеров.

    Это намеренно «слабая» по абсолютному качеству, но устойчивая к атакам модель.
    Adversarial-возмущения подбираются под основную (не робастную) модель, поэтому
    робастный зонд они почти не сдвигают. Идея интеграции: сравнить P(unsafe)
    робастной модели с unsafe-оценкой основного пайплайна — большое расхождение
    выдаёт подделку (пайплайн обманут, робастная модель держит истину).

    Сам адаптер решений не выносит и категории таксономии не выставляет: он лишь
    отдаёт скаляр P(unsafe) и аудиторский SignalResult. Логику расхождения и блок
    держит пайплайн (GuardrailPipeline.check_adversarial).

    Тяжёлая модель грузится лениво и переиспользуется. Если robustbench/torch/веса
    недоступны — мягко деградируем (skipped/error), не роняя сервис: робастный
    индикатор просто молчит, остальной пайплайн работает как прежде.
    """

    name = "robust_guard"

    def __init__(self, enabled: bool, probe_path: str, model_dir: str) -> None:
        self.enabled = enabled
        self.probe_path = probe_path
        self.model_dir = model_dir
        self._censor = None
        self._load_error: str | None = None
        # True, если причина недоступности — отсутствие зависимости/весов (нормальная
        # мягкая деградация, status=skipped), а не реальный сбой (status=error).
        self._soft_unavailable = False

    def _load(self):
        if self._censor is not None:
            return self._censor
        if self._load_error is not None:
            return None
        if not Path(self.probe_path).exists():
            self._load_error = f"probe weights not found at {self.probe_path}"
            self._soft_unavailable = True
            return None
        _install_autoattack_shim()
        try:
            from robust_censor import RobustCensor
        except ImportError as exc:
            # robustbench/torch не установлены — тихо деградируем.
            self._load_error = f"robust_censor import failed: {exc}"
            self._soft_unavailable = True
            return None
        try:
            self._censor = RobustCensor(probe_path=self.probe_path, model_dir=self.model_dir)
        except ImportError as exc:
            # robustbench/torch подгружаются внутри RobustCensor.__init__ — отсутствие
            # зависимости всплывает здесь. Это мягкая деградация, а не сбой.
            self._load_error = f"robust dependencies missing: {exc}"
            self._soft_unavailable = True
            return None
        except Exception as exc:  # pragma: no cover - backend-specific failures
            self._load_error = str(exc)
            return None
        return self._censor

    def unsafe_score(self, image) -> float | None:
        """P(unsafe) робастной модели в [0,1] или None, если модель недоступна."""
        if not self.enabled:
            return None
        censor = self._load()
        if censor is None:
            return None
        try:
            return float(censor.unsafe_score(image))
        except Exception as exc:  # pragma: no cover - backend-specific failures
            self._load_error = self._load_error or f"robust inference failed: {exc}"
            return None

    def moderate(self, image) -> SignalResult:
        """Аудиторский сигнал: P(unsafe) робастной модели + её бинарный вердикт.

        Категории таксономии НЕ выставляет (это индикатор adversarial, а не сенсор
        нарушения), поэтому в fusion как «улика нарушения» не попадает."""
        if not self.enabled:
            return SignalResult(name=self.name, status="skipped", reason="Robust guard disabled by configuration.")
        censor = self._load()
        if censor is None:
            return SignalResult(
                name=self.name,
                status="skipped" if self._soft_unavailable else "error",
                reason=f"Robust guard unavailable: {self._load_error}",
            )
        score = self.unsafe_score(image)
        if score is None:
            return SignalResult(name=self.name, status="error", reason=f"Robust guard inference failed: {self._load_error}")
        return SignalResult(
            name=self.name,
            status="ok",
            reason=f"Robust unsafe probability {score:.3f} (verdict={'unsafe' if score >= censor.threshold else 'safe'}).",
            raw={"robust_unsafe": score, "robust_threshold": float(censor.threshold)},
        )
