import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Dict

# Synthetic package name. Deliberately not "camera_management" and not a real
# importable path, so nothing can accidentally resolve to the CUDA-laden package.
_PKG = "_ocr_cv_detectors"

_MODULES = ("image_proc_detector", "color_verifier")

_lock = threading.Lock()
_loaded: Dict[str, ModuleType] = {}


def _verification_dir() -> Path:
    """Locate ai_services/camera_management/verification.

    Prefers a path derived from this file (repo-relative, survives being checked
    out anywhere) and falls back to the $HOME layout the deploy scripts assume.
    """
    override = os.environ.get("OCR_AI_SERVICES_DIR")
    candidates = []
    if override:
        candidates.append(Path(override))
    # <repo>/backend/app/services/cv_detectors.py → parents[3] == <repo>
    candidates.append(Path(__file__).resolve().parents[3] / "ai_services")
    candidates.append(Path(os.environ.get("HOME", "~")).expanduser() / "Source" / "ocr_datecode" / "ai_services")

    for base in candidates:
        d = base / "camera_management" / "verification"
        if d.is_dir():
            return d
    raise FileNotFoundError(
        "Could not locate ai_services/camera_management/verification; "
        f"tried: {[str(c) for c in candidates]}. Set $OCR_AI_SERVICES_DIR."
    )


def _ensure_pkg(verification_dir: Path) -> None:
    if _PKG in sys.modules:
        return
    pkg = ModuleType(_PKG)
    # __path__ is what makes `from .image_proc_detector import X` resolve to a
    # sibling FILE in this directory rather than to the real package.
    pkg.__path__ = [str(verification_dir)]  # type: ignore[attr-defined]
    pkg.__package__ = _PKG
    sys.modules[_PKG] = pkg


def _load(name: str) -> ModuleType:
    cached = _loaded.get(name)
    if cached is not None:
        return cached

    with _lock:
        if name in _loaded:  # another thread won the race
            return _loaded[name]

        vdir = _verification_dir()
        _ensure_pkg(vdir)

        # color_verifier does `from .image_proc_detector import ...` inside a
        # function; load the dependency first so it is already in sys.modules.
        if name == "color_verifier" and "image_proc_detector" not in _loaded:
            _load_locked("image_proc_detector", vdir)

        return _load_locked(name, vdir)


def _load_locked(name: str, vdir: Path) -> ModuleType:
    """Actual file load. Caller must hold _lock."""
    if name in _loaded:
        return _loaded[name]

    full = f"{_PKG}.{name}"
    path = vdir / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"detector module not found: {path}")

    spec = importlib.util.spec_from_file_location(full, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # spec.parent is _PKG, so relative imports inside the file resolve against
    # the synthetic package. Set explicitly rather than relying on it.
    module.__package__ = _PKG
    sys.modules[full] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full, None)  # don't leave a half-initialised module behind
        raise

    _loaded[name] = module
    return module


def load_image_proc_detector() -> ModuleType:
    """Pure cv2/numpy/scipy edge + wall + cap-axis detectors."""
    return _load("image_proc_detector")


def load_color_verifier() -> ModuleType:
    """Pure cv2/numpy colour verification (we only use `_detect_bottle`)."""
    return _load("color_verifier")


def cuda_modules_present() -> list:
    """Names of CUDA-binding modules currently imported into THIS process.

    Must stay empty in the backend. Mirrors the runbook's periodic check
    (`grep -c pycuda /proc/<uvicorn pid>/maps` == 0) but from inside Python.
    """
    return [m for m in ("pycuda", "pycuda.driver", "pycuda.autoinit", "tensorrt")
            if m in sys.modules]
