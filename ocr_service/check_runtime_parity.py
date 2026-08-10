"""
Check that app/services/smtr_runtime/ still matches ai_services' originals.

ocr_service vendors ai_services' SMTR decode path rather than importing it (see
smtr_runtime/__init__.py for why). The accuracy this service reports is only
meaningful if it decodes exactly the way production does, so the copies drifting
apart is a real failure mode — and a silent one, since both sides keep working.

Run this after changing either copy. Exits 1 on any difference.

    python check_runtime_parity.py
    python check_runtime_parity.py --diff     # show what changed
"""
import argparse
import difflib
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).parent
AI_SERVICES = (SERVICE_ROOT / ".." / "ai_services").resolve()
VENDORED = SERVICE_ROOT / "app" / "services" / "smtr_runtime"

# vendored file -> original, plus the one edit the copy is allowed to carry
FILES = {
    "smtr_utils.py": ("camera_management/ocr/smtr_utils.py", []),
    "smtr_trt.py": ("camera_management/ocr/backends/smtr_trt.py",
                    [("from .smtr_utils import", "from ..smtr_utils import")]),
    "smtr_onnx.py": ("camera_management/ocr/backends/smtr_onnx.py",
                     [("from .smtr_utils import", "from ..smtr_utils import")]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", action="store_true", help="print a unified diff per mismatch")
    args = ap.parse_args()

    if not AI_SERVICES.is_dir():
        print(f"ai_services not found at {AI_SERVICES} — cannot compare", file=sys.stderr)
        return 2

    failures = 0
    for name, (rel, rewrites) in FILES.items():
        mine = VENDORED / name
        theirs = AI_SERVICES / rel
        if not mine.is_file() or not theirs.is_file():
            print(f"MISSING  {name}: {'copy' if not mine.is_file() else 'original'} absent")
            failures += 1
            continue

        # Undo the allowed rewrites so the comparison is otherwise exact.
        text = mine.read_text(encoding="utf-8")
        for frm, to in rewrites:
            text = text.replace(frm, to)
        original = theirs.read_text(encoding="utf-8")

        if text == original:
            print(f"OK       {name}")
            continue

        failures += 1
        print(f"DRIFT    {name}  (vs {rel})")
        if args.diff:
            for line in difflib.unified_diff(
                original.splitlines(), text.splitlines(),
                fromfile=f"ai_services/{rel}", tofile=f"ocr_service/.../{name}", lineterm="",
            ):
                print(f"    {line}")

    if failures:
        print(f"\n{failures} file(s) out of sync. Reconcile before trusting any accuracy "
              f"number this service reports — it decodes with the copy, production "
              f"decodes with the original.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
