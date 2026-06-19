"""
Export nhãn ĐÃ CONFIRM từ check_data_fastapi.py → format training giống data_ocr:

    <out>/train/<file>.jpg   +   <out>/rec_gt_train.txt   (dòng: "train/<file>\\t<label>")
    <out>/test/<file>.jpg    +   <out>/rec_gt_test.txt    (dòng: "test/<file>\\t<label>")

CHỈ lấy ảnh đã Confirm (có trong gradio_data/part_labels/*.txt). Ảnh Reject /
chưa review KHÔNG được export. Chạy được BẤT CỨ LÚC NÀO (kể cả khi đang label dở).

Usage:
    python backend/scripts/export_training_from_check.py --folder ./ocr_check_data
    python backend/scripts/export_training_from_check.py --folder ./ocr_check_data \
        --out ./data_ocr_export --ratio 0.85
"""

import argparse
import shutil
import zlib
from pathlib import Path


def read_labels(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                out[k] = v
    return out


def auto_split(name: str, ratio: float) -> str:
    """Chia train/test ổn định theo hash tên file (giữa các lần export như nhau)."""
    return "train" if (zlib.crc32(name.encode()) % 1000 / 1000.0) < ratio else "test"


def main():
    ap = argparse.ArgumentParser(description="Export confirmed labels → training format (data_ocr)")
    ap.add_argument("--folder", required=True, help="Folder đã chạy check_data_fastapi.py (chứa ảnh + gradio_data/)")
    ap.add_argument("--out", default="./data_ocr_export", help="Thư mục output (default: ./data_ocr_export)")
    ap.add_argument("--ratio", type=float, default=0.85, help="Tỉ lệ train (còn lại test). (default: 0.85)")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    part_labels_dir = folder / "gradio_data" / "part_labels"
    base_labels = read_labels(folder / "labels.txt")  # nhãn gốc (nếu có)

    # Gộp tất cả nhãn CONFIRMED từ các phần (giống merge_labels của tool)
    confirmed = dict(base_labels)
    n_parts = 0
    if part_labels_dir.exists():
        for f in sorted(part_labels_dir.glob("*_labels.txt")):
            confirmed.update(read_labels(f))
            n_parts += 1

    if not confirmed:
        print(f"⚠ Chưa có nhãn confirmed nào trong {part_labels_dir}")
        print("  (Confirm vài ảnh trong web tool trước, rồi chạy lại.)")
        return

    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    lines = {"train": [], "test": []}
    counts = {"train": 0, "test": 0, "missing": 0}
    for filename, label in sorted(confirmed.items()):
        src = folder / filename
        if not src.exists():
            counts["missing"] += 1
            continue
        split = auto_split(filename, args.ratio)
        shutil.copy(str(src), str(out / split / filename))
        lines[split].append(f"{split}/{filename}\t{label}")
        counts[split] += 1

    (out / "rec_gt_train.txt").write_text(
        "\n".join(lines["train"]) + ("\n" if lines["train"] else ""), encoding="utf-8")
    (out / "rec_gt_test.txt").write_text(
        "\n".join(lines["test"]) + ("\n" if lines["test"] else ""), encoding="utf-8")

    print(f"✓ Đọc {len(confirmed)} nhãn confirmed từ {n_parts} phần")
    print(f"✓ Export: {counts['train']} train + {counts['test']} test"
          + (f" (thiếu {counts['missing']} ảnh)" if counts["missing"] else ""))
    print(f"  → {out.resolve()}")
    print(f"    ├─ train/ ({counts['train']} ảnh) + rec_gt_train.txt")
    print(f"    └─ test/  ({counts['test']} ảnh) + rec_gt_test.txt")


if __name__ == "__main__":
    main()
