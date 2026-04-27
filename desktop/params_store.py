"""
Load/save trained char-comparison parameters next to annotations.json.

File format (v1) — one shared param set for all regions:
{
  "version": 1,
  "trained_at": "ISO-8601",
  "trained_on": ["filename1.png", ...],
  "n_images": 12,
  "params": {...},
  "metrics": {"score": ..., "clean_pass_rate": ..., "ng_catch_rate": ..., "margin": ...}
}
"""
import json
import os
from datetime import datetime


PARAMS_FILENAME = "compare_params.json"


def params_path_for(annotations_json_path):
    """Return the canonical compare_params.json path for a given annotations.json."""
    if not annotations_json_path:
        return None
    return os.path.join(os.path.dirname(annotations_json_path), PARAMS_FILENAME)


def load_trained_params(annotations_json_path):
    """
    Returns the saved record dict (with `params`, `metrics`, ...) or None
    if the file is missing or unreadable.
    """
    p = params_path_for(annotations_json_path)
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'params' not in data:
            return None
        return data
    except Exception as e:
        print(f"[params_store] failed to load {p}: {e}")
        return None


def save_trained_params(annotations_json_path, params, metrics=None,
                        trained_on=None):
    """Write the params + metadata. Overwrites any existing file."""
    p = params_path_for(annotations_json_path)
    if not p:
        raise ValueError("annotations_json_path is required")

    record = {
        'version': 1,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'trained_on': list(trained_on or []),
        'n_images': len(trained_on or []),
        'params': dict(params),
        'metrics': dict(metrics or {}),
    }
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return p
