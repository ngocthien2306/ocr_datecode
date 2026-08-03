"""
Per-character quality comparison: given a template char crop and the captured
one, decide how well they match and what kind of defect (if any) is present.

Four alternative CV algorithms plus the original one are all selectable at
runtime via the recipe's `cv_method`. They are genuinely different approaches,
not revisions superseding one another — none is dead code:

    v3  dilate-tolerant ink over/under scoring on translation-aligned binaries
    v4  affine alignment + stroke-width ratio (catches scale/shear drift)
    v5  v4 split into a 3x3 tile grid, so a local defect isn't averaged away
    v7  gradient-orientation agreement (shape-based, ignores ink thickness)
    legacy  the original IoU/NCC pipeline, also home to char segmentation

Known coupling, kept as-is: v4 and v7 import shared primitives from v3, and v5
imports them from v4, so v3/v4 are load-bearing for the others rather than
being self-contained experiments. Extracting those primitives into a neutral
module is a worthwhile follow-up; it is deliberately not bundled with the move
into this package so the two changes stay separately verifiable.
"""

from typing import Callable, Dict

from .legacy import (  # noqa: F401
    char_quality,
    img_to_b64,
    save_char_comparison,
    segment_characters_from_image,
)


def _v3(tmpl_gray, tgt_gray):
    from .v3 import compute_char_quality_v3
    return compute_char_quality_v3(tmpl_gray, tgt_gray)


def _v4(tmpl_gray, tgt_gray):
    from .v4 import compute_char_quality_v4
    return compute_char_quality_v4(tmpl_gray, tgt_gray)


def _v5(tmpl_gray, tgt_gray):
    from .v5 import compute_char_quality_v5
    return compute_char_quality_v5(tmpl_gray, tgt_gray)


def _v7(tmpl_gray, tgt_gray):
    from .v7 import compute_char_quality_v7
    return compute_char_quality_v7(tmpl_gray, tgt_gray)


# cv_method (as stored on the recipe) -> metric function.
# Replaces the if/elif chain that used to live in embedding_classifier; adding
# a method is now one entry here instead of another near-identical branch.
# Imports stay lazy per entry: each algorithm pulls in its own helpers, and
# only the selected one should be paid for.
METRIC_METHODS: Dict[str, Callable] = {
    'v3': _v3,
    'v4': _v4,
    'v5': _v5,
    'v7': _v7,
    'shape_v7': _v7,   # alias kept — recipes in the field use both spellings
}


def get_metric_method(cv_method: str):
    """Return the metric function for `cv_method`, or None for 'legacy' and
    any unrecognised value — callers fall back to the legacy CV pipeline,
    which is what the old `else` branch did."""
    return METRIC_METHODS.get(cv_method)


__all__ = [
    'char_quality',
    'img_to_b64',
    'save_char_comparison',
    'segment_characters_from_image',
    'METRIC_METHODS',
    'get_metric_method',
]
