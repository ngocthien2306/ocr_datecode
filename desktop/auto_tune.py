"""
Random-search auto-tuner for char-comparison parameters.

Uses the existing NG augmenter as a labelled validation set:
  - clean target ↔ template  → must PASS  (true positive)
  - NG samples  ↔ template  → must FAIL  (true negative; caught defect)

Score = balanced (clean_pass_rate + ng_catch_rate) / 2, with a small bonus for
high average confidence on clean to break ties between configs that all reach
100% accuracy.
"""
import random
import time
import numpy as np

from char_segmenter import DEFAULT_PARAMS, compare_arrays, compare_char_pairs
from ng_augmenter import generate_samples


# Search space — sensible discrete values per param. Keep small so random
# search converges fast. None means "don't search this param, use base value".
SEARCH_SPACE = {
    'min_proc_h':           [60, 80, 100, 140, 200],
    'clahe_clip':           [0.5, 1.0, 1.5, 2.0, 3.0],
    'clahe_grid':           [4, 8, 16],
    'blur_kernel':          [1, 3, 5],
    'close_kernel_factor':  [0.005, 0.015, 0.025, 0.05],
    'min_char_h_factor':    [0.2, 0.3, 0.4, 0.5],
    'tm_blur_sigma':        [0.6, 1.2, 1.8, 2.4],
    'iou_dilate':           [3, 5, 7, 9],
    'pixel_dev_tol':        [0.8, 1.2, 1.6, 2.0],
    'align_max_shift':      [4, 8, 12, 16],
    'align_scale_tol':      [0.0, 0.10, 0.20, 0.30],
    'align_scale_steps':    [3, 5, 7],
    'keep_largest_cc':      [0, 1],
    'pass_threshold':       [0.65, 0.70, 0.75, 0.80, 0.85],
}


def _score(tmpl_img, tgt_img, ng_samples, params):
    """
    Single-region scoring used by the in-dialog auto-tune.

    Strict per-region binary: clean_pass = 1.0 only if `overall_pass=True`
    (char count matches AND every char PASS); else 0.0. NG sample is
    "caught" when its `overall_pass=False`.

    Returns None if neither side could be evaluated.
    """
    try:
        clean_strip, clean_results, clean_overall = compare_arrays(tmpl_img, tgt_img, params=params)
    except Exception:
        return None
    if clean_strip is None or not clean_results:
        return None

    clean_pass_rate = 1.0 if clean_overall else 0.0
    clean_avg_conf = float(np.mean([r[1]['confidence'] for r in clean_results]))

    ng_caught = 0
    ng_valid = 0
    ng_avg_conf = []
    for sample in ng_samples:
        try:
            _, ng_results, ng_overall = compare_arrays(tmpl_img, sample['image'], params=params)
        except Exception:
            continue
        ng_valid += 1
        if not ng_overall:
            ng_caught += 1
        if ng_results:
            ng_avg_conf.append(float(np.mean([r[1]['confidence'] for r in ng_results])))

    if ng_valid == 0:
        return None
    ng_catch_rate = ng_caught / ng_valid

    margin = clean_avg_conf - (float(np.mean(ng_avg_conf)) if ng_avg_conf else 0.0)
    score = 0.5 * clean_pass_rate + 0.5 * ng_catch_rate + 0.05 * max(0.0, margin)

    return {
        'score': float(score),
        'clean_pass_rate': float(clean_pass_rate),
        'ng_catch_rate': float(ng_catch_rate),
        'clean_avg_conf': float(clean_avg_conf),
        'margin': float(margin),
        'n_chars': len(clean_results),
    }


def auto_tune(tmpl_img, tgt_img, base_params=None,
              n_trials=40, n_ng_samples=12, seed=42,
              progress_cb=None, cancel_cb=None):
    """
    Random search over SEARCH_SPACE. Returns (best_params, best_metrics, all_trials).

    `progress_cb(i, n, best_score)` is called after each trial; `cancel_cb()`
    is polled — return True to stop early. Both optional.
    """
    rng = random.Random(seed)
    base = dict(DEFAULT_PARAMS)
    if base_params:
        base.update(base_params)

    # Pre-generate NG samples once so every trial sees the same defects
    ng_samples = generate_samples(tgt_img, params={
        'n_samples': n_ng_samples, 'seed': seed,
    })
    if not ng_samples:
        return base, None, []

    # Always include the user's current config as trial 0 so auto-tune
    # never returns something worse than what they had.
    trials = [dict(base)]
    for _ in range(max(0, n_trials - 1)):
        cand = dict(base)
        for key, choices in SEARCH_SPACE.items():
            cand[key] = rng.choice(choices)
        trials.append(cand)

    best_metrics = None
    best_params = base
    history = []

    for i, params in enumerate(trials):
        if cancel_cb and cancel_cb():
            break
        t0 = time.time()
        metrics = _score(tmpl_img, tgt_img, ng_samples, params)
        elapsed = time.time() - t0
        if metrics is not None:
            history.append({'params': dict(params), 'metrics': metrics, 'elapsed': elapsed})
            if best_metrics is None or metrics['score'] > best_metrics['score']:
                best_metrics = metrics
                best_params = dict(params)
        if progress_cb:
            progress_cb(i + 1, len(trials),
                        best_metrics['score'] if best_metrics else 0.0)

    return best_params, best_metrics, history


def auto_tune_multi_image(prepared_pairs, base_params=None,
                          n_trials=20, n_ng_per_pair=4, seed=42,
                          locked_keys=None,
                          progress_cb=None, cancel_cb=None):
    """
    Multi-image / multi-region random search.

    `prepared_pairs` is a flat list of (template_crop_bgr, target_crop_bgr)
    tuples — already extracted from each (image, region) combo by the caller.

    `locked_keys` (optional set/iterable): parameter names that are NOT
    randomised — they keep their value from `base_params` (or DEFAULT_PARAMS)
    throughout the entire search. Use this to fix values the user explicitly
    set in the UI (e.g. PASS threshold).

    For each trial: each pair contributes 1 clean compare + `n_ng_per_pair`
    augmented NG compares. Score is balanced across all pairs.

    Returns (best_params, best_metrics, history). best_metrics is None if no
    trial succeeded.
    """
    import random as _random
    rng = _random.Random(seed)

    base = dict(DEFAULT_PARAMS)
    if base_params:
        base.update(base_params)
    locked = set(locked_keys or [])

    if not prepared_pairs:
        return base, None, []

    # Normalise pairs: accept either (tmpl, tgt) or (tmpl, tgt, kind).
    # `kind` ∈ {'char', 'region'} routes between compare_char_pairs (no
    # segmentation, used when annotation provides explicit char bboxes) and
    # compare_arrays (segments both sides, the original whole-region path).
    norm_pairs = []
    for pair in prepared_pairs:
        if len(pair) == 3:
            tmpl, tgt, kind = pair
        else:
            tmpl, tgt = pair
            kind = 'region'
        norm_pairs.append((tmpl, tgt, kind))

    # Pre-generate NG samples per pair so every trial sees the same defects
    # (deterministic scoring → trials comparable).
    ng_per_pair = []
    for i, (_, tgt, _kind) in enumerate(norm_pairs):
        samples = generate_samples(tgt, params={
            'n_samples': max(0, int(n_ng_per_pair)),
            'seed': seed + i + 1,
        })
        ng_per_pair.append([s['image'] for s in samples])

    prepared_pairs = norm_pairs  # downstream uses 3-tuple form

    # Trial 0 = baseline so we never return something worse than the user had
    trials = [dict(base)]
    for _ in range(max(0, n_trials - 1)):
        cand = dict(base)
        for key, choices in SEARCH_SPACE.items():
            if key in locked:
                # User pinned this param → don't randomise it
                continue
            cand[key] = rng.choice(choices)
        trials.append(cand)

    best_params = base
    best_metrics = None
    history = []

    for t_idx, params in enumerate(trials):
        if cancel_cb and cancel_cb():
            break
        t0 = time.time()
        metrics = _score_multi(prepared_pairs, ng_per_pair, params)
        elapsed = time.time() - t0
        if metrics is not None:
            history.append({'params': dict(params), 'metrics': metrics, 'elapsed': elapsed})
            if best_metrics is None or metrics['score'] > best_metrics['score']:
                best_metrics = metrics
                best_params = dict(params)
        if progress_cb:
            progress_cb(t_idx + 1, len(trials),
                        best_metrics['score'] if best_metrics else 0.0)

    return best_params, best_metrics, history


def _score_multi(prepared_pairs, ng_per_pair, params):
    """Compute aggregated score for one param config across all pairs.

    Strict per-region binary scoring: a region only counts as PASS when
    `compare_arrays` returns overall_pass=True (i.e. char count matches AND
    every char is PASS). Even 1 FAIL char → whole region FAIL.

    Score = 0.5*clean_pass_rate + 0.5*ng_catch_rate + 0.05*max(0, margin)
    Returns None if no pair could be evaluated.
    """
    n_clean_total = 0
    n_clean_pass = 0
    n_clean_count_mismatch = 0  # bookkeeping: how many regions had char count mismatch
    clean_confs = []

    n_ng_total = 0
    n_ng_caught = 0
    ng_confs = []

    for pair, ng_list in zip(prepared_pairs, ng_per_pair):
        # Pair may be 3-tuple (tmpl, tgt, kind) or 2-tuple (tmpl, tgt) — normalise.
        if len(pair) == 3:
            tmpl, tgt, kind = pair
        else:
            tmpl, tgt = pair
            kind = 'region'

        def _compare(t, g):
            """Dispatch by pair kind. Returns (results, overall_pass)."""
            try:
                if kind == 'char':
                    out = compare_char_pairs([(t, g)], params=params)
                    return out['results'], out['overall_pass']
                else:
                    _, r, op = compare_arrays(t, g, params=params)
                    return r, op
            except Exception:
                return None, False

        # ---- Clean: 1 unit = 1 binary outcome ----
        results, overall_pass = _compare(tmpl, tgt)
        n_clean_total += 1
        if overall_pass:
            n_clean_pass += 1
        if results:
            clean_confs.extend(r[1]['confidence'] for r in results)
            if (not overall_pass) and all(r[2] == 'PASS' for r in results):
                n_clean_count_mismatch += 1

        # ---- NG: each augmented sample = 1 binary outcome (caught = FAIL) ----
        for ng_img in ng_list:
            ng_results, ng_overall = _compare(tmpl, ng_img)
            n_ng_total += 1
            if not ng_overall:
                n_ng_caught += 1
            if ng_results:
                ng_confs.append(float(np.mean([r[1]['confidence'] for r in ng_results])))

    if n_clean_total == 0 or n_ng_total == 0:
        return None

    clean_pass_rate = n_clean_pass / n_clean_total
    ng_catch_rate = n_ng_caught / n_ng_total
    mean_clean = float(np.mean(clean_confs)) if clean_confs else 0.0
    mean_ng = float(np.mean(ng_confs)) if ng_confs else 0.0
    margin = mean_clean - mean_ng

    score = 0.5 * clean_pass_rate + 0.5 * ng_catch_rate + 0.05 * max(0.0, margin)
    return {
        'score': float(score),
        'clean_pass_rate': float(clean_pass_rate),
        'ng_catch_rate': float(ng_catch_rate),
        'clean_avg_conf': mean_clean,
        'ng_avg_conf': mean_ng,
        'margin': float(margin),
        'n_pairs': n_clean_total,
        'n_ng_samples': n_ng_total,
        'n_clean_pass': n_clean_pass,
        'n_clean_count_mismatch': n_clean_count_mismatch,
    }
