---
name: Check_Color feature — HSV color check pipeline
description: End-to-end design of the bottle-color verification feature. Covers FE Setup Color modal, BE pydantic plumbing, AI service color_verifier + OR aggregation across color cameras. Independent of SuperPoint matching.
type: project
---
# Check_Color (HSV color verification)

## Purpose
Detect a wrong-color product entering the line by checking the HSV pixel histogram of the bottle against a per-template reference. Independent of OCR/SuperPoint — image-proc finds the bottle and `cv2.inRange` counts matching pixels.

## Pipeline modes per (function_type, template annotations)

| `camera.function_type` | Template has `product`? | Template has `text/datecode`? | Pipeline |
|---|---|---|---|
| `Check_Color` | ✅ | * | **color_check**: image-proc detect bottle + HSV pixel count. NO rotate, NO SuperPoint dependency. |
| `Check_Color` | ❌ | ✅ | **rotate_ocr** (legacy "Rotate Bottle"): OBB rotate cap → SuperPoint → OCR. |
| `Check_Type_Product` | ✅ + has `label` | * | existing product_verifier (alignment + wrinkle) + OCR. |
| `Check_Type_Product` | ❌ | ✅ | OCR-only (SuperPoint + OCR, no product verify). |

## Per-camera pass/fail rules

- **Color cameras** (Check_Color + product): result depends ONLY on `color_verifier.match`. SuperPoint match success is irrelevant — image-proc handles localization.
- **Other cameras**: text AND char AND template AND product AND SuperPoint match — original AND logic.

## Cross-camera aggregation (multi_camera.py)

- **Color group** (any cam where function_type=Check_Color + product): **OR** — at least 1 PASS → group OK.
- **Other group**: **AND** — every cam must PASS.
- **Overall = color_group_ok AND non_color_group_ok**.
- No color cameras in recipe → color group is vacuously OK; fallback to pure AND.

### OR-pass promotion (UX consistency)

When `color_group_ok` is True (i.e. at least one color cam passed) we **retroactively promote any individually-failing color cameras to `result='PASS'`** so the per-camera UI badge / stats logs match the recipe-level verdict. The `product_verification.color_check` sub-dict still carries the original `matching_pixels` so the historical detail view can show "this camera underperformed individually" if anyone digs in.

We also strip `verification_status='fail'` from the product polygon bboxes on promoted cameras so the on-frame overlay isn't drawn red. Look for the log line `Color group OR-PASS: promoted N individually-failing color camera(s) to PASS`.

Single-camera pipeline doesn't have this promotion because OR-with-one-element is just that camera's own result.

## Annotation coordinate convention (CRITICAL gotcha)

- TemplateEditor normalizes annotation coords to **[0, 1]** relative to image bounds (`TemplateEditorRefactored.tsx:912-941`).
- `annotation.x/y/width/height` (rectangle) and `annotation.points` (polygon) are BOTH stored normalized.
- Consumers MUST denormalize by multiplying with `image_width`/`image_height` (template) or `frame.shape[:2]` (frame) before treating as pixel coords.
- The existing product_verifier path receives points in FRAME pixel coords because SuperPoint matcher denormalizes when producing `transformed_bboxes`. Anything that BYPASSES SuperPoint (like color_verifier) must denormalize itself.
- **Bug history**: yellow overlay didn't show because raw normalized coords were treated as pixel coords → polygon mask collapsed to ~1 pixel.

## color_config schema (per template)

```python
{
  'h_min': int, 'h_max': int,   # 0–180 (OpenCV HSV)
  's_min': int, 's_max': int,   # 0–255
  'v_min': int, 'v_max': int,   # 0–255
  'pixel_threshold': int,        # PASS if matching pixels >= this
  'roi_circle': {                # UI-only — persisted so user can re-edit
    'center': [x, y],            # in TEMPLATE pixel coords (already denormalized at save time)
    'radius': float,
  }
}
```

Stored on `TemplateImage` Pydantic model in **both** `backend/app/models/recipe.py` and `backend/app/schemas/recipe.py` (BE models are duplicated — see [[recipe-system]]). Passes through to AI service via the normal camera_templates plumbing — no extra 19-step plumb required because it's a nested per-template field, not a top-level recipe field.

## Bottle detection algorithm (color_verifier)

Sharpness-based, ported from `tests/test_bottle_detect_compare.py:detect_bottle_cv`:

1. `cv2.Laplacian(gray, CV_32F, ksize=3)` → `|Laplacian|²`
2. `cv2.boxFilter(..., ksize=(31, 31))` → smoothed sharpness map
3. Threshold at 30% of max → binary mask (focal plane = bottle, blurred bg = 0)
4. Morphology `MORPH_CLOSE (25×5)` + `MORPH_OPEN (9×9)`
5. Connected components, score each by `area × aspect_match × (1 - 0.5 × center_distance)`. Aspect hint comes from the product polygon (template).
6. Best component → return YOLO-OBB-compatible dict `{box, score, class='product', corners, source='image_proc_color'}`.

Works on ~280ms / 2MP frame on CPU. No GPU.

## HSV match

`cv2.cvtColor(roi, BGR2HSV)` + `cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))` → `count_nonzero`. ROI = the bottle bbox detected in step 6 above.

## FE — ColorSetupModal

- Opens from filmstrip "Setup Color" button (only visible when `function_type=Check_Color` + template has `product` annotation).
- Inputs: template image, `productPolygons` (denormalized to template pixel coords by `RecipeFormModal.tsx` before passing in), `initialConfig` (for re-edit).
- Canvas overlays: image → product polygon outline (purple) → match-pixel yellow overlay (ONLY inside polygon mask) → ROI cyan circle.
- **Yellow overlay strict**: pixel must satisfy `polyMask[i] && in HSV range`. If `productPolygons` is empty the mask is empty and NO yellow is drawn (do NOT fall back to "treat full image" — that was the original bug).
- ROI drag: while dragging only the cyan circle moves. On **mouseup** we re-run `autoDetectFromRoi` so HSV range + yellow overlay refresh to the new sample location.
- Histogram: 3 mini line charts (H red, S green, V blue), bg `#f3f4f6`.
- Light-theme palette matches [[feedback-ui-theme]].

## Key files

| File | Role |
|------|------|
| `frontend-ts/src/components/recipe/ColorSetupModal.tsx` | Canvas + HSV sliders + auto-detect |
| `frontend-ts/src/styles/ColorSetupModal.css` | Light-theme styles |
| `frontend-ts/src/components/recipe/RecipeFormModal.tsx` | Setup Color button in filmstrip + `productPolys` denormalization + map `color_config` to submit payload + validation skips text/template requirement for color templates |
| `frontend-ts/src/components/inference/InferenceRealtime.tsx` | Realtime overlay badge: `Color: matching/threshold (pct%)` next to frame Pass/Fail. Green when ≥ threshold, red when below. `ColorCheck` interface added to `ProductVerification`. |
| `frontend-ts/src/components/dashboard/historical/InspectionResultRow.tsx` | Color Check details grid (Status, Matching, Threshold, Bottle Area, Match %, Detected, H/S/V ranges). Renders inside the existing Product Verification section. |
| `frontend-ts/src/components/dashboard/historical/InspectionResultsTab.tsx` | "Color Check" fail-reason filter pill. |
| `frontend-ts/src/services/inferenceResults.ts` | `ColorCheck` interface + `color_check` field on `ProductVerification`. `FailReason` includes `'color'`. |
| `backend/app/models/recipe.py` | `ColorConfig`, `ColorRoiCircle`, `color_config` field on `TemplateImage` |
| `backend/app/schemas/recipe.py` | Same as models — duplicated |
| `backend/app/repositories/inference_result_repository.py` | `fail_reasons=color` filter: `{product_verification.color_check: {$exists: True}, product_verification.match: False}` |
| `ai_services/camera_management/verification/color_verifier.py` | `ColorVerificationService.verify_batch`. Logs per-camera matching count at INFO level. |
| `ai_services/camera_management/verification/product_verifier.py` | Routes Check_Color+product frames to color_verifier (split + merge for mixed batch) |
| `ai_services/camera_management/pipeline/single_camera.py` | OBB rotation gate (disabled for color), removed Check_Color skip in product verify, color frame pass/fail = product_ok only |
| `ai_services/camera_management/pipeline/multi_camera.py` | OR-aggregation for color camera group, color frame pass/fail = product_ok only, OR-pass promotion |
| `ai_services/camera_management/utils.py` | `draw_color_match_overlay()` — recomputes HSV mask from frame + bottle bbox, paints semi-transparent yellow on matching pixels. Called by both `encode_frame_for_display` (realtime base64) and `save_and_encode_frame` (disk-saved viz). |

## UI result schema for color_check

The frame's `product_verification.color_check` carries:
```ts
{
  matching_pixels: number;    // HSV-matched pixels in the bottle bbox
  bottle_pixels: number;      // total pixels in bottle bbox (denominator)
  pixel_threshold: number;    // PASS threshold from color_config
  detected: boolean;          // false → image-proc failed to find bottle
  h_range: [number, number];
  s_range: [number, number];
  v_range: [number, number];
}
```

PASS criterion (UI side): `matching_pixels >= pixel_threshold`. Match % shown to user is `matching_pixels / bottle_pixels`.

## Template validation rules

Validation in `RecipeFormModal.validateTemplates()` (and the filmstrip "!" badge in the same file) branches on `cameraFunctionTypes[cameraId]`:

- **Check_Color cameras**: `"template"` region is NEVER required (image-proc skips SuperPoint entirely). Validation only asks for at least one of `product` (color check sub-mode) OR `text`/`barcode`/`datecode` (rotate-OCR sub-mode). The branch is on `function_type=='Check_Color'` alone — it doesn't wait for product to be drawn before relaxing the template requirement.
- **Other cameras** (Check_Type_Product, default): must have 1 `template` region + at least 1 of `text`/`barcode`/`datecode`. Existing behavior.

Filmstrip "!" badge follows the same rule: `isCheckColorCam ? (hasProduct || hasRequired) : (hasTemplate && hasRequired)`.

When adding more annotation requirements in the future, branch on `function_type=='Check_Color'` so Check_Color cameras aren't blocked by OCR-oriented rules.

## Non-obvious behaviors / gotchas

- **OBB rotation is gated per-camera-mode**, not per-template (ship-simple decision). Rule: rotate iff `function_type=Check_Color AND first template has NO product`. Check_Color cameras with product → rotation OFF. See `single_camera.py:101`.
- **SuperPoint match SKIPPED for color cameras** (multi_camera.run_inference): the match batch is split by `_is_color_cam`, color cams get a synthetic `success=True` result with raw template annotations denormalized to frame pixel coords as `transformed_bboxes`. Saves ~200ms per color camera. Single-camera pipeline still runs match (color templates are rare in single-camera setups).
- **template_verifier SKIPPED for color cameras** (both pipelines): pixel-similarity check between template image and frame crop is meaningless without the SuperPoint alignment. Saves ~80ms per color camera.
- **Frame ≈ template image assumption**: color_verifier denormalizes annotation coords using `frame.shape[:2]`, NOT `template.image_width/image_height`. This assumes the camera that captured the template is the same one running inference (true for fixed-camera setups). If the user uploads a template from a different camera/resolution, polygon hint may be in the wrong place.
- **product_verification dict reuses existing field**: color check result goes into `frame_result['product_verification']` with a `color_check` sub-dict. Visualizers/DB/FE that read `match` field for PASS/FAIL keep working. New consumers can read `product_verification.color_check.matching_pixels` etc.
- **OR aggregation only in multi_camera**: single-camera recipes trivially have 1 camera so OR/AND are the same. If user has only 1 color camera and it fails → overall fails (no group to compensate).
- **Color verifier ALWAYS includes frame_img** in frames_data even if SuperPoint match failed (see `single_camera._batch_verify_products` + `multi_camera._batch_verify_products`). Non-color paths still gate on match success.
- **Legacy `Check_Color` recipes (cap OCR mode)**: existed before the color check feature. They have no product annotation → mode resolves to `rotate_ocr` → rotation ON, OCR runs as before. NO migration needed; backward-compatible.

## Test plan
1. Recipe with 3 cams: cam1+cam2 Check_Color with product polygon + color_config, cam3 Check_Type_Product with text/datecode.
2. Setup Color modal: drag ROI → on release, yellow overlay updates (inside polygon only).
3. Inference: log should show `Color verification batch: N frames` and `Color camera group: M/N PASS (OR-logic)`.
4. Swap one product to a wrong color → 1 color cam fails, 1 passes → overall PASS (OR).
5. Both wrong-color → both color cams fail → overall FAIL.
6. Legacy Check_Color recipe (no product) → still rotates + OCRs as before.

## MatcherFactory fast path — ColorCheckStubMatcher

`MatcherFactory.create_matcher` requires a `type='template'` annotation to build the SuperPoint matcher. Check_Color cameras with a `product` annotation don't need SuperPoint (image-proc handles bottle detection), but the inference pipeline still gates on `serial in self.camera_matchers` — so a missing matcher means the camera is silently dropped (`No cameras with matchers to process` warning) and frames pile up with no inference running.

Fast path in `matchers/factory.py`:
- Detect `function_type == 'Check_Color'` AND template has `product` annotation.
- Skip TRT engine load + template-bbox requirement.
- Return a `ColorCheckStubMatcher` that carries only `crop_area` (parsed via `AnnotationParser._parse_crop_area`).
- Stub has `match_batch()` returning empty success in case of mis-routing.

`single_camera.run_inference` and `multi_camera.run_inference` both detect `matcher.is_color_check_stub` (or `_is_color_cam`) and skip `match_batch` entirely, synthesizing `transformed_bboxes` from denormalized raw template annotations.

User-visible effect: deleting the `'template'` annotation from a Check_Color template no longer breaks the recipe. As long as `product` is drawn, the camera continues to run color check.

## Cap rotation in multi_camera pipeline (GOTCHA)

Until 2026-05-21, `OBBRotationService` was only invoked from `single_camera.preprocess`. The `multi_camera.preprocess` had **no rotation gate** — meaning a 3-camera recipe (e.g. 2 color + 1 datecode-on-cap) had its cap camera processed without rotation regardless of recipe's `cap_rotation_method`. Symptom: OCR on cap consistently failed because text wasn't aligned upright.

Fix (`multi_camera.preprocess`): per-camera rotation gate, mirrors `single_camera`:

```python
_need_rotation = (
    camera.function_type == 'Check_Color'
    and not first_template_has_product
)
if _need_rotation:
    _rot_svc = (context.cv_rotation_service
                if camera.cap_rotation_method == 'yolo_segment'
                else context.obb_rotation_service)
    if _rot_svc.available:
        rotated, _ = _rot_svc.rotate_frame(frame, ...)
        context.results[sn]['frames'][0] = rotated  # propagate to save/viz
        frame = rotated
```

Rotation is cap-only (circular mask) so it happens **before** crop without affecting `crop_area` coordinates.

## OBBRotationService 180° flip via shape-match

Legacy `compute_need_flip` in `obb_rotator.py` used `M @ text_center → check new_y < cap_cy`. This is fragile when `text_box.cy ≈ cap.cy` (text centered on cap) — small floating-point differences flip the wrong way. Symptom: caps with text centered on the disc get rotated upside-down ~50% of the time.

Fixed: `OBBRotationService.rotate_frame` now imports `cv_rotator._need_flip` (rendered-"BEST" template-match at 0° vs 180°). Falls back to legacy `compute_need_flip` if shape-match throws. Log line includes `flip_via=shape-match (s0=... s180=...)` or `flip_via=legacy heuristic`.

Both `OBBRotationService` and `CVRotationService` now use the **same** shape-match flip resolution — the only difference between methods is whether the cap+angle detection uses TRT engine or HoughCircles+projection-profile.

## Realtime update gotcha — function_type / color_config silent drop

`InferenceRealtime.applyRealtimeAndPersist` runs TWO calls in a row:
1. `updateRecipeRealtime(recipeId, patch)` → BE `update-realtime` (in-memory, BE preserves DB values for omitted fields).
2. `updateReceipt(recipeId, dbPayload)` → BE `PUT /recipes/{id}` (persists to DB via Pydantic `RecipeUpdate`).

**Trap**: if `dbPayload.camera_templates[i]` omits `function_type`, Pydantic `CameraTemplates.function_type` defaults to `"OCR"` → DB silently overwrites the recipe's `Check_Color` (or other) value. Same for nested `templates[].color_config` if the build helper strips it.

`InferenceRealtime.buildFullRecipePayloadForDB` and `InferenceRealtimeSettingsModal` submit payload MUST include:
- `camera_templates[i].function_type`
- `camera_templates[i].templates[j].color_config`

Both fields preserved via `?? ct.function_type ?? 'OCR'` / `?? tmpl.color_config ?? null`. BE `update-realtime` also merges these now (per-camera function_type, per-template color_config) so realtime patches don't quietly diverge from DB.

## Cap rotation method — recipe-level `cap_rotation_method`

Pattern mirrors `product_detection_method`: recipe-level setting, applies to **all cameras** in the recipe. Two options:

| Value | Path |
|---|---|
| `yolo_obb` (default) | `OBBRotationService` → `best_bottle_m.engine` TRT model. Needs GPU. Detects `bottle_cap` + `text_box` OBBs, rotates by `text_box` angle, flips 180° if text is above cap center. |
| `yolo_segment` | `CVRotationService` (pure CV). HoughCircles → cap. Otsu inverse threshold inside cap → text mask. Projection-profile search over [-90°, 90°) → reading-direction angle. Render "BEST" via cv2.putText, template-match (TM_CCOEFF_NORMED) at 0° vs 180° → 180° flip resolution. Rotates cap region only (circular mask). |

UI: Recipe Form → **Model tab** → "🧢 Cap Rotation Method" dropdown directly under "🍶 Bottle Edge Detection".

Field plumbing (19-step):
- `backend/app/models/recipe.py` + `schemas/recipe.py`: `cap_rotation_method: Optional[str] = "yolo_obb"`.
- `recipes.py`: `recipe_to_response`, `clone_recipe`, `load_recipe` (metadata + recipe_dict), `update_realtime` (recipe_dict).
- `frontend-ts/src/types/index.ts` Recipe + Receipt.
- `RecipeFormModal.tsx`: FormDataType + initial state + edit-load + create-reset + handleSubmit + dropdown UI.
- `Receipts.tsx`: 2 transformedReceipts (load + clone) — search path inherits from load.
- AI service `camera.py`: `Camera.cap_rotation_method` field (default `"yolo_obb"`) + parse in `load_recipe`.
- AI service `inference_handler.py`: instantiate both `OBBRotationService` and `CVRotationService`, inject both into `PipelineContext`.
- AI service `pipeline/base.py`: `cv_rotation_service: Any = None` on `PipelineContext`.
- AI service `pipeline/single_camera.py:101`: `_rotation_service = context.cv_rotation_service if camera.cap_rotation_method == 'yolo_segment' else context.obb_rotation_service`.
- BE `services/rotate_cv_service.py`: standalone module exposing `rotate_frame_cv(image)` for the `/api/cameras/frames/rotate` endpoint (parallel to `rotate_obb_service.rotate_frame`).
- BE `api/endpoints/cameras.py`: `/frames/rotate` body accepts `method: str = "yolo_obb"`.

Pure-CV algorithm summary (`backend/app/services/rotate_cv_service.py` + `ai_services/.../preprocessing/cv_rotator.py` — same algorithm, two homes because BE and AI service are separate Python apps):

1. **Cap detection**: `cv2.HoughCircles` with radius 12-35% of `min(H, W)`, `param2=50`, validate interior brightness ≥ 140.
2. **Text mask**: Otsu inverse threshold inside cap (shrunk 0.88×) → `MORPH_OPEN` clean.
3. **Reading angle**: rotate text mask by candidate θ ∈ [-90°, 90°) in 2° steps, pick θ maximizing `var(horizontal_projection)`. Refine ±2° at 0.5°.
4. **180° flip**: render `"BEST"` via cv2.putText at multiple scales, `cv2.matchTemplate(rotated_roi, tpl, TM_CCOEFF_NORMED)` at 0° and 180°, flip if score_180 > score_0.
5. **Output**: rotate cap region only (`warpAffine` + circular mask), background untouched. Returns `(rotated_frame, None)` — `None` matches OBB service's convention (no inverse-transform needed because rotation is local to cap region).

## Related memory
- [[recipe-system]] — recipe data flow, the 19-step plumb checklist (color_config skips most of it because it's nested per-template)
- [[feedback-ui-theme]] — modal CSS theme conventions
