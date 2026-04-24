# ML Golden Template — Implementation Plan

**Goal**: Cognex OCVMax-style character defect detection — per-char golden reference + alignment + difference-map features. Expected accuracy gain: +15-25% on top of local-grid features (Option A).

**Approach**: Option 3 — auto-import `char_id` from recipe (user xác nhận để tránh cluster nhầm `0/O`, `s/S`).

---

## 🎯 Phase strategy

### Phase 1 — Backend + Frontend (train + test trước)

User có thể tạo project, import từ recipe, train model với golden, test predict trên FE → **validate accuracy trước khi deploy AI service**.

### Phase 2 — AI Service integration (sau khi Phase 1 test OK)

AI service load bundle format mới, pass `char_id` từ recipe bbox vào `classify_batch`, inference production.

**Phase 1 độc lập với Phase 2** — Phase 1 tự đủ để train/test; Phase 2 chỉ là tích hợp runtime.

---

## 🏗️ Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│ Recipe (đã có, KHÔNG sửa)                                │
│   camera_template.annotations                            │
│     { type: 'text', text: 'B', points: [...] }           │
└──────────────────┬───────────────────────────────────────┘
                   │
                   │ (1) Import-from-recipe (PHASE 1)
                   ↓
┌──────────────────────────────────────────────────────────┐
│ ML Project (char_id auto-populated)                      │
│   annotation.region.segment {                            │
│     char_id: 'B',          ← từ recipe                   │
│     label: 'OK' | 'NG',    ← user label tay              │
│   }                                                       │
└──────────────────┬───────────────────────────────────────┘
                   │
                   │ (2) Training (PHASE 1)
                   ↓
┌──────────────────────────────────────────────────────────┐
│ build_dataset                                            │
│   - Group OK by char_id                                  │
│   - compute_golden per char → dict {'B': 48×48, ...}     │
│   - Augment OK (mild) + NG (destructive) balanced        │
│                                                           │
│ extract_features_v2(crop, char_id, goldens) → 1016 dim   │
│   - CLAHE preprocess                                     │
│   - Align to goldens[char_id] (cv2.matchTemplate ±5px)   │
│   - Base 856 features + diff 160 features                │
│                                                           │
│ Train RF → save bundle: {clf, goldens, feat_version}     │
└──────────────────┬───────────────────────────────────────┘
                   │
                   │ (3) Test on FE (PHASE 1)
                   ↓
┌──────────────────────────────────────────────────────────┐
│ /ml/projects/{id}/predict (existing endpoint)            │
│   load bundle → use goldens for alignment + diff         │
│   FE hiển thị results, user validate accuracy            │
└──────────────────┬───────────────────────────────────────┘
                   │
                   │ (4) AI Service integration (PHASE 2)
                   ↓
┌──────────────────────────────────────────────────────────┐
│ text_verifier → ml_classifier.classify_batch             │
│   Pass expected_text as char_id per item                 │
│   Bundle load + golden alignment                         │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 File references

### Phase 1 — BE

| File | Lý do sửa |
|------|-----------|
| `backend/app/models/ml_training.py` | Thêm `char_id: Optional[str]` vào `MLSegmentBase`/`MLSegmentInDB` |
| `backend/app/services/ml_training_service.py` | Thêm `compute_golden`, `align_to_golden`, `preprocess_canonical`, `extract_features_v2`, `extract_diff_features`; update `build_dataset`, `train_model`, `predict_on_image` |
| `backend/app/api/endpoints/ml_training.py` | Thêm endpoint `POST /ml/projects/{id}/import-from-recipe` |
| `backend/app/repositories/ml_training_repository.py` | Nếu cần helper upsert annotation bulk (cho import) |
| `backend/app/repositories/recipe_repository.py` | **Read-only**, dùng để load recipe khi import |

### Phase 1 — FE

| File | Lý do sửa |
|------|-----------|
| `frontend-ts/src/services/mlTraining.ts` | Thêm API `importFromRecipe`; cập nhật `SyntheticCrop`/`LabeledCrop` type để include `char_id` |
| `frontend-ts/src/components/ml-training/TrainTab.tsx` | Hiển thị `char_id` badge trên mỗi crop preview; có thể thêm filter theo char |
| `frontend-ts/src/components/ml-training/LabelTab.tsx` | (hoặc file tương đương) Hiển thị `char_id` badge khi label; cho phép user override char_id nếu import sai |
| `frontend-ts/src/components/ml-training/MLProjectManager.tsx` (hoặc tương đương) | Button "Import from Recipe" — mở modal chọn recipe + filenames |
| `frontend-ts/src/types/index.ts` | Add `char_id?: string` vào types liên quan nếu cần |

### Phase 2 — AI Service (sau khi Phase 1 test OK)

| File | Lý do sửa |
|------|-----------|
| `ai_services/camera_management/verification/ml_classifier.py` | `load_model` handle bundle format `{clf, goldens}`; `classify_batch` nhận `char_id` per item; `extract_features` port v2 (giữ đồng bộ với BE) |
| `ai_services/camera_management/verification/text_verifier.py` | `_build_items_for_camera` pass `bbox.get('text') or expected_text` as `char_id` vào ml_items |

---

## 🔧 Phase 1 — Chi tiết implementation

### 1. Schema — `MLSegmentInDB`

**File**: `backend/app/models/ml_training.py`

```python
class MLSegmentBase(BaseModel):
    id: str
    x: float
    y: float
    w: float
    h: float
    label: Optional[str] = None         # 'OK' | 'NG' | None (chưa label)
    char_id: Optional[str] = None       # ← MỚI: 'B', 'E', '0', '/', ...
```

MongoDB schema = flexible nên không cần migration data cũ. Sample cũ `char_id = None` → fallback no-golden mode.

### 2. Core functions — `ml_training_service.py`

**File**: `backend/app/services/ml_training_service.py`

#### `preprocess_canonical(img) -> np.ndarray(48,48, uint8)`
```python
def preprocess_canonical(img):
    gray = _to_gray(img)
    # CLAHE contrast normalization (Cognex style)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normed = clahe.apply(gray)
    # Resize preserve aspect, center-pad to 48×48
    h, w = normed.shape[:2]
    scale = min(48 / w, 48 / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv.resize(normed, (nw, nh), interpolation=cv.INTER_AREA)
    canvas = np.zeros((48, 48), dtype=np.uint8)
    yo, xo = (48 - nh) // 2, (48 - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas
```

#### `compute_golden(ok_crops) -> np.ndarray(48,48, uint8)`
```python
def compute_golden(ok_crops):
    if len(ok_crops) < 2:
        return None
    canonical = [preprocess_canonical(c).astype(np.float32) for c in ok_crops]
    return np.mean(canonical, axis=0).astype(np.uint8)
```

#### `align_to_golden(input_48, golden_48, search=5) -> (aligned, offset)`
```python
def align_to_golden(input_48, golden_48, search=5):
    padded = cv.copyMakeBorder(input_48, search, search, search, search, cv.BORDER_REPLICATE)
    result = cv.matchTemplate(padded, golden_48, cv.TM_CCOEFF_NORMED)
    _, _, _, (mx, my) = cv.minMaxLoc(result)
    dx, dy = mx - search, my - search
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv.warpAffine(input_48, M, (48, 48), borderMode=cv.BORDER_REPLICATE)
    return aligned, (dx, dy)
```

#### `extract_diff_features(diff_48) -> np.ndarray(160)`
```python
def extract_diff_features(diff):
    feats = []
    # 6×6 grid × 4 stats = 144
    cs = 8
    for i in range(6):
        for j in range(6):
            cell = diff[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
            feats.append(float(cell.mean()) / 255)
            feats.append(float(cell.max()) / 255)
            feats.append(float(cell.std()) / 128)
            feats.append(float(np.count_nonzero(cell > 30)) / (cs * cs))  # significant diff pixels
    # 4×4 region-max = 16 (coarser view)
    rs = 12
    for i in range(4):
        for j in range(4):
            region = diff[i*rs:(i+1)*rs, j*rs:(j+1)*rs]
            feats.append(float(region.max()) / 255)
    return np.asarray(feats, dtype=np.float32)
```

#### `extract_features_v2(img, char_id=None, goldens=None) -> np.ndarray(1016)`

Signature:
- Base 856 features extracted from aligned input (hoặc non-aligned nếu không có golden)
- +160 diff features (zero nếu không có golden → model học ignore)

```python
FEAT_DIM_V2 = 856 + 160  # 1016

def extract_features_v2(img, char_id=None, goldens=None):
    canvas = preprocess_canonical(img)   # 48×48 CLAHE-normalized
    
    if char_id and goldens and char_id in goldens:
        aligned, _ = align_to_golden(canvas, goldens[char_id])
        diff = np.abs(aligned.astype(np.int16) - goldens[char_id].astype(np.int16)).astype(np.uint8)
    else:
        aligned = canvas
        diff = np.zeros((48, 48), dtype=np.uint8)
    
    base = _extract_base_features(aligned)      # 856 (current Option A)
    diff_feats = extract_diff_features(diff)     # 160
    return np.concatenate([base, diff_feats])
```

Giữ `extract_features` cũ (v1) để backward compat model cũ. Bundle version field quyết định dùng cái nào.

#### `build_dataset` update

```python
def build_dataset(annotations, images_dir, augment_factor):
    # Group by char_id
    ok_by_char = defaultdict(list)
    ng_by_char = defaultdict(list)
    for ann in annotations:
        for region in ann.regions:
            for seg in region.segments:
                if seg.label not in ('OK', 'NG'):
                    continue
                crop = crop_segment(...)
                if crop is None: continue
                key = seg.char_id or '_unknown'
                (ok_by_char if seg.label == 'OK' else ng_by_char)[key].append(crop)
    
    # Compute golden per char (chỉ char có ≥2 OK samples)
    goldens = {}
    for char_id, crops in ok_by_char.items():
        if char_id == '_unknown' or len(crops) < 2:
            continue
        goldens[char_id] = compute_golden(crops)
    
    # Augment + extract features
    X_rows, y_rows, crops_rows = [], [], []
    for char_id, ok_list in ok_by_char.items():
        for crop in ok_list:
            X_rows.append(extract_features_v2(crop, char_id, goldens))
            y_rows.append(1); crops_rows.append(crop)
        if augment_factor >= 2:
            for crop in ok_list:
                for aug in augment_ok(crop, n=augment_factor - 1):
                    X_rows.append(extract_features_v2(aug, char_id, goldens))
                    y_rows.append(1); crops_rows.append(aug)
    
    # Similar for NG (real + synthetic)
    # ...
    
    return np.array(X_rows), np.array(y_rows), crops_rows, goldens, n_ok_total, n_ng_total
```

#### `train_model` — save bundle

```python
def train_model(...):
    X, y, crops_raw, goldens, n_ok, n_ng = build_dataset(...)
    clf = _build_classifier(request)
    clf.fit(X_train, y_train)
    
    # Save bundle instead of raw classifier
    bundle = {
        'clf': clf,
        'goldens': goldens,   # {char_id: np.ndarray(48,48,uint8)}
        'feat_version': 'v2',
    }
    joblib.dump(bundle, str(model_save_path))
```

#### `predict_on_image` — load bundle

```python
def predict_on_image(model_path, image_path, ...):
    data = joblib.load(str(model_path))
    if isinstance(data, dict) and 'clf' in data:
        clf = data['clf']
        goldens = data.get('goldens', {})
        version = data.get('feat_version', 'v2')
    else:
        clf = data; goldens = {}; version = 'v1'
    
    # Predict using v2 features if bundle, v1 if legacy
    extractor = extract_features_v2 if version == 'v2' else extract_features
    # For predict_on_image standalone, char_id is None → no golden alignment
    # (endpoint này dùng cho test single image upload)
```

### 3. Import-from-recipe endpoint

**File**: `backend/app/api/endpoints/ml_training.py`

```python
from app.repositories.recipe_repository import RecipeRepository

@router.post("/ml/projects/{project_id}/import-from-recipe")
async def import_from_recipe(
    project_id: str,
    body: dict,     # { recipe_id, camera_serial, filenames: [str] }
    repo: MLTrainingRepository = Depends(get_repo),
    recipe_repo: RecipeRepository = Depends(get_recipe_repository),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Pre-populate ML annotations from recipe template's text/datecode bboxes.
    Each auto-created segment inherits char_id from bbox.expected_text.
    User labels OK/NG afterwards.

    Partial-success semantics: 1 file lỗi không rollback toàn bộ; trả về
    detail errors cho FE hiển thị bảng status.
    """
    recipe = await recipe_repo.get_by_id(body['recipe_id'])
    camera = next((c for c in recipe.cameras if c.serial_number == body['camera_serial']), None)
    templates = recipe.camera_templates.get(camera.id, []) if camera else []

    imported, skipped, errors = 0, 0, []
    char_ids_seen = set()

    for filename in body['filenames']:
        try:
            image_path = _images_dir(project_id) / filename
            if not image_path.exists():
                raise FileNotFoundError(f"{filename} not found in project images")

            # Build annotations from recipe bboxes
            regions = []
            for tpl in templates:
                for ann in tpl.annotations:
                    if ann.type not in ('text', 'datecode'):
                        continue
                    xs = [p[0] for p in ann.points]; ys = [p[1] for p in ann.points]
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y
                    # Build MLRegion with 1 MLSegment matching the bbox
                    segment = MLSegmentBase(
                        id=str(uuid.uuid4()),
                        x=x, y=y, w=w, h=h,
                        label=None,               # user labels later
                        char_id=ann.text or None, # ← auto-populated from recipe
                    )
                    regions.append(MLRegionBase(
                        id=str(uuid.uuid4()),
                        x=x, y=y, w=w, h=h,
                        segments=[segment],
                    ))
                    if ann.text:
                        char_ids_seen.add(ann.text)

            if not regions:
                raise ValueError("No text/datecode annotations in recipe template")

            await repo.save_annotation(
                project_id, filename,
                MLAnnotationSave(regions=regions),
            )
            imported += 1
        except Exception as e:
            skipped += 1
            errors.append({"filename": filename, "reason": str(e)})

    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "char_ids": sorted(char_ids_seen),
    }
```

**Limitations Phase 1**: chỉ work khi filename là template image (1 ảnh gốc của recipe). Production images phải match với matcher — để Phase 2.

### 4. Char-coverage endpoint (Phase 1)

**File**: `backend/app/api/endpoints/ml_training.py`

```python
@router.get("/ml/projects/{project_id}/models/{model_id}/char-coverage")
async def char_coverage(
    project_id: str,
    model_id: str,
    chars: str,    # comma-separated: "A,B,C"
    repo: MLTrainingRepository = Depends(get_repo),
    current_user: UserInDB = Depends(get_current_user),
):
    """Return which chars from `chars` are covered by the model's goldens."""
    model_record = await repo.get_model(model_id)
    if not model_record:
        raise HTTPException(404, "Model not found")

    bundle = joblib.load(model_record.model_path)
    goldens_chars = set((bundle.get('goldens') or {}).keys()) if isinstance(bundle, dict) else set()

    requested = [c.strip() for c in chars.split(',') if c.strip()]
    covered = [c for c in requested if c in goldens_chars]
    missing = [c for c in requested if c not in goldens_chars]

    return {
        "covered": covered,
        "missing": missing,
        "coverage_pct": round(len(covered) / len(requested) * 100, 1) if requested else 0.0,
        "model_chars": sorted(goldens_chars),
    }
```

### 5. Diff heatmap trong prediction response (Phase 1)

**File**: `backend/app/services/ml_training_service.py` → `predict_on_image`

```python
def predict_on_image(model_path, image_path, region=None, threshold=0.5, char_id=None):
    ...
    # Nếu có golden cho char_id → compute aligned + diff
    if char_id and char_id in goldens:
        aligned = align_to_golden(preprocess_canonical(crop), goldens[char_id])[0]
        diff = np.abs(aligned.astype(np.int16) - goldens[char_id].astype(np.int16)).astype(np.uint8)
        # Encode với JET colormap cho heatmap visualization
        heatmap = cv.applyColorMap(diff, cv.COLORMAP_JET)
        diff_b64 = img_to_b64(heatmap)
        aligned_b64 = img_to_b64(cv.cvtColor(aligned, cv.COLOR_GRAY2BGR))
        golden_b64 = img_to_b64(cv.cvtColor(goldens[char_id], cv.COLOR_GRAY2BGR))
    else:
        diff_b64, aligned_b64, golden_b64 = None, None, None

    results.append({
        ...,
        'char_id': char_id,
        'aligned_b64': aligned_b64,     # input sau khi align to golden
        'golden_b64': golden_b64,       # golden reference for this char
        'diff_b64': diff_b64,           # heatmap of diff (JET colormap)
    })
```

### 4. FE — Import button + char_id badge

**File**: `frontend-ts/src/services/mlTraining.ts`
```typescript
export interface SyntheticCrop { crop_b64: string; label: string; char_id?: string; }
export interface LabeledCrop { ... char_id?: string; }

export const mlTrainingAPI = {
  ...
  importFromRecipe: (projectId: string, recipeId: string, cameraSerial: string, filenames: string[]) =>
    api.post<{ imported: number; char_ids_used: string[] }>(
      `/ml/projects/${projectId}/import-from-recipe`,
      { recipe_id: recipeId, camera_serial: cameraSerial, filenames },
    ).then(r => r.data),
};
```

**File**: `frontend-ts/src/components/ml-training/TrainTab.tsx` — hiển thị char_id badge trên CropGrid items:
```tsx
{item.char_id && (
  <span className="char-id-badge">{item.char_id}</span>
)}
```

**Import button** — component riêng (có thể là modal trong project manager hoặc Label tab):
- Dropdown select recipe + camera
- Multi-select filenames from available snapshots
- Call `importFromRecipe` → refresh annotations list
- Toast: "Imported N annotations, char_ids: [B, E, S, T, ...]"

---

## 🔧 Phase 2 — AI Service integration (sau khi test Phase 1 OK)

### File: `ai_services/camera_management/verification/ml_classifier.py`

1. `load_model` handle bundle:
```python
def load_model(self, project_id, model_id):
    data = joblib.load(path)
    if isinstance(data, dict) and 'clf' in data:
        clf = data['clf']
        goldens = data.get('goldens', {})
    else:
        clf = data; goldens = {}
    self._cache[(project_id, model_id)] = (clf, goldens)
    return clf, goldens
```

2. Port `extract_features_v2`, `compute_golden`, `align_to_golden` (copy từ BE để sync).

3. `classify_batch(items)` — mỗi item giờ cần `char_id`:
```python
def classify_batch(self, items):
    # items: [{region_img, project_id, model_id, char_id, conf_threshold, ...}]
    groups = defaultdict(list)
    for i, item in enumerate(items):
        key = (item['project_id'], item['model_id'])
        groups[key].append((i, item))
    
    for (pid, mid), group in groups.items():
        clf, goldens = self.load_model(pid, mid)
        feats = np.array([
            extract_features_v2(it['region_img'], it.get('char_id'), goldens)
            for _, it in group
        ])
        probas = clf.predict_proba(feats)
        # distribute results
```

### File: `ai_services/camera_management/verification/text_verifier.py`

`_build_items_for_camera` — chỉ add vào ml_items nếu char_id có golden trong model:

```python
# Preload goldens set từ bundle (cache theo project_id+model_id)
goldens_chars = self.ml_classifier_service.get_golden_chars(ml_project_id, ml_model_id)
# Returns set of char_ids available in the model; empty if model not loaded yet

if use_ml_task and expected_text and expected_text in goldens_chars:
    # Full ML — có golden, chạy align + diff
    ml_items.append({
        'cropped_region': cropped,
        'serial_number': serial_number,
        'annotation_idx': ann_idx,
        'conf_threshold': conf_threshold,
        'ml_project_id': ml_project_id,
        'ml_model_id': ml_model_id,
        'char_id': expected_text,
    })
else:
    # Skip ML — char không có trong model
    # → region['match'] = ocr_match only (không AND với ml_pass)
    # → region['ml_skipped_reason'] = 'char_not_in_model'
    pass
```

Trong `_run_ocr_batch_with_checks`, khi merge kết quả, nếu bbox không có ml_result:
```python
region['ml_pass'] = None
region['ml_p_ok'] = None
region['ml_label'] = None
region['ml_skipped_reason'] = 'char_not_in_model' if use_ml else None
# match vẫn = match (chỉ OCR, không AND ml)
```

### `ml_classifier.get_golden_chars` helper

```python
def get_golden_chars(self, project_id: str, model_id: str) -> set:
    """Return set of char_ids available in the model's goldens (for skip-decision)."""
    if project_id is None or model_id is None:
        return set()
    bundle = self._cache.get((project_id, model_id))
    if bundle is None:
        bundle = self.load_model(project_id, model_id)
        if bundle is None:
            return set()
    goldens = bundle.get('goldens') if isinstance(bundle, dict) else None
    return set(goldens.keys()) if goldens else set()
```

---

## 🧪 Testing plan (Phase 1)

1. **Unit test `compute_golden`**: n=5 ảnh char "B" → golden shape (48,48) uint8, visually giống "B".
2. **Unit test `align_to_golden`**: input shifted +3px → alignment recover offset (-3, 0).
3. **Unit test `extract_features_v2`**:
   - Với golden: diff features non-zero cho crop khác golden.
   - Không golden: diff features = 0.
   - Dim đúng 1016.
4. **Integration train**: project với 3 char (B, E, S), mỗi char 10 OK + 2 NG → train model, check acc > 85%.
5. **Integration predict**: upload 1 image → FE hiển thị kết quả với `char_id`, `p_ok`, heatmap diff (optional).
6. **Import flow**: tạo project mới, import từ recipe → check annotations có char_id đúng với `expected_text`.

---

## ✅ Design decisions (đã chốt)

| # | Question | Decision |
|---|----------|----------|
| 1 | Min OK samples per char để compute golden | **≥ 5 samples**. Nếu < 5 → skip golden cho char đó (log warning trong training metrics) |
| 2 | Legacy model compat (v1 features 1120 dim) | Giữ v1 code **~1 tháng**, detect qua bundle `feat_version` field. Sau đó cleanup. |
| 3 | Import-from-recipe: 1 file lỗi → xử lý? | **Partial success**: skip file lỗi, tiếp tục. Response trả `{imported: N, skipped: M, errors: [{filename, reason}]}`. FE hiển thị bảng summary. |
| 4 | FE Import UI | **Modal dialog** — click button "Import from Recipe" → popup. Fields: recipe dropdown, camera dropdown, multi-select filenames, [Cancel] [Import]. |
| 5 | Diff heatmap preview trong prediction result | **Có hiển thị** — BE encode diff_map → base64 (JET colormap) → FE show 3 panels: Input / Golden / Diff heatmap. Giúp user validate model nhìn đúng vùng defect. |
| 6 | Recipe có char không được train (missing from goldens) | **Skip ML hoàn toàn cho bbox đó**. Không degraded mode. ML signal = None. Final `match = ocr_match` (không AND với ml_pass). FE warning khi user chọn model có coverage < 100%. |

## ⚠️ Missing-char handling (Q6 chi tiết)

### 3 lớp defense

**1. FE check — khi user chọn ML model cho recipe**

Endpoint mới:
```
GET /ml/projects/{project_id}/models/{model_id}/char-coverage?chars=A,B,X,Y,Z

Response:
{
  "covered":     ["A", "B"],
  "missing":     ["X", "Y", "Z"],
  "coverage_pct": 40
}
```

UI hiển thị:
```
⚠️ Model coverage: 2/5 chars (40%)
   Missing: [X, Y, Z]
   ML check sẽ bị SKIP cho 3 bbox có chars này.
   [Add missing chars to project →]
```

User vẫn **được phép** chọn model (production có thể chấp nhận ML partial coverage cho phase test).

**2. AI service runtime — skip ML cho bbox missing char**

```python
# Trong text_verifier._build_items_for_camera
if use_ml_task and expected_text in goldens_chars_set:
    ml_items.append({
        'cropped_region': cropped,
        'char_id': expected_text,
        ...
    })
# else: không append → bbox đó không có ML signal → match = ocr_match only
```

`goldens_chars_set` lấy từ bundle đã load. Cache theo (project_id, model_id).

**3. Result struct reflect trạng thái**

```python
region_result = {
    'annotation_idx': ...,
    'match': ocr_match,     # ML không AND vào vì skipped
    'ml_pass': None,         # Marker: "ML was not run for this bbox"
    'ml_p_ok': None,
    'ml_label': None,
    'ml_skipped_reason': 'char_not_in_model',   # ← MỚI, FE dùng để hiển thị
}
```

FE render:
- `ml_pass === null && ml_skipped_reason === 'char_not_in_model'` → badge "ML: SKIPPED (no golden)"
- `ml_pass === true/false` → badge OK/NG với màu

### Có bắt user retrain không?

**Không bắt** — production có thể chạy với partial coverage. Nhưng FE **strongly suggest**:
- Warning visible khi select model
- Link trực tiếp đến project labeling page với filter "missing chars"
- Sau retrain: user chọn lại model → warning biến mất

---

## 📋 Effort breakdown

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Schema + `char_id` field | 30min |
| 1 | Core functions (`preprocess_canonical`, `compute_golden`, `align_to_golden`, `extract_diff_features`, `extract_features_v2`) | 2h |
| 1 | `build_dataset`, `train_model` (bundle save), `predict_on_image` (heatmap) | 1.5h |
| 1 | `import-from-recipe` endpoint | 1.5h |
| 1 | `char-coverage` endpoint | 30min |
| 1 | FE: types + API services (`importFromRecipe`, `charCoverage`) | 1h |
| 1 | FE: Import modal dialog | 1.5h |
| 1 | FE: `char_id` badge + Label UI | 1h |
| 1 | FE: Prediction result heatmap (3-panel Input/Golden/Diff) | 1h |
| 1 | FE: Recipe "Select Model" coverage warning | 1h |
| 1 | Testing + debugging | 2h |
| **Phase 1 total** | | **~13h** |
| 2 | AI service: bundle load, `classify_batch` + `char_id` per item | 1.5h |
| 2 | AI service: port `extract_features_v2` + `compute_golden` (sync BE) | 1h |
| 2 | `text_verifier`: pass `expected_text` as `char_id`; skip ML when char missing | 1h |
| 2 | Result struct: `ml_skipped_reason` field + FE badge update | 30min |
| 2 | Integration test end-to-end | 1h |
| **Phase 2 total** | | **~5h** |

---

## 🔒 Backward compatibility

- Old models (no bundle): loaded as legacy, dùng features v1, không có golden → inference vẫn chạy nhưng không tận dụng improvement
- Old segments (no `char_id`): gán '_unknown' → fallback no-golden path trong training
- Old FE: không hiển thị char_id badge, hoạt động bình thường
- Recipe model: KHÔNG đổi (chỉ READ)

---

## 🚀 Rollout

1. Deploy BE với Phase 1 code
2. User tạo 1 project test, import từ recipe có 5-10 char
3. Label OK/NG cho segments (char_id đã auto)
4. Train → check metrics confusion_matrix
5. Predict test trên 5-10 new images → validate accuracy
6. Nếu OK → Phase 2 deploy AI service
7. Recipe update trỏ model mới → production test
