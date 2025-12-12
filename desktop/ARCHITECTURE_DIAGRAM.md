# Inference Tab - Architecture Diagram

## Application Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      MainWindow (QMainWindow)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              QTabWidget (mainTabs)                     │     │
│  ├────────────────────────────────────────────────────────┤     │
│  │                                                        │     │
│  │  ┌──────────────────┐  ┌─────────────────────────┐    │     │
│  │  │  Tab 1           │  │  Tab 2                  │    │     │
│  │  │  "Annotation"    │  │  "Inference"            │    │     │
│  │  └──────────────────┘  └─────────────────────────┘    │     │
│  │                                                        │     │
│  │  ┌──────────────────────────────────────────────┐     │     │
│  │  │         Active Tab Content                   │     │     │
│  │  │                                              │     │     │
│  │  │  When Tab 1 (Annotation):                   │     │     │
│  │  │    - Image Viewer                           │     │     │
│  │  │    - BBox List                              │     │     │
│  │  │    - Annotation Tools                       │     │     │
│  │  │                                              │     │     │
│  │  │  When Tab 2 (Inference):                    │     │     │
│  │  │    - InferenceWidget (see below)            │     │     │
│  │  └──────────────────────────────────────────────┘     │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Inference Widget Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    InferenceWidget (QWidget)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                  QHBoxLayout (Main)                     │     │
│  ├────────────────────────────────────────────────────────┤     │
│  │                                                        │     │
│  │  ┌──────────────────────────────────────────────────┐ │     │
│  │  │         QSplitter (Horizontal, 3 panels)         │ │     │
│  │  ├──────────────────────────────────────────────────┤ │     │
│  │  │  ╔════════════════════════════════════════════╗  │ │     │
│  │  │  ║  Left Panel (220-300px)                   ║  │ │     │
│  │  │  ╠════════════════════════════════════════════╣  │ │     │
│  │  │  ║  [Template Configuration]                 ║  │ │     │
│  │  │  ║  • JSON Path Label                        ║  │ │     │
│  │  │  ║  • [Load Annotations JSON] Button         ║  │ │     │
│  │  │  ║                                           ║  │ │     │
│  │  │  ║  [Test Images]                            ║  │ │     │
│  │  │  ║  • Folder Path Label                      ║  │ │     │
│  │  │  ║  • [Load Image Folder] Button             ║  │ │     │
│  │  │  ║                                           ║  │ │     │
│  │  │  ║  [Images]                                 ║  │ │     │
│  │  │  ║  • QListWidget (image files)              ║  │ │     │
│  │  │  ║                                           ║  │ │     │
│  │  │  ║  • [◄ Prev] [Next ►] Buttons              ║  │ │     │
│  │  │  ║  • Info Label                             ║  │ │     │
│  │  │  ╚════════════════════════════════════════════╝  │ │     │
│  │  │                                                  │ │     │
│  │  │  ╔════════════════════════════════════════════╗  │ │     │
│  │  │  ║  Center Panel (flexible)                  ║  │ │     │
│  │  │  ╠════════════════════════════════════════════╣  │ │     │
│  │  │  ║  [Image Preview]                          ║  │ │     │
│  │  │  ║                                           ║  │ │     │
│  │  │  ║  ┌──────────────────────────────────┐     ║  │ │     │
│  │  │  ║  │                                  │     ║  │ │     │
│  │  │  ║  │   QLabel (Image Display)         │     ║  │ │     │
│  │  │  ║  │   - Shows original image         │     ║  │ │     │
│  │  │  ║  │   - Shows result with bboxes     │     ║  │ │     │
│  │  │  ║  │                                  │     ║  │ │     │
│  │  │  ║  └──────────────────────────────────┘     ║  │ │     │
│  │  │  ║                                           ║  │ │     │
│  │  │  ║  • QProgressBar (hidden until running)    ║  │ │     │
│  │  │  ║  • Status Label                           ║  │ │     │
│  │  │  ║  • [Run Inference] Button                 ║  │ │     │
│  │  │  ╚════════════════════════════════════════════╝  │ │     │
│  │  │                                                  │ │     │
│  │  │  ╔════════════════════════════════════════════╗  │ │     │
│  │  │  ║  Right Panel (250-400px)                  ║  │ │     │
│  │  │  ╠════════════════════════════════════════════╣  │ │     │
│  │  │  ║  [OCR Results]                            ║  │ │     │
│  │  │  ║                                           ║  │ │     │
│  │  │  ║  ┌──────────────────────────────────┐     ║  │ │     │
│  │  │  ║  │ QTextEdit (Read-only)            │     ║  │ │     │
│  │  │  ║  │                                  │     ║  │ │     │
│  │  │  ║  │ Shows:                           │     ║  │ │     │
│  │  │  ║  │ • Confidence                     │     ║  │ │     │
│  │  │  ║  │ • Inliers count                  │     ║  │ │     │
│  │  │  ║  │ • Processing time                │     ║  │ │     │
│  │  │  ║  │ • Detected regions               │     ║  │ │     │
│  │  │  ║  │ • OCR text (TODO)                │     ║  │ │     │
│  │  │  ║  │                                  │     ║  │ │     │
│  │  │  ║  └──────────────────────────────────┘     ║  │ │     │
│  │  │  ║                                           ║  │ │     │
│  │  │  ║  • [Export Results] Button (disabled)     ║  │ │     │
│  │  │  ╚════════════════════════════════════════════╝  │ │     │
│  │  └──────────────────────────────────────────────────┘ │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Processing Flow

```
┌──────────────────┐
│  User Actions    │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  1. Load Annotations JSON                              │
│     ↓                                                   │
│  2. Initialize SuperPointMatcherONNX                   │
│     • Load template image                              │
│     • Parse annotations                                │
│     • Load ONNX model                                  │
│     • Initialize CUDA/CPU                              │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  3. Load Image Folder                                  │
│     • Scan for supported formats                       │
│     • Populate image list                              │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  4. Select Image                                       │
│     • Load and display image                           │
│     • Enable inference button                          │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  5. Click "Run Inference"                              │
│     ↓                                                   │
│  6. Create InferenceWorker Thread                      │
│     • Pass matcher and image path                      │
│     • Connect signals                                  │
│     • Start thread                                     │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  InferenceWorker.run() [Background Thread]             │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 7. matcher.match(image_path)                     │  │
│  │    ├─ Load target image                          │  │
│  │    ├─ Resize to 32x multiple                     │  │
│  │    ├─ Convert to tensors                         │  │
│  │    ├─ Run ONNX inference                         │  │
│  │    ├─ Extract & match keypoints                  │  │
│  │    ├─ Filter by score threshold                  │  │
│  │    ├─ Compute homography (RANSAC)                │  │
│  │    └─ Transform all bboxes                       │  │
│  │                                                   │  │
│  │ 8. Draw bboxes on image                          │  │
│  │    • Color-code by type                          │  │
│  │    • Add labels                                  │  │
│  └──────────────────────────────────────────────────┘  │
│     │                                                   │
│     ▼                                                   │
│  9. Emit signals                                       │
│     • finished(result, annotated_image) [on success]   │
│     • error(error_message) [on failure]                │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  10. UI Updates [Main Thread]                          │
│      ┌─────────────────────────────────────────────┐   │
│      │ on_inference_finished():                    │   │
│      │  • Display annotated image                  │   │
│      │  • Show metrics in results panel            │   │
│      │  • Update status label                      │   │
│      │  • Hide progress bar                        │   │
│      │  • Re-enable inference button               │   │
│      └─────────────────────────────────────────────┘   │
│                  OR                                     │
│      ┌─────────────────────────────────────────────┐   │
│      │ on_inference_error():                       │   │
│      │  • Show error message                       │   │
│      │  • Update status label                      │   │
│      │  • Hide progress bar                        │   │
│      │  • Re-enable inference button               │   │
│      └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

## Class Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                        MainWindow                           │
│                     (QMainWindow)                           │
├─────────────────────────────────────────────────────────────┤
│  • QTabWidget tab_widget                                    │
│  • InferenceWidget inference_tab                            │
│  • (other annotation components)                            │
└───────────────────────┬─────────────────────────────────────┘
                        │ contains
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    InferenceWidget                          │
│                       (QWidget)                             │
├─────────────────────────────────────────────────────────────┤
│  • SuperPointMatcherONNX matcher                            │
│  • InferenceWorker worker (temporary)                       │
│  • UI components (labels, buttons, list, etc.)              │
│                                                             │
│  Methods:                                                   │
│  • load_annotations_json()                                  │
│  • load_image_folder()                                      │
│  • on_image_selected()                                      │
│  • run_inference()                                          │
│  • on_inference_finished()                                  │
│  • on_inference_error()                                     │
└───────────────────────┬─────────────────────────────────────┘
                        │ creates
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                SuperPointMatcherONNX                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  • ort.InferenceSession pipeline_sess                       │
│  • template_img, template_gray                              │
│  • template_bbox, other_bboxes                              │
│  • scale factor                                             │
│                                                             │
│  Methods:                                                   │
│  • _resize_to_32(img)                                       │
│  • match(target_path) → Dict                                │
│  • crop_regions(result) → Dict[str, ndarray]                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    InferenceWorker                          │
│                       (QThread)                             │
├─────────────────────────────────────────────────────────────┤
│  Signals:                                                   │
│  • finished(dict, object)                                   │
│  • error(str)                                               │
│                                                             │
│  Methods:                                                   │
│  • run()                                                    │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
annotations.json → SuperPointMatcherONNX
                         ↓
                   Load template
                   Parse bboxes
                   Init ONNX model
                         ↓
                    [Matcher Ready]
                         ↓
                         ↓
Target Image → InferenceWorker → matcher.match()
                         ↓
                    ONNX Pipeline
                         ↓
                 ┌───────────────┐
                 │ SuperPoint    │ → Keypoints
                 └───────────────┘
                         ↓
                 ┌───────────────┐
                 │ LightGlue     │ → Matches + Scores
                 └───────────────┘
                         ↓
                    Filter (>0.3)
                         ↓
                 ┌───────────────┐
                 │ RANSAC        │ → Homography
                 └───────────────┘
                         ↓
                Transform all bboxes
                         ↓
                 ┌───────────────┐
                 │ Result Dict   │
                 │ • success     │
                 │ • confidence  │
                 │ • bboxes      │
                 │ • timings     │
                 └───────────────┘
                         ↓
                  UI Display
```

---

This architecture provides:
- **Modularity**: Clear separation of concerns
- **Responsiveness**: Async processing
- **Extensibility**: Easy to add OCR
- **Maintainability**: Well-structured code
