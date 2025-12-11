# Inference Tab - User Guide

## Overview

The **Inference Tab** allows you to test and run template matching inference on images using the annotated templates you created in the Annotation tab.

## Features

### Three-Panel Layout

1. **Left Panel**: Configuration and image selection
2. **Center Panel**: Image preview and inference visualization
3. **Right Panel**: OCR results (to be implemented)

## Usage Guide

### Step 1: Load Template Configuration

1. Click **"Load Annotations JSON"** button in the left panel
2. Select the `annotations.json` file from your annotated template folder
3. The system will automatically look for the ONNX model at:
   - `weights/superpoint_lightglue_pipeline.onnx`
4. Wait for the template to load (you'll see initialization info)

**Requirements:**
- Valid `annotations.json` file with template annotations
- ONNX model file must exist in the `weights/` folder
- Template image must exist and be accessible

### Step 2: Load Test Images

1. Click **"Load Image Folder"** button in the left panel
2. Select the folder containing images you want to test
3. The image list will populate with all supported image formats:
   - `.jpg`, `.jpeg`, `.png`, `.bmp`

### Step 3: Run Inference

1. Select an image from the list (or use **Prev/Next** buttons to navigate)
2. The image will display in the center panel
3. Click **"Run Inference"** button
4. Wait for processing (progress bar will show)
5. Results will display:
   - **Center Panel**: Annotated image with detected regions
   - **Right Panel**: Inference metrics and OCR results

### Navigation

- **Click on image**: Select from list
- **◄ Prev**: Go to previous image
- **Next ►**: Go to next image

## Inference Results

When inference completes successfully, you'll see:

### Visual Results (Center Panel)
- Original image with detected bounding boxes overlaid
- Color-coded regions:
  - 🟢 Green: Template region
  - 🟠 Orange: Text regions
  - 🟣 Magenta: Barcode regions
  - 🔵 Cyan: Datecode regions

### Metrics (Right Panel)
- **Confidence**: Match quality (higher is better, > 70% is good)
- **Inliers**: Number of matched keypoints / total matches
- **Processing Time**: Total inference time in milliseconds
- **Detected Regions**: List of found regions

## Troubleshooting

### "ONNX Runtime not available"
**Solution**: Install ONNX Runtime
```bash
pip install onnxruntime-gpu  # For GPU
# or
pip install onnxruntime      # For CPU
```

### "Model Not Found"
**Solution**: Ensure the ONNX model exists at:
```
project_root/
  weights/
    superpoint_lightglue_pipeline.onnx
```

Download or export the model if missing.

### "Too few matches" Error
**Possible causes:**
- Image doesn't contain the template
- Image is too different (rotation, scale, lighting)
- Template quality is poor

**Solutions:**
- Verify the correct template is loaded
- Check image quality
- Try different images
- Re-annotate template with better coverage

### Low Confidence (<50%)
**Possible causes:**
- Partial template visibility
- Significant perspective distortion
- Poor image quality

**Solutions:**
- Adjust score threshold (default: 0.3)
- Check lighting conditions
- Verify template annotations cover distinctive features

## Technical Details

### Model Configuration
- **Scale**: 0.5x (configurable in code)
- **Score Threshold**: 0.3 (minimum match quality)
- **RANSAC Threshold**: 5.0 (homography outlier threshold)

### Processing Pipeline
1. Load template and target images
2. Resize to multiples of 32 for model compatibility
3. Extract SuperPoint features
4. Match features with LightGlue
5. Filter by score threshold
6. Compute homography with RANSAC
7. Transform all annotated regions
8. Display results

### Performance
- Typical inference time: 50-200ms (GPU)
- Depends on:
  - Image resolution
  - GPU availability
  - Number of keypoints

## Future Features (OCR Integration)

The right panel is prepared for OCR results. Future implementation will:

1. Crop detected regions (text, barcode, datecode)
2. Run OCR on each region
3. Display extracted text
4. Allow export of results to JSON/CSV

## Tips for Best Results

1. **Good Templates**:
   - High contrast features
   - Distinctive patterns
   - Consistent lighting
   - Clear boundaries

2. **Test Images**:
   - Similar resolution to template
   - Avoid extreme rotations (>30°)
   - Good lighting
   - Minimal blur

3. **Performance**:
   - Use GPU for faster inference
   - Lower scale factor (0.3-0.5) for speed
   - Batch process multiple images

## Keyboard Shortcuts (Planned)

- `Left Arrow`: Previous image
- `Right Arrow`: Next image
- `Space`: Run inference
- `Ctrl+O`: Load JSON
- `Ctrl+Shift+O`: Load folder

---

**Need Help?** Check the main README or contact the development team.
