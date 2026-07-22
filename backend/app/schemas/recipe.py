from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class TemplateAnnotation(BaseModel):
    """Single annotation in a template"""
    type: str = Field(..., description="Annotation type: template, text, barcode, datecode, crop_area")
    shape: str = Field(..., description="Shape type: rectangle, polygon")
    text: Optional[str] = Field(None, description="Text content for text/datecode annotations")
    conf: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence threshold (0-1)")
    # Rectangle fields - accept float for precision
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    # Polygon fields
    points: Optional[List[List[float]]] = None


class ColorRoiCircle(BaseModel):
    """Circular ROI used to seed HSV range in ColorSetupModal."""
    center: List[float] = Field(default_factory=lambda: [0.0, 0.0], description="[x, y] in template image coords")
    radius: float = Field(default=50.0, ge=1.0, description="Radius in pixels")


class ColorConfig(BaseModel):
    """HSV color check config for Check_Color templates with a 'product' annotation."""
    h_min: int = Field(default=0, ge=0, le=180)
    h_max: int = Field(default=180, ge=0, le=180)
    s_min: int = Field(default=0, ge=0, le=255)
    s_max: int = Field(default=255, ge=0, le=255)
    v_min: int = Field(default=0, ge=0, le=255)
    v_max: int = Field(default=255, ge=0, le=255)
    pixel_threshold: int = Field(default=1000, ge=0, description="Minimum matching HSV pixels (summed across product polygons) required to pass")
    localization_method: Optional[str] = Field(default="image_proc", description="Color ROI localization: 'image_proc' (legacy bottle detect) | 'superpoint' (use SuperPoint-transformed product polygon)")
    roi_circle: Optional[ColorRoiCircle] = Field(default=None, description="Persisted ROI used by ColorSetupModal so user can re-edit")
    bottle_sharp_threshold: Optional[float] = Field(default=0.30, ge=0.05, le=0.95, description="Sharpness mask threshold (fraction of max). Higher → stricter mask, fewer false positives but may miss low-contrast bottles.")
    bottle_min_height_ratio: Optional[float] = Field(default=0.20, ge=0.05, le=0.95, description="Bottle height ≥ this fraction of crop height to be considered valid")
    bottle_min_aspect: Optional[float] = Field(default=1.2, ge=0.5, le=5.0, description="Minimum h/w aspect ratio (bottles are taller than wide)")


class EdgeWalls(BaseModel):
    """Template inner/outer wall gaps (px outward from label edge), computed on
    the template image at EdgeSetupModal save time and reused at inference."""
    inner_L: float = 0.0
    inner_R: float = 0.0
    outer_L: float = 0.0
    outer_R: float = 0.0
    plastic_L: float = 0.0
    plastic_R: float = 0.0


class EdgeConfig(BaseModel):
    """Per-template edge-detection tuning. Only used when the recipe's
    product_detection_method == 'yolo_segment' (image-proc bottle-wall detection).
    Field names mirror EdgeParams in ai_services .../image_proc_detector.py."""
    outer_search_max: float = Field(default=150.0, ge=10.0, le=600.0)
    inner_search_max: float = Field(default=80.0, ge=0.0, le=400.0)
    edge_margin: float = Field(default=2.0, ge=0.0, le=50.0)
    y_extension: float = Field(default=0.2, ge=0.0, le=1.0)
    fill_keep_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    peak_height: float = Field(default=0.05, ge=0.0, le=1.0)
    peak_prom: float = Field(default=0.02, ge=0.0, le=1.0)
    peak_dist: int = Field(default=4, ge=1, le=50)
    strong_thr: float = Field(default=0.15, ge=0.0, le=1.0)
    outer_min_hratio: float = Field(default=0.55, ge=0.0, le=1.0)
    inner_min_hratio: float = Field(default=0.20, ge=0.0, le=1.0)
    inner_tol_px: int = Field(default=12, ge=0, le=100)
    specular_thr: int = Field(default=230, ge=0, le=255)
    detect_mode: str = Field(default="gradient", description="'gradient' (|Scharr| edge) | 'brightness' (signed Scharr edge of a chosen polarity)")
    edge_polarity: str = Field(default="light_to_dark", description="Brightness-mode edge direction when scanning outward: 'light_to_dark' (bright rim→dark bg, default) | 'dark_to_light' (inverted contrast). Ignored by gradient mode.")
    find_by: str = Field(default="farthest", description="Which qualifying peak becomes the OUTER wall: 'farthest' (outermost, default) | 'nearest' (closest to label) | 'strongest' (highest profile)")
    edge_width: float = Field(default=3.0, ge=1.0, le=15.0, description="Expected edge transition width in px; profile smoothing sigma = edge_width/2")
    template_walls: Optional[EdgeWalls] = Field(default=None, description="Walls computed on the template image at setup; used directly at inference")


class AnomalyConfig(BaseModel):
    """Per-template anomaly-detection model binding. When enabled, this
    replaces WrinkledSegmenterTRT for this template's label-defect check
    (see ai_services/.../verification/anomaly_inference.py) — same 'label'
    crop, different model. `enabled` is a separate flag from having a model
    selected so an operator can roll back to the legacy wrinkle check
    without losing their anomaly project/model selection.

    `onnx_path` and `image_size` are captured by the recipe UI at selection
    time (read from anomaly_service's model record) and stored directly on
    the recipe — ai_services reads this path straight off disk and never
    needs to call anomaly_service at inference time."""
    enabled: bool = Field(default=False, description="If False, falls back to WrinkledSegmenterTRT even if a model is selected below")
    anomaly_project_id: Optional[str] = Field(default=None, description="anomaly_service project id")
    anomaly_model_id: Optional[str] = Field(default=None, description="anomaly_service model id (specific trained version, not 'latest')")
    onnx_path: Optional[str] = Field(default=None, description="Absolute path to the exported ONNX file, resolved at selection time")
    image_size: int = Field(default=256, ge=64, le=512, description="Must match the size the model was exported with")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="pred_score >= threshold => abnormal/FAIL")


class TemplateImage(BaseModel):
    """Template image with annotations"""
    name: str = Field(..., description="Template name")
    image_url: str = Field(..., description="URL to template image stored on server")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    annotations: List[TemplateAnnotation] = Field(default_factory=list, description="List of annotations")
    center_offset_threshold_left: float = Field(default=50.0, ge=0.0, le=500.0, description="Left alignment threshold (value in pixels OR %, depending on center_offset_unit)")
    center_offset_threshold_right: float = Field(default=50.0, ge=0.0, le=500.0, description="Right alignment threshold (value in pixels OR %, depending on center_offset_unit)")
    center_offset_threshold: float = Field(default=50.0, ge=0.0, le=500.0, description="Center alignment threshold (legacy, pixels)")
    center_offset_unit: Optional[str] = Field(default="px", description="Unit for center_offset_threshold_left/right: 'px' (pixels) or 'pct' (percent of reference width)")
    wrinkle_area: Optional[float] = Field(default=2000.0, ge=0.0, description="Total wrinkle area threshold in pixels — sum of valid regions ≥ this value → FAIL")
    wrinkle_min_area: Optional[float] = Field(default=0.0, ge=0.0, description="Per-region min area filter — regions smaller than this are ignored (0 = no filter)")
    wrinkle_max_area: Optional[float] = Field(default=0.0, ge=0.0, description="Per-region critical area — any region ≥ this value triggers FAIL immediately (0 = disabled)")
    color_config: Optional[ColorConfig] = Field(default=None, description="HSV color check config (only used when function_type=Check_Color and template has 'product' annotation)")
    edge_config: Optional[EdgeConfig] = Field(default=None, description="Per-bottle edge-detection tuning (only used when product_detection_method='yolo_segment')")
    anomaly_config: Optional[AnomalyConfig] = Field(default=None, description="Anomaly-detection model binding (replaces wrinkle check when enabled)")

class CameraTemplates(BaseModel):
    """Templates configuration for a camera"""
    camera_id: str = Field(..., description="Camera identifier")
    function_type: Optional[str] = Field(default="OCR", description="Function type: OCR, Check_Type_Product, Check_Color, Check_Defect, Check_Position, Barcode_Detection, DateCode_Detection")
    templates: List[TemplateImage] = Field(default_factory=list, description="List of template images")


class TriggerConfiguration(BaseModel):
    """Camera trigger configuration"""
    trigger_selector: str = Field(default="FrameStart", description="Trigger selector (FrameStart, ExposureStart, FrameBurstStart)")
    trigger_activation: str = Field(default="RisingEdge", description="Trigger edge activation (RisingEdge, FallingEdge, AnyEdge)")
    di_number: int = Field(default=0, ge=0, le=3, description="Digital Input number for software trigger (0-3)")
    trigger_source: str = Field(default="Line0", description="Trigger source for hardware trigger (Line0, Line1, Line2, Line3)")


class CameraConfiguration(BaseModel):
    """Configuration for a single camera in a recipe"""
    camera_id: str = Field(..., description="Camera identifier")
    model_name: str = Field(..., description="Camera model name")
    serial_number: str = Field(..., description="Camera serial number")
    location: Optional[str] = Field(None, description="Camera location")

    # Camera-specific settings for this recipe
    exposure_time: float = Field(..., description="Exposure time in milliseconds")
    delay_trigger: float = Field(..., description="Delay trigger in milliseconds (sensor to first frame)")
    delay_interval: float = Field(default=500.0, description="Delay between frames in milliseconds (for multi-template)")
    gain: float = Field(default=1.0, description="Camera gain")

    # Trigger mode and configuration
    trigger_mode: str = Field(default="continuous", description="Trigger mode: continuous, software_trigger, hardware_trigger")
    trigger_config: TriggerConfiguration = Field(default_factory=TriggerConfiguration)

    # Pixel format
    pixel_format: str = Field(default="Mono8", description="Pixel format (Mono8, Mono12, RGB8, YUV422, etc.)")


class CameraSettings(BaseModel):
    """Camera configuration settings (deprecated, kept for backward compatibility)"""
    exposure_time: float = Field(..., description="Exposure time in milliseconds")
    delay_trigger: float = Field(..., description="Delay trigger in milliseconds")
    delay_interval: Optional[float] = Field(default=500.0, description="Delay between frames in milliseconds (for multi-template)")
    gain: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None


class ModelThresholds(BaseModel):
    """Model thresholds for OCR processing"""
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    recognition_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    matching_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Template matching similarity threshold")
    min_text_size: Optional[int] = None
    max_text_size: Optional[int] = None


class RecipeBase(BaseModel):
    """Base recipe schema"""
    name: str = Field(..., min_length=1, max_length=100, description="Recipe name")
    product_code: str = Field(..., min_length=1, max_length=50, description="Product code")
    description: Optional[str] = Field(None, max_length=500)
    delay_reject: Optional[float] = Field(default=100.0, description="Delay reject time in milliseconds")
    reject_pulse: Optional[float] = Field(default=50.0, description="Reject pulse duration in milliseconds")
    reject_method: Optional[str] = Field(default='DIO_OUT', description="Reject output method: PLC or DIO_OUT")
    do_reject_number: Optional[int] = Field(default=2, ge=0, le=3, description="Digital Output number for reject (0-3)")
    do_alarm_number: Optional[int] = Field(default=0, ge=0, le=4, description="Digital Output number for alarm (0-4)")
    allow_late_reject: Optional[bool] = Field(default=False, description="Allow reject to fire immediately when inference_time > delay_reject (use for carton/alarm-only systems where late reject is harmless)")
    normal_pulse_ms: Optional[float] = Field(default=0.0, ge=0.0, le=999999.0, description="Expected normal DI pulse width in ms (used for stuck bottle detection)")

    # Reject scheduling mode
    reject_mode: Optional[str] = Field(default='time_based', description="time_based: AI canh thời gian (delay_reject + reject_pulse). sensor_based: PLC tự bắn pulse khi sensor trước rejector kích hoạt; AI chỉ gửi verdict 0/1 + handshake.")
    # Sensor-based PLC handshake config (used when reject_mode='sensor_based')
    # Each Modbus address has a "prefix" (cosmetic — picks how the field is labelled
    # to match the PLC ladder's device naming, e.g. Delta D/M/Y/HR/T/C/SD/S/SM/L).
    # At the Modbus TCP layer only the integer address + register-vs-coil function
    # matters; prefix is documentation only.
    plc_verdict_register: Optional[int] = Field(default=0, ge=0, le=9999, description="Modbus register address where AI writes verdict (0=PASS, 1=FAIL)")
    plc_verdict_prefix: Optional[str] = Field(default='D', max_length=4, description="Vendor-side device prefix for verdict register (D/HR/T/C/SD/custom). Cosmetic.")
    plc_ready_coil: Optional[int] = Field(default=0, ge=0, le=2047, description="Modbus coil address AI sets to signal 'verdict ready'")
    plc_ready_prefix: Optional[str] = Field(default='M', max_length=4, description="Vendor-side device prefix for ready coil (M/Y/S/SM/L/custom). Cosmetic.")
    plc_ack_coil: Optional[int] = Field(default=1, ge=0, le=2047, description="Modbus coil address PLC sets to acknowledge verdict consumed")
    plc_ack_prefix: Optional[str] = Field(default='M', max_length=4, description="Vendor-side device prefix for ack coil (M/Y/S/SM/L/custom). Cosmetic.")
    plc_pulse_register: Optional[int] = Field(default=10, ge=0, le=9999, description="Modbus register address where AI writes pulse_width (ms) once at recipe load")
    plc_pulse_prefix: Optional[str] = Field(default='D', max_length=4, description="Vendor-side device prefix for pulse register (D/HR/T/C/SD/custom). Cosmetic.")
    plc_ack_timeout_ms: Optional[int] = Field(default=200, ge=10, le=5000, description="Max time (ms) AI waits for PLC ack after writing verdict")

    # Multiple cameras support (new approach)
    cameras: List[CameraConfiguration] = Field(default_factory=list, description="Camera configurations for this recipe")

    # Camera templates (new multi-template support)
    camera_templates: List[CameraTemplates] = Field(default_factory=list, description="Template configurations for each camera")

    # Keep old camera_settings for backward compatibility (make optional)
    camera_settings: Optional[CameraSettings] = Field(default=None, description="Legacy camera settings (deprecated)")

    model_thresholds: ModelThresholds
    template_config: Optional[Dict[str, Any]] = Field(None, description="Legacy template matching configuration (deprecated)")
    roi_config: Optional[Dict[str, Any]] = Field(None, description="Region of interest configuration")
    is_active: bool = True

    # OCR / ML model selection
    ocr_model_type: Optional[str] = Field(default=None, description="OCR model type: SMTR, SVTRV2_CTC, OPENOCR_REPSVTR, PADDLEV5")
    ml_project_id: Optional[str] = Field(default=None, description="ML Training project ID for quality inspection")
    ml_model_id: Optional[str] = Field(default=None, description="Trained ML model ID within the ML project")
    defect_model: Optional[str] = Field(default="arcface", description="Embedding model used for defect detection: arcface | supcon")
    classifier_backend: Optional[str] = Field(default="embedding", description="Active classifier method: 'embedding' (defect_model cosine) | 'ml' (trained ML model)")
    cv_method: Optional[str] = Field(default="v4", description="CV pipeline variant when classifier_backend='embedding': 'legacy' | 'v4' | 'shape_v7'")
    template_bank_enabled: Optional[bool] = Field(default=False, description="Enable adaptive template bank for embedding-mode char verification")
    template_bank_size: Optional[int] = Field(default=10, ge=1, le=50, description="Max dynamic templates per (camera, annotation)")
    char_denoise_enabled: Optional[bool] = Field(default=False, description="Largest-CC noise filter before centroid alignment in char verification")
    product_detection_method: Optional[str] = Field(default="yolo_obb", description="Method to detect bottle edges: 'yolo_obb' (YOLO OBB model) | 'yolo_segment' (image processing — Sobel/edge detection)")
    cap_rotation_method: Optional[str] = Field(default="yolo_obb", description="Method to rotate caps so date-code text faces upright: 'yolo_obb' (trained YOLO OBB model) | 'yolo_segment' (pure CV — HoughCircles + projection profile + shape match)")
    cap_crop_method: Optional[str] = Field(default="none", description="Detect+crop bottle cap for matching: 'none' (use user-drawn crop_area) | 'yolo_obb' (trained YOLO OBB model) | 'yolo_segment' (HoughCircles)")
    crop_match_method: Optional[str] = Field(default="superpoint", description="Method for matching template ↔ target crop: 'superpoint' (TRT deep model, ~95ms) | 'shape_outline' (ECC on Sobel gradient magnitude, ~30ms — best for cap-OCR with cap_rotation+cap_crop active)")
    dual_rotation_check: Optional[bool] = Field(default=False, description="Try BOTH cap rotation candidates (angle + flipped180) and pick the one with higher match confidence. Only applies to Check_Color cap-OCR cameras. Doubles match cost but fixes ambiguous flip detection.")
    product_box_wall_type: Optional[str] = Field(default="outer", description="When product_detection_method='yolo_segment', which wall to use for product box corners: 'outer' (bottle silhouette) | 'inner' (closer to label)")

    save_pass_images: Optional[bool] = Field(default=True, description="Save PASS-frame images to disk (ring buffer of 200 most-recent per camera). Used to re-test missed defects.")

    # Wrinkle segmentation model confidence threshold (recipe-level, applies to all cameras)
    wrinkle_conf: Optional[float] = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold for wrinkle segmentation model (0.0 - 1.0)")
    wrinkle_show_when_pass: Optional[bool] = Field(default=True, description="Draw detected wrinkle regions on _viz.jpg even when frame passes (for debugging/visualization)")
    mask_overlap_threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0, description="Wrinkle region with >= this fraction of pixels inside a 'mask' annotation will be excluded (0.0 - 1.0)")

    # SuperPoint matching confidence threshold (recipe-level): skip OCR/verify when inlier_ratio falls below this
    matching_conf: Optional[float] = Field(default=0.20, ge=0.0, le=1.0, description="SuperPoint matching confidence threshold (inliers/total_matches). Below this, verify is skipped and frame is marked FAIL early.")

    # Horizontal erosion preprocessing before SuperPoint matching
    match_erosion_enabled: Optional[bool] = Field(default=False, description="Apply horizontal erosion to crop area before SuperPoint matching to suppress variable text (date codes)")
    match_erosion_kernel_w: Optional[int] = Field(default=80, ge=1, le=300, description="Horizontal erosion kernel width in pixels (larger = heavier erosion)")
    match_erosion_kernel_h: Optional[int] = Field(default=1, ge=1, le=50, description="Erosion kernel height in pixels (1 = pure horizontal, 15 = fills letter gaps)")
    match_erosion_iterations: Optional[int] = Field(default=1, ge=1, le=5, description="Number of erosion iterations (more = stronger effect)")


class RecipeCreate(RecipeBase):
    """Schema for creating a new recipe"""
    pass


class RecipeUpdate(BaseModel):
    """Schema for updating a recipe"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    product_code: Optional[str] = Field(None, min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    delay_reject: Optional[float] = None
    reject_pulse: Optional[float] = None
    reject_method: Optional[str] = None
    do_reject_number: Optional[int] = Field(None, ge=0, le=3)
    do_alarm_number: Optional[int] = Field(None, ge=0, le=4)
    allow_late_reject: Optional[bool] = None
    normal_pulse_ms: Optional[float] = Field(None, ge=0.0, le=999999.0)
    reject_mode: Optional[str] = None
    plc_verdict_register: Optional[int] = Field(None, ge=0, le=9999)
    plc_verdict_prefix: Optional[str] = Field(None, max_length=4)
    plc_ready_coil: Optional[int] = Field(None, ge=0, le=2047)
    plc_ready_prefix: Optional[str] = Field(None, max_length=4)
    plc_ack_coil: Optional[int] = Field(None, ge=0, le=2047)
    plc_ack_prefix: Optional[str] = Field(None, max_length=4)
    plc_pulse_register: Optional[int] = Field(None, ge=0, le=9999)
    plc_pulse_prefix: Optional[str] = Field(None, max_length=4)
    plc_ack_timeout_ms: Optional[int] = Field(None, ge=10, le=5000)
    cameras: Optional[List[CameraConfiguration]] = None
    camera_templates: Optional[List[CameraTemplates]] = None
    camera_settings: Optional[CameraSettings] = None
    model_thresholds: Optional[ModelThresholds] = None
    template_config: Optional[Dict[str, Any]] = None
    roi_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    ocr_model_type: Optional[str] = None
    ml_project_id: Optional[str] = None
    ml_model_id: Optional[str] = None
    defect_model: Optional[str] = None
    classifier_backend: Optional[str] = None
    cv_method: Optional[str] = None
    template_bank_enabled: Optional[bool] = None
    template_bank_size: Optional[int] = Field(None, ge=1, le=50)
    char_denoise_enabled: Optional[bool] = None
    product_detection_method: Optional[str] = None
    product_box_wall_type: Optional[str] = None
    save_pass_images: Optional[bool] = None
    cap_rotation_method: Optional[str] = None
    cap_crop_method: Optional[str] = None
    crop_match_method: Optional[str] = None
    dual_rotation_check: Optional[bool] = None
    wrinkle_conf: Optional[float] = Field(None, ge=0.0, le=1.0)
    wrinkle_show_when_pass: Optional[bool] = None
    matching_conf: Optional[float] = Field(None, ge=0.0, le=1.0)
    mask_overlap_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    match_erosion_enabled: Optional[bool] = None
    match_erosion_kernel_w: Optional[int] = Field(None, ge=1, le=300)
    match_erosion_kernel_h: Optional[int] = Field(None, ge=1, le=50)
    match_erosion_iterations: Optional[int] = Field(None, ge=1, le=5)


class RecipeInDB(RecipeBase):
    """Schema for recipe stored in database"""
    id: str = Field(alias="_id")
    created_by: str = Field(..., description="User ID who created this recipe")
    updated_by: str = Field(..., description="User ID who last updated this recipe")
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True


class RecipeResponse(RecipeBase):
    """Schema for recipe response"""
    id: str
    created_by: str
    updated_by: str
    created_by_name: Optional[str] = None  # Full name of creator
    updated_by_name: Optional[str] = None  # Full name of updater
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True
