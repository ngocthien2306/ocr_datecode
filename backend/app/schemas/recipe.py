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
    cv_method: Optional[str] = Field(default="legacy", description="CV pipeline variant when classifier_backend='embedding': 'legacy' | 'v4' | 'shape_v7'")
    template_bank_enabled: Optional[bool] = Field(default=False, description="Enable adaptive template bank for embedding-mode char verification")
    template_bank_size: Optional[int] = Field(default=10, ge=1, le=50, description="Max dynamic templates per (camera, annotation)")
    char_denoise_enabled: Optional[bool] = Field(default=False, description="Largest-CC noise filter before centroid alignment in char verification")
    product_detection_method: Optional[str] = Field(default="yolo_obb", description="Method to detect bottle edges: 'yolo_obb' (YOLO OBB model) | 'yolo_segment' (image processing — Sobel/edge detection)")
    product_box_wall_type: Optional[str] = Field(default="outer", description="When product_detection_method='yolo_segment', which wall to use for product box corners: 'outer' (bottle silhouette) | 'inner' (closer to label)")

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
