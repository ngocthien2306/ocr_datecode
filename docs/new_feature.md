/Users/ngocthien.ai/Source/Projects/ocr_datecode/public/images đây là folder mà máy sẽ lúc nào cũng lưu 100 ảnh mới nhất vào đây

tôi đang muốn dev 1 tính năng mới là training ML model cho việc phân loại kí tự tốt hay NG

Thiết kế: 
bên FE tạo 1 page mới /Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components ở đây
page mới này cho phép create project training (nên tạo database mới) để lưu các project 
mỗi project sẽ cho upload folder hình ảnh, khi load ảnh lên thì cho phép select all hoặc select ảnh tuỳ thích, khi bấm nút lưu thì nó sẽ copy ảnh vào folder project. Ảnh trong mỗi rồi project có thể xoá hoặc thêm tuỳ thích

Tiếp theo là tới label ảnh, sẽ có 1 list ảnh bên tay trái, click vào thì ảnh hiện chính giữa, bên trái là phần hiện annotation.
Sau khi vẽ rectangle quanh ảnh đó thì có thể chọn label (OK, NG) cho ảnh đó, và có thể edit được 
Hãy tham khảo code này để biết làm sao load ảnh lên và vẽ label /Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/recipe/TemplateEditorRefactored.tsx

Sau khi label xong thì có thể qua tab train để training model. Tab train có phần hiện các vùng ký tự đã crop từ tab trước, hiện badge là ký tự OK/NG ở label tab trước. Thường ko có quá nhiều ký tự lỗi nên tham khảo 2 code này, nó agument để tạo ra các mẫu lỗi, có option cho người dùng chọn là có augemnet hay ko, nếu có thì cho select chọn số lượng x2,x3,x4,x5 ...
/Users/ngocthien.ai/Source/Projects/ocr_datecode/ui_compare.py
/Users/ngocthien.ai/Source/Projects/ocr_datecode/ml_classifier.py

Sau khi check xong thì ta training model ở đây thiết kế làm sao cho người dùng chọn thuật toán để train thường là SVM, Deep (MLP bên trong), RF ... 
Sau khi train xong thì show kết quả đánh giá, và cho chọn 1 vài ảnh ký tự đã crop để test 

Với thiết kế trên thì ta cần design API, hãy giúp tôi phân tích thêm các API cần có và check sơ cấu trúc source /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/app để code theo cho đúng pattern 


Với phần UI hãy tham khảm các CSS này trong page này có dùng /Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/inference/InferenceRealtime.tsx
inference-realtime-overlay
inference-realtime-container
inference-realtime

để có thể cho màng hình page show full chứ không phải dạng popup 

Hãy phân tích trước cho tôi và góp ý đề xuất hoặc cải tiến và đặt câu hỏi nếu chưa có gì rõ ràng 


===============

Sau khi training xong tôi muốn apply model này bằng cách vào recipe /Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/recipe/Receipts.tsx

vào tab tab model bên /Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/recipe/RecipeFormModal.tsx

Ở đây cho phép select chọn project ML training sau đó chọn model đã train, tạm thời hoàn thành bước này sau đó tôi sẽ yêu cầu thêm, tạo thêm 1 select box cho các option này 
# OCRModelType.SMTR          (large-x) 
# OCRModelType.SVTRV2_CTC    (large)
# OCRModelType.OPENOCR_REPSVTR (medium)
# OCRModelType.PADDLEV5        (small)

có những cái cần lưu ý các api cho recipe /Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/app/api/endpoints/recipes.py

/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/app/schemas/recipe.py
/Users/ngocthien.ai/Source/Projects/ocr_datecode/backend/app/models/recipe.py

async def create_recipe()
async def update_recipe()

async def clone_recipe() vui lòng kiểm tra lại để clone hết tất cả các field trong recipe cũ 

async def list_recipes()
async def search_recipes() cần trả ra các field để bên FE maping cho đầy đủ nha, tôi hay bị tình huống là lưu xuống BE được rồi mà FE ko có show lên field mới vừa update mà show default 

những điểm cần chú ý bên FE là /Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/recipe/Receipts.tsx

/Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/recipe/RecipeFormModal.tsx

Kiểm tra xem các model mapping (DTO) để để chắc rằng nó mapping data lên đúng khi tôi bấm và update vì nó lấy data hiện lên các form chắc rằng data nó lấy đúng 

=========

Tôi có tính năng là load recipe để AI Service chạy recipe này, khi load nó sẽ gửi API cho BE, BE gửi socket qua AI service, hiện tại tôi cần làm logic mapping model OCR trước, còn ML trainign chỉ gửi thông tin qua thôi logic tôi sẽ làm sau 

@router.get("/loads/latest") api giúp khi mà máy bị tắt và mở lên lên bên AI service sẽ call api để lấy ra recipe đã load hiện tại 

còn API này sẽ load recipe trực tiếp @router.post("/{recipe_id}/load", response_model=ReceiptLoadResponse)

có 1 api nữa là @router.post("/{recipe_id}/update-realtime")
phải chắc rằng có cũng gửi thông tin của OCR model và ML project cho tôi 

===========

/Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/ml-training/LabelTab.tsx

bên này có chức năng là vẽ 1 vùng sau đó dùng auto segment 

bên recipe template này tôi cũng muốn làm tương tự 
/Users/ngocthien.ai/Source/Projects/ocr_datecode/frontend-ts/src/components/recipe/TemplateEditorRefactored.tsx

đầu tiên hãy đọc hiểu các annotation, bây giờ khi vẽ 1 vùng. Khi mà tôi chọn type là text hoặc datecode thì mới hiện option auto segment. 
Ví dụ tôi chọn text segment xong có 12 vùng -> tạo 12 annotation type là text, còn text kí tự tôi sẽ tự nhập. Tương tự với datecode  





PLCController not available, PLC mode disabled
PLCController not available, PLC mode disabled
2026-04-24 11:00:51,869 - __main__ - INFO - ================================================================================
2026-04-24 11:00:51,870 - __main__ - INFO - Camera Management Service v1.0
2026-04-24 11:00:51,870 - __main__ - INFO - ================================================================================
2026-04-24 11:00:51,870 - camera_management.reject_scheduler - INFO - RejectScheduler initialized
2026-04-24 11:00:51,871 - camera_management.matchers.factory - INFO - MatcherFactory using optimized shared TensorRT engine
2026-04-24 11:00:51,871 - camera_management.inference_handler - INFO - InferenceHandler initialized (models will be lazy-loaded on worker thread)
2026-04-24 11:00:51,871 - camera_management.camera_manager - INFO - CameraManager initialized with handlers
2026-04-24 11:00:51,871 - __main__ - INFO - CameraManagementService initialized
2026-04-24 11:00:51,871 - __main__ - INFO - API Base: http://localhost:8000
2026-04-24 11:00:51,871 - __main__ - INFO - WebSocket URL: ws://localhost:8000/ws/camera-management
2026-04-24 11:00:51,871 - __main__ - INFO - Starting CameraManagementService...
2026-04-24 11:00:51,871 - camera_management.websocket_client - INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
2026-04-24 11:00:51,891 - camera_management.websocket_client - INFO - WebSocket connected successfully
2026-04-24 11:00:51,891 - camera_management.websocket_client - INFO - WebSocket client started
2026-04-24 11:00:51,892 - __main__ - INFO - WebSocket connected, checking for running recipes...
2026-04-24 11:00:51,892 - __main__ - INFO - Checking for running recipes...
2026-04-24 11:00:51,916 - __main__ - INFO - Found running recipe: PL_KS_WHOLE BLACK PEPPER CN (Copy)
2026-04-24 11:00:51,916 - __main__ - INFO - 📋 Latest load - cameras config:
2026-04-24 11:00:51,916 - __main__ - INFO -   Camera 1:
2026-04-24 11:00:51,916 - __main__ - INFO -     - camera_id: CameraCarton
2026-04-24 11:00:51,916 - __main__ - INFO -     - serial_number: 40767173
2026-04-24 11:00:51,916 - __main__ - INFO -     - function_type: Check_Type_Product ⭐
2026-04-24 11:00:51,916 - __main__ - INFO -     - trigger_mode: software_trigger
2026-04-24 11:00:51,917 - camera_management.camera - INFO - Camera instance created: 40767173, pixel_format=BGR8
2026-04-24 11:00:52,508 - camera_management.camera - INFO - [40767173] PixelFormat set: BGR8
2026-04-24 11:00:52,513 - camera_management.camera - INFO - [40767173] Exposure: 500.0 μs
2026-04-24 11:00:52,517 - camera_management.camera - INFO - [40767173] Gain: 1.0000470811938786
2026-04-24 11:00:52,522 - camera_management.camera - INFO - Ring buffer created: camera_40767173, size=32.96MB, slots=5, slot_size=6751.00KB
2026-04-24 11:00:52,522 - camera_management.camera - INFO - Camera connected: a2A1920-51gcBAS (SN: 40767173), resolution: 1920x1200, ring buffer: 32.96MB (5 frames)
2026-04-24 11:00:52,523 - camera_management.camera - INFO - [40767173] Mode changed: idle → continuous
2026-04-24 11:00:52,524 - camera_management.camera - INFO - [40767173] Starting camera loop (mode: continuous)
2026-04-24 11:00:52,524 - camera_management.camera - INFO - [40767173] Camera loop thread started
2026-04-24 11:00:52,524 - camera_management.camera_manager - INFO - Camera 40767173 added and connected
2026-04-24 11:00:52,524 - __main__ - INFO - Connected camera 40767173 with BGR8
2026-04-24 11:00:52,524 - camera_management.camera_manager - INFO - 📥 Loading recipe: PL_KS_WHOLE BLACK PEPPER CN (Copy) (ID: 69eadff5cba186658062a667)
2026-04-24 11:00:52,524 - camera_management.camera_manager - INFO - 📋 Recipe cameras config:
2026-04-24 11:00:52,524 - camera_management.camera_manager - INFO -   Camera 1:
2026-04-24 11:00:52,524 - camera_management.camera_manager - INFO -     - camera_id: CameraCarton
2026-04-24 11:00:52,524 - camera_management.camera_manager - INFO -     - serial_number: 40767173
2026-04-24 11:00:52,525 - camera_management.camera_manager - INFO -     - function_type: Check_Type_Product ⭐
2026-04-24 11:00:52,525 - camera_management.camera_manager - INFO -     - trigger_mode: software_trigger
2026-04-24 11:00:52,525 - camera_management.camera - INFO - [40767173] Loaded thresholds: matching=0.85, recognition=0.5
2026-04-24 11:00:52,525 - camera_management.camera - INFO - [40767173] OCR model: default, ML project: none, ML model: none
2026-04-24 11:00:52,525 - camera_management.camera - INFO - [40767173] Camera config from recipe:
2026-04-24 11:00:52,525 - camera_management.camera - INFO -   - pixel_format: BGR8
2026-04-24 11:00:52,525 - camera_management.camera - INFO -   - function_type: Check_Type_Product
2026-04-24 11:00:52,525 - camera_management.camera - INFO -   - exposure_time: 500.0
2026-04-24 11:00:52,525 - camera_management.camera - INFO -   - delay_trigger: 500.0
2026-04-24 11:00:52,525 - camera_management.camera - INFO -   - delay_reject: 1500.0ms (recipe level)
2026-04-24 11:00:52,525 - camera_management.camera - INFO -   - do_reject_number: DO3 (recipe level)
2026-04-24 11:00:52,525 - camera_management.camera - INFO -   - do_alarm_number: DO0 (recipe level)
2026-04-24 11:00:52,577 - camera_management.camera - INFO - [40767173] Exposure time set to 500.0μs
2026-04-24 11:00:52,579 - camera_management.camera - INFO - [40767173] Gain set to 10.0
2026-04-24 11:00:52,579 - camera_management.camera - WARNING - [40767173] Pixel format change requires reconnect (camera is grabbing)
2026-04-24 11:00:52,683 - camera_management.camera - INFO - [40767173] PixelFormat set: BGR8
2026-04-24 11:00:52,688 - camera_management.camera - INFO - [40767173] Exposure: 500.0 μs
2026-04-24 11:00:52,690 - camera_management.camera - INFO - [40767173] Gain: 9.999998795594486
2026-04-24 11:00:52,744 - camera_management.camera - INFO - [40767173] Restarted grabbing after pixel format change
2026-04-24 11:00:52,745 - camera_management.camera - INFO - [40767173] Settings updated
2026-04-24 11:00:52,745 - camera_management.camera - INFO -   - Template 0, Annotation 2 (text): 'BEST BEFORE'
2026-04-24 11:00:52,745 - camera_management.camera - INFO -   - Template 0, Annotation 3 (text): '04/2028'
2026-04-24 11:00:52,746 - camera_management.camera - INFO -   - Template 1, Annotation 2 (text): '297538'
2026-04-24 11:00:52,746 - camera_management.camera - INFO -   - Template 1, Annotation 3 (text): 'SAIGON CINNAMON'
2026-04-24 11:00:52,746 - camera_management.camera - INFO -   - Template 1, Annotation 4 (text): 'UK'
2026-04-24 11:00:52,746 - camera_management.camera - INFO - [40767173] Loaded expected_texts for 2 templates
2026-04-24 11:00:52,746 - camera_management.camera - INFO - [40767173] Recipe loaded: PL_KS_WHOLE BLACK PEPPER CN (Copy), templates: 2, trigger_mode: software_trigger
2026-04-24 11:00:52,853 - camera_management.camera - WARNING - [40767173] TriggerActivation not writable (AccessMode=1, model: a2A1920-51gcBAS)
2026-04-24 11:00:52,868 - camera_management.camera - INFO - [40767173] Camera grabbing started (OneByOne)
2026-04-24 11:00:52,869 - camera_management.camera - INFO - [40767173] Mode changed: continuous → software_trigger
2026-04-24 11:00:52,869 - camera_management.camera - INFO - [40767173] Software trigger mode configured
2026-04-24 11:00:52,869 - camera_management.camera_manager - INFO - Recipe loaded to camera 40767173
2026-04-24 11:00:52,869 - camera_management.camera_manager - INFO - Recipe 'PL_KS_WHOLE BLACK PEPPER CN (Copy)' loaded to 1/1 cameras
2026-04-24 11:00:52,869 - camera_management.camera_manager - INFO - Initializing inference matchers for 1 cameras
2026-04-24 11:00:52,869 - camera_management.inference_handler - INFO - Submitting matcher initialization to worker thread...
2026-04-24 11:00:52,870 - camera_management.inference_handler - INFO - 🔧 Initializing matchers on InferenceWorker_0...
2026-04-24 11:00:52,870 - camera_management.inference_handler - INFO - 🔧 Lazy-initializing OCR/YOLO models on InferenceWorker_0...
2026-04-24 11:00:52,925 - camera_management.inference_handler - INFO - ✅ CUDA context created on InferenceWorker_0 (handle: 187651334530560)
2026-04-24 11:00:52,966 - camera_management.ocr.factory - INFO - OCR availability: {'tensorrt_openocr': True, 'onnx_openocr': True, 'tensorrt_paddlev5': True, 'onnx_paddlev5': True, 'tensorrt_svtrv2': False, 'onnx_svtrv2': False, 'tensorrt_smtr': False, 'onnx_smtr': False}
[04/24/2026-11:00:52] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
2026-04-24 11:00:53,132 - camera_management.text_recognizer_openocr_trt - INFO - Dynamic batch range: 1 - 16
2026-04-24 11:00:53,134 - camera_management.text_recognizer_openocr_trt - INFO - ✅ Pre-allocated CUDA buffers (persistent, like YOLO OBB):
2026-04-24 11:00:53,134 - camera_management.text_recognizer_openocr_trt - INFO -    Input:  2.81 MB
2026-04-24 11:00:53,134 - camera_management.text_recognizer_openocr_trt - INFO -    Output: 10.11 MB
2026-04-24 11:00:53,134 - camera_management.text_recognizer_openocr_trt - INFO -    Total:  12.92 MB
2026-04-24 11:00:53,135 - camera_management.text_recognizer_openocr_trt - INFO - ✅ TextRecognizerOpenOCRTRT initialized
2026-04-24 11:00:53,135 - camera_management.text_recognizer_openocr_trt - INFO -    TensorRT API: 10.x+
2026-04-24 11:00:53,135 - camera_management.text_recognizer_openocr_trt - INFO -    Engine: openocr_rec_model_batch.engine
2026-04-24 11:00:53,135 - camera_management.text_recognizer_openocr_trt - INFO -    Dictionary: 6625 characters
2026-04-24 11:00:53,135 - camera_management.text_recognizer_openocr_trt - INFO -    Batch range: 1-16
2026-04-24 11:00:53,135 - camera_management.ocr.factory - INFO - TensorRTOpenOCRBackend initialized: /home/suntech/Source/ocr_datecode/languages/english/openocr_rec_model_batch.engine (batch_size=4)
2026-04-24 11:00:53,135 - camera_management.ocr.factory - INFO - ✅ OCR backend (auto): tensorrt_openocr
2026-04-24 11:00:53,135 - camera_management.inference_handler - INFO - ✅ OCR backend initialized: tensorrt_openocr
2026-04-24 11:00:53,136 - camera_management.verification.ml_classifier - INFO - MLClassifierService initialized: base_dir=/home/suntech/Source/ocr_datecode/public/ml_projects, save_debug=True
2026-04-24 11:00:53,136 - camera_management.inference_handler - INFO - MLClassifierService initialized: base_dir=/home/suntech/Source/ocr_datecode/public/ml_projects
2026-04-24 11:00:53,136 - camera_management.inference_handler - INFO - TextVerificationService initialized with tensorrt_openocr backend
2026-04-24 11:00:53,136 - camera_management.inference_handler - INFO - TemplateVerificationService initialized
2026-04-24 11:00:53,136 - camera_management.verification.product_verifier - ERROR - Failed to load YOLO OBB model: [Errno 2] No such file or directory: '/home/suntech/Source/ocr_datecode/weights/best_bottle_m_320_new.engine'
[04/24/2026-11:00:53] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
2026-04-24 11:00:53,325 - camera_management.verification.wrinkle_segmenter - INFO - WrinkledSegmenterTRT loaded: /home/suntech/Source/ocr_datecode/weights/best_wrinkled_instance_segmentation_crop_bottle.engine img_size=320 max_batch=8 min_area=2000px
2026-04-24 11:00:53,326 - camera_management.inference_handler - INFO - ProductVerificationService initialized with YOLO OBB: /home/suntech/Source/ocr_datecode/weights/best_bottle_m_320_new.engine, debug_mode=never, check_misalignment=True, check_label_boundary=True
2026-04-24 11:00:53,326 - camera_management.preprocessing.obb_rotator - INFO - OBB rotation log: /home/suntech/Source/ocr_datecode/ai_services/logs/obb_rotation.log
2026-04-24 11:00:53,327 - camera_management.preprocessing.obb_rotator - ERROR - OBBRotationService: Failed to load engine: [Errno 2] No such file or directory: '/home/suntech/Source/ocr_datecode/weights/best_bottle_m.engine'
2026-04-24 11:00:53,327 - camera_management.inference_handler - INFO - OBBRotationService initialized: /home/suntech/Source/ocr_datecode/weights/best_bottle_m.engine
2026-04-24 11:00:53,327 - camera_management.inference_handler - INFO - OBBRotationService available: False
2026-04-24 11:00:53,327 - camera_management.inference_handler - INFO - ✅ OCR/YOLO models initialized on InferenceWorker_0
2026-04-24 11:00:53,327 - camera_management.inference_handler - INFO - [40767173] Single camera scenario: 2 templates -> creating 2 matchers
2026-04-24 11:00:53,360 - camera_management.matchers.annotation_parser - INFO - [40767173] Template 0: Found crop_area: (295, 559) -> (1437, 1067), size: 1142x508
2026-04-24 11:00:53,360 - camera_management.matchers.factory - INFO - [40767173] Template 0: Applying crop_area
2026-04-24 11:00:53,366 - camera_management.matchers.factory - INFO - [40767173] Template 0: Cropped template saved: 1142x508
2026-04-24 11:00:53,367 - camera_management.matchers.factory - INFO - [40767173] Template 0: Adjusted 3 bbox(es) to cropped coordinates
2026-04-24 11:00:53,376 - inference_engine_shared - INFO - ⏱️  Load template config: 8.5ms
[04/24/2026-11:00:53] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
2026-04-24 11:00:53,749 - inference_engine_shared - INFO - ⏱️  Load engine: 372.4ms
2026-04-24 11:00:53,749 - inference_engine_shared - INFO - 🔥 Warming up engine...
2026-04-24 11:00:53,903 - inference_engine_shared - INFO -    Warm-up done: 153.4ms
2026-04-24 11:00:53,903 - inference_engine_shared - INFO - ✅ SuperPointEngineTRT initialized (526.9ms)
2026-04-24 11:00:53,903 - inference_engine_shared - INFO -    Engine: pipeline_fp16_dynamic_315_560.engine
2026-04-24 11:00:53,903 - inference_engine_shared - INFO -    Input shape: (-1, 1, 315, 560)
2026-04-24 11:00:53,904 - inference_engine_shared - INFO - ⏱️  Get shared engine: 528.3ms
2026-04-24 11:00:53,905 - inference_engine_shared - INFO - ✅ SuperPointMatcherTRTOptimized initialized (537.2ms)
2026-04-24 11:00:53,905 - inference_engine_shared - INFO -    Template: 1142x508
2026-04-24 11:00:53,905 - inference_engine_shared - INFO -    Bboxes: template + 2 regions
2026-04-24 11:00:53,905 - inference_engine_shared - INFO -    Using shared engine: pipeline_fp16_dynamic_315_560.engine
2026-04-24 11:00:53,905 - camera_management.matchers.factory - INFO - [40767173] Matcher 0 created successfully
2026-04-24 11:00:53,905 - camera_management.matchers.factory - INFO - [40767173] Matcher 1/2 initialized
2026-04-24 11:00:53,935 - camera_management.matchers.annotation_parser - INFO - [40767173] Template 1: Found crop_area: (61, 511) -> (1426, 996), size: 1365x485
2026-04-24 11:00:53,935 - camera_management.matchers.factory - INFO - [40767173] Template 1: Applying crop_area
2026-04-24 11:00:53,943 - camera_management.matchers.factory - INFO - [40767173] Template 1: Cropped template saved: 1365x485
2026-04-24 11:00:53,943 - camera_management.matchers.factory - INFO - [40767173] Template 1: Adjusted 4 bbox(es) to cropped coordinates
2026-04-24 11:00:53,951 - inference_engine_shared - INFO - ✅ SuperPointMatcherTRTOptimized initialized (7.6ms)
2026-04-24 11:00:53,952 - inference_engine_shared - INFO -    Template: 1365x485
2026-04-24 11:00:53,952 - inference_engine_shared - INFO -    Bboxes: template + 3 regions
2026-04-24 11:00:53,952 - inference_engine_shared - INFO -    Using shared engine: pipeline_fp16_dynamic_315_560.engine
2026-04-24 11:00:53,952 - camera_management.matchers.factory - INFO - [40767173] Matcher 1 created successfully
2026-04-24 11:00:53,952 - camera_management.matchers.factory - INFO - [40767173] Matcher 2/2 initialized
2026-04-24 11:00:53,952 - camera_management.inference_handler - INFO - [40767173] 2/2 matchers initialized
2026-04-24 11:00:53,952 - camera_management.inference_handler - INFO - Initialized 2 matchers using SHARED TensorRT engine (1 engine, 2 template configs)
2026-04-24 11:00:53,952 - camera_management.inference_handler - INFO - ✅ Initialized 2 matchers on InferenceWorker_0
2026-04-24 11:00:53,952 - camera_management.inference_handler - INFO - ✅ Matcher initialization completed: 2 matchers
2026-04-24 11:00:53,953 - camera_management.camera_manager - INFO - ✅ 2 inference matchers initialized successfully
2026-04-24 11:00:53,953 - camera_management.camera_manager - INFO - normal_pulse_ms set to 100000.0 ms from recipe
2026-04-24 11:00:53,953 - camera_management.trigger_handler - INFO - DI0 → 1 camera(s): ['40767173']
2026-04-24 11:00:53,954 - camera_management.trigger_handler - INFO - Pulse width logging to: /home/suntech/Source/ocr_datecode/ai_services/logs/pulse_width.log
2026-04-24 11:00:53,955 - camera_management.trigger_handler - INFO - DI polling loop started
2026-04-24 11:00:53,955 - camera_management.trigger_handler - INFO - Trigger polling started (100Hz)
2026-04-24 11:00:53,956 - camera_management.trigger_handler - INFO - Statistics monitoring started (interval: 10s)
2026-04-24 11:00:53,956 - camera_management.camera_manager - INFO - Reject config: method=DIO_OUT, pulse=50.0ms
2026-04-24 11:00:53,956 - camera_management.trigger_handler - INFO - Statistics logging to: /home/suntech/Source/ocr_datecode/ai_services/camera_management/../logs/trigger_stats.log
2026-04-24 11:00:53,956 - camera_management.reject_scheduler - INFO - Reject config updated: method=DIO_OUT, pulse=50.0ms
2026-04-24 11:00:53,957 - camera_management.reject_scheduler - INFO - RejectScheduler thread started
2026-04-24 11:00:53,957 - camera_management.reject_scheduler - INFO - Reject monitoring started (interval: 10s)
2026-04-24 11:00:53,957 - camera_management.reject_scheduler - INFO - RejectScheduler started
2026-04-24 11:00:53,957 - __main__ - INFO - Recipe auto-loaded successfully: PL_KS_WHOLE BLACK PEPPER CN (Copy)
2026-04-24 11:00:53,958 - __main__ - INFO - CameraManagementService started successfully
2026-04-24 11:01:03,966 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=0/0 (0.0%), Inferences=0, CaptureFail=0, Active=0g/0t
2026-04-24 11:01:03,967 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 0, Rejected: 0 (PLC: 0, DIO: 0, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 0, Active: 0
2026-04-24 11:01:13,974 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=0/0 (0.0%), Inferences=0, CaptureFail=0, Active=0g/0t
2026-04-24 11:01:13,976 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 0, Rejected: 0 (PLC: 0, DIO: 0, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 0, Active: 0
2026-04-24 11:01:23,984 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=0/0 (0.0%), Inferences=0, CaptureFail=0, Active=0g/0t
2026-04-24 11:01:23,988 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 0, Rejected: 0 (PLC: 0, DIO: 0, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 0, Active: 0
2026-04-24 11:01:33,994 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=0/0 (0.0%), Inferences=0, CaptureFail=0, Active=0g/0t
2026-04-24 11:01:33,998 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 0, Rejected: 0 (PLC: 0, DIO: 0, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 0, Active: 0
2026-04-24 11:01:44,002 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=0/0 (0.0%), Inferences=0, CaptureFail=0, Active=0g/0t
2026-04-24 11:01:44,007 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 0, Rejected: 0 (PLC: 0, DIO: 0, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 0, Active: 0
2026-04-24 11:01:44,257 - camera_management.trigger_handler - INFO - Simulating rising_edge for 1 camera(s)
2026-04-24 11:01:44,258 - camera_management.trigger_handler - INFO - [Group #1] Creating capture group for 1 camera(s): [('40767173', 500.0)]
2026-04-24 11:01:44,258 - camera_management.trigger_handler - INFO - [Group #1] Scheduling Camera 40767173 capture in 0.50s
2026-04-24 11:01:44,759 - camera_management.trigger_handler - INFO - [Group #1] Camera 40767173 timer FIRED! Capturing now (Thread: CaptureTimer-1-40767173)
2026-04-24 11:01:44,760 - camera_management.camera - INFO - [40767173] Capturing 2 frames with delay_interval=1200.0ms between frames (REVERSED order)
2026-04-24 11:01:46,226 - camera_management.camera - INFO - [40767173] ✅ Captured 2 frames (reordered to match templates)
2026-04-24 11:01:46,226 - camera_management.trigger_handler - INFO - [Group #1] Camera 40767173 captured 2 frames. Progress: 1/1
2026-04-24 11:01:46,227 - camera_management.image_saver - INFO - [ImageSaver] Started. Save dir: /home/suntech/Source/ocr_datecode/public/images, max images: 100
2026-04-24 11:01:46,227 - camera_management.trigger_handler - INFO - [Group #1] ✅ All 1 cameras completed! Triggering batch inference.
2026-04-24 11:01:46,228 - __main__ - INFO - Frames captured (Group #1), submitting async inference for 1 camera(s)...
2026-04-24 11:01:46,228 - camera_management.inference_handler - INFO - [Job #1] Submitting inference for 1 camera(s) (queue depth: 0)
2026-04-24 11:01:46,229 - camera_management.inference_handler - INFO - [Job #1] Starting inference in InferenceWorker_0 (group_id: 1)
2026-04-24 11:01:46,230 - camera_management.inference_handler - INFO - [Job #1] Using SingleCameraPipeline
2026-04-24 11:01:46,473 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 2: skipping OCR (oversize_crop_1348x541)
2026-04-24 11:01:46,473 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 3: skipping OCR (oversize_crop_1008x267)
2026-04-24 11:01:46,473 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 2 text regions
2026-04-24 11:01:46,473 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: 'BEST BEFORE', 3: '04/2028'}
2026-04-24 11:01:46,473 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3]
2026-04-24 11:01:46,474 - camera_management.verification.text_verifier - WARNING - [40767173] ALL 2 bboxes invalid — template likely mismatched. Skipping OCR batch.
2026-04-24 11:01:46,474 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:01:46,474 - camera_management.verification.template_verifier - ERROR - [40767173] Invalid crop size: 9769x1364. Bounds: (-8893,-171)→(877,1193), Frame: 1920x1200. Skipping template verification.
2026-04-24 11:01:46,483 - camera_management.utils - INFO - Drew 3 bboxes on frame
2026-04-24 11:01:46,507 - camera_management.utils - INFO - AsyncImageSaver initialized with 4 workers
2026-04-24 11:01:46,509 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f1_20260424_040146480721_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:01:46,515 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f0_20260424_040146483973_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:01:46,526 - camera_management.result_builder.builder - INFO - Result builder (parallel): 2 frames encoded in 50.2ms, total=51.1ms
2026-04-24 11:01:46,526 - camera_management.pipeline.base - INFO - [Job #1] Pipeline complete: FAIL, time: 0.296s
2026-04-24 11:01:46,526 - camera_management.inference_handler - INFO - [Job #1] Camera 40767173: pass=0, fail=2, error=0
2026-04-24 11:01:46,529 - camera_management.utils - INFO - Reject action logger initialized: /home/suntech/Source/ocr_datecode/ai_services/logs/reject_actions_2026-04-24.log
2026-04-24 11:01:46,529 - camera_management.reject_scheduler - INFO - [Group #1] 🕐 Reject scheduled @ T=1777003307.728s (in 1.204s), Reject(DO/D3) + Alarm(DO/D0)
2026-04-24 11:01:46,529 - camera_management.inference_handler - INFO - [Group #1] Reject scheduled (delay_reject=1500.0ms, DO3)
2026-04-24 11:01:46,529 - camera_management.inference_handler - INFO - [Job #1] Completed successfully
2026-04-24 11:01:47,728 - camera_management.reject_scheduler - INFO - [Group #1] 🔴 REJECTING! Method: DIO_OUT, Reject(3) + Alarm(0), Pulse: 50.0ms
2026-04-24 11:01:47,728 - camera_management.reject_scheduler - INFO - [Group #1] Executing parallel DIO pulses (DO3 + DO0)
2026-04-24 11:01:47,875 - camera_management.utils - INFO - DO0 pulse complete (50.0ms)
2026-04-24 11:01:47,919 - camera_management.utils - INFO - DO3 pulse complete (50.0ms)
2026-04-24 11:01:47,919 - camera_management.reject_scheduler - INFO - [Group #1] ✅ DIO parallel pulses successful (Reject DO3 + Alarm DO0)
2026-04-24 11:01:47,920 - camera_management.reject_scheduler - INFO - [Group #1] Reject complete (method: DIO_OUT, duration: 191.26ms)
2026-04-24 11:01:54,014 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:01:54,017 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:02:04,024 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:02:04,026 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:02:14,034 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:02:14,034 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:02:24,042 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:02:24,044 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:02:34,050 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:02:34,055 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:02:44,058 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:02:44,065 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:02:54,066 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:02:54,074 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:03:04,074 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:03:04,084 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:03:14,079 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:03:14,094 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:03:24,087 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:03:24,103 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:03:34,094 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:03:34,111 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:03:44,102 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 1, Rejected: 1 (PLC: 0, DIO: 1, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:03:44,119 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=0, Groups=1/1 (100.0%), Inferences=1, CaptureFail=0, Active=0g/0t
2026-04-24 11:03:49,319 - camera_management.trigger_handler - INFO - DI0 edge detected (0→1), triggering 1 camera(s)
2026-04-24 11:03:49,320 - camera_management.trigger_handler - INFO - [Group #2] Creating capture group for 1 camera(s): [('40767173', 500.0)]
2026-04-24 11:03:49,320 - camera_management.trigger_handler - INFO - [Group #2] Scheduling Camera 40767173 capture in 0.50s
2026-04-24 11:03:49,821 - camera_management.trigger_handler - INFO - [Group #2] Camera 40767173 timer FIRED! Capturing now (Thread: CaptureTimer-2-40767173)
2026-04-24 11:03:49,821 - camera_management.camera - INFO - [40767173] Capturing 2 frames with delay_interval=1200.0ms between frames (REVERSED order)
2026-04-24 11:03:51,187 - camera_management.camera - INFO - [40767173] ✅ Captured 2 frames (reordered to match templates)
2026-04-24 11:03:51,188 - camera_management.trigger_handler - INFO - [Group #2] Camera 40767173 captured 2 frames. Progress: 1/1
2026-04-24 11:03:51,188 - camera_management.trigger_handler - INFO - [Group #2] ✅ All 1 cameras completed! Triggering batch inference.
2026-04-24 11:03:51,194 - __main__ - INFO - Frames captured (Group #2), submitting async inference for 1 camera(s)...
2026-04-24 11:03:51,194 - camera_management.inference_handler - INFO - [Job #2] Submitting inference for 1 camera(s) (queue depth: 0)
2026-04-24 11:03:51,195 - camera_management.inference_handler - INFO - [Job #2] Starting inference in InferenceWorker_0 (group_id: 2)
2026-04-24 11:03:51,195 - camera_management.inference_handler - INFO - [Job #2] Using SingleCameraPipeline
2026-04-24 11:03:51,405 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 2: skipping OCR (oversize_crop_489x68)
2026-04-24 11:03:51,409 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 2 text regions
2026-04-24 11:03:51,409 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: 'BEST BEFORE', 3: '04/2028'}
2026-04-24 11:03:51,409 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3]
2026-04-24 11:03:51,409 - camera_management.verification.text_verifier - INFO - [40767173] 1/2 bboxes invalid, OCR will skip those
2026-04-24 11:03:51,409 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 1 regions (chunked by max_batch=4)
2026-04-24 11:03:51,468 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 1 regions in 1 chunk(s), 58.1ms
2026-04-24 11:03:51,469 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 3: expected='04/2028', recognized='04/2028', match=True, conf=100.00%
2026-04-24 11:03:51,469 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:03:51,492 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (146, 547, 3) vs (148, 554, 3)
2026-04-24 11:03:51,495 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (146, 547) -> (53, 200) (scale=0.37)
2026-04-24 11:03:51,496 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.9543, threshold=0.65, match=True, method=TM_CCOEFF_NORMED, total=26.4ms (crop=9.5ms, resize=1.0ms, opt_resize=1.1ms, matching=0.6ms)
2026-04-24 11:03:51,496 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 2: skipping OCR (oversize_crop_315x66)
2026-04-24 11:03:51,496 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 3: skipping OCR (oversize_crop_708x74)
2026-04-24 11:03:51,499 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 3 text regions
2026-04-24 11:03:51,499 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: '297538', 3: 'SAIGON CINNAMON', 4: 'UK'}
2026-04-24 11:03:51,499 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3, 4]
2026-04-24 11:03:51,499 - camera_management.verification.text_verifier - INFO - [40767173] 2/3 bboxes invalid, OCR will skip those
2026-04-24 11:03:51,499 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 1 regions (chunked by max_batch=4)
2026-04-24 11:03:51,509 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 1 regions in 1 chunk(s), 9.4ms
2026-04-24 11:03:51,509 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 4: expected='UK', recognized='UK', match=True, conf=87.04%
2026-04-24 11:03:51,510 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:03:51,547 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (234, 845, 3) vs (234, 833, 3)
2026-04-24 11:03:51,551 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (234, 845) -> (55, 200) (scale=0.24)
2026-04-24 11:03:51,552 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.9818, threshold=0.6, match=True, method=TM_CCOEFF_NORMED, total=42.4ms (crop=8.5ms, resize=1.1ms, opt_resize=2.4ms, matching=0.6ms)
2026-04-24 11:03:51,557 - camera_management.utils - INFO - Drew 3 bboxes on frame
2026-04-24 11:03:51,557 - camera_management.utils - INFO - Drew 4 bboxes on frame
2026-04-24 11:03:51,565 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f0_20260424_040351558401_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:03:51,575 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f1_20260424_040351558726_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:03:51,579 - camera_management.result_builder.builder - INFO - Result builder (parallel): 2 frames encoded in 26.2ms, total=26.4ms
2026-04-24 11:03:51,579 - camera_management.pipeline.base - INFO - [Job #2] Pipeline complete: FAIL, time: 0.383s
2026-04-24 11:03:51,579 - camera_management.inference_handler - INFO - [Job #2] Camera 40767173: pass=0, fail=2, error=0
2026-04-24 11:03:51,580 - camera_management.reject_scheduler - INFO - [Group #2] 🕐 Reject scheduled @ T=1777003432.689s (in 1.116s), Reject(DO/D3) + Alarm(DO/D0)
2026-04-24 11:03:51,580 - camera_management.inference_handler - INFO - [Group #2] Reject scheduled (delay_reject=1500.0ms, DO3)
2026-04-24 11:03:51,580 - camera_management.inference_handler - INFO - [Job #2] Completed successfully
2026-04-24 11:03:51,583 - camera_management.websocket_client - ERROR - Error sending message: Object of type float32 is not JSON serializable
2026-04-24 11:03:52,196 - camera_management.websocket_client - INFO - Attempting to reconnect in 1s...
2026-04-24 11:03:52,688 - camera_management.reject_scheduler - INFO - [Group #2] 🔴 REJECTING! Method: DIO_OUT, Reject(3) + Alarm(0), Pulse: 50.0ms
2026-04-24 11:03:52,689 - camera_management.reject_scheduler - INFO - [Group #2] Executing parallel DIO pulses (DO3 + DO0)
2026-04-24 11:03:52,825 - camera_management.utils - INFO - DO0 pulse complete (50.0ms)
2026-04-24 11:03:52,868 - camera_management.utils - INFO - DO3 pulse complete (50.0ms)
2026-04-24 11:03:52,869 - camera_management.reject_scheduler - INFO - [Group #2] ✅ DIO parallel pulses successful (Reject DO3 + Alarm DO0)
2026-04-24 11:03:52,870 - camera_management.reject_scheduler - INFO - [Group #2] Reject complete (method: DIO_OUT, duration: 180.32ms)
2026-04-24 11:03:53,200 - camera_management.websocket_client - INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
2026-04-24 11:03:53,217 - camera_management.websocket_client - INFO - WebSocket connected successfully
2026-04-24 11:03:53,220 - camera_management.websocket_client - WARNING - WebSocket connection closed
2026-04-24 11:03:54,110 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 2, Rejected: 2 (PLC: 0, DIO: 2, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:03:54,129 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=1, Groups=2/2 (100.0%), Inferences=2, CaptureFail=0, Active=0g/0t
2026-04-24 11:03:54,221 - camera_management.websocket_client - INFO - Attempting to reconnect in 1s...
2026-04-24 11:03:55,224 - camera_management.websocket_client - INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
2026-04-24 11:03:55,234 - camera_management.websocket_client - INFO - WebSocket connected successfully
2026-04-24 11:04:04,118 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 2, Rejected: 2 (PLC: 0, DIO: 2, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:04:04,138 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=1, Groups=2/2 (100.0%), Inferences=2, CaptureFail=0, Active=0g/0t
2026-04-24 11:04:14,127 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 2, Rejected: 2 (PLC: 0, DIO: 2, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:04:14,148 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=1, Groups=2/2 (100.0%), Inferences=2, CaptureFail=0, Active=0g/0t
2026-04-24 11:04:24,134 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 2, Rejected: 2 (PLC: 0, DIO: 2, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:04:24,159 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=1, Groups=2/2 (100.0%), Inferences=2, CaptureFail=0, Active=0g/0t
2026-04-24 11:04:31,246 - camera_management.trigger_handler - INFO - DI0 edge detected (0→1), triggering 1 camera(s)
2026-04-24 11:04:31,246 - camera_management.trigger_handler - INFO - [Group #3] Creating capture group for 1 camera(s): [('40767173', 500.0)]
2026-04-24 11:04:31,246 - camera_management.trigger_handler - INFO - [Group #3] Scheduling Camera 40767173 capture in 0.50s
2026-04-24 11:04:31,747 - camera_management.trigger_handler - INFO - [Group #3] Camera 40767173 timer FIRED! Capturing now (Thread: CaptureTimer-3-40767173)
2026-04-24 11:04:31,748 - camera_management.camera - INFO - [40767173] Capturing 2 frames with delay_interval=1200.0ms between frames (REVERSED order)
2026-04-24 11:04:33,098 - camera_management.camera - INFO - [40767173] ✅ Captured 2 frames (reordered to match templates)
2026-04-24 11:04:33,099 - camera_management.trigger_handler - INFO - [Group #3] Camera 40767173 captured 2 frames. Progress: 1/1
2026-04-24 11:04:33,099 - camera_management.trigger_handler - INFO - [Group #3] ✅ All 1 cameras completed! Triggering batch inference.
2026-04-24 11:04:33,100 - __main__ - INFO - Frames captured (Group #3), submitting async inference for 1 camera(s)...
2026-04-24 11:04:33,100 - camera_management.inference_handler - INFO - [Job #3] Submitting inference for 1 camera(s) (queue depth: 0)
2026-04-24 11:04:33,100 - camera_management.inference_handler - INFO - [Job #3] Starting inference in InferenceWorker_0 (group_id: 3)
2026-04-24 11:04:33,101 - camera_management.inference_handler - INFO - [Job #3] Using SingleCameraPipeline
2026-04-24 11:04:33,335 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 2: skipping OCR (oversize_crop_485x63)
2026-04-24 11:04:33,340 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 2 text regions
2026-04-24 11:04:33,341 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: 'BEST BEFORE', 3: '04/2028'}
2026-04-24 11:04:33,341 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3]
2026-04-24 11:04:33,341 - camera_management.verification.text_verifier - INFO - [40767173] 1/2 bboxes invalid, OCR will skip those
2026-04-24 11:04:33,341 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 1 regions (chunked by max_batch=4)
2026-04-24 11:04:33,351 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 1 regions in 1 chunk(s), 10.5ms
2026-04-24 11:04:33,352 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 3: expected='04/2028', recognized='04/2028', match=True, conf=99.97%
2026-04-24 11:04:33,352 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:04:33,372 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (146, 547, 3) vs (148, 564, 3)
2026-04-24 11:04:33,374 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (146, 547) -> (53, 200) (scale=0.37)
2026-04-24 11:04:33,375 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.9103, threshold=0.65, match=True, method=TM_CCOEFF_NORMED, total=22.8ms (crop=4.9ms, resize=0.9ms, opt_resize=1.2ms, matching=0.6ms)
2026-04-24 11:04:33,375 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 2: skipping OCR (oversize_crop_317x65)
2026-04-24 11:04:33,376 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 3: skipping OCR (oversize_crop_713x73)
2026-04-24 11:04:33,377 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 3 text regions
2026-04-24 11:04:33,378 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: '297538', 3: 'SAIGON CINNAMON', 4: 'UK'}
2026-04-24 11:04:33,378 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3, 4]
2026-04-24 11:04:33,378 - camera_management.verification.text_verifier - INFO - [40767173] 2/3 bboxes invalid, OCR will skip those
2026-04-24 11:04:33,378 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 1 regions (chunked by max_batch=4)
2026-04-24 11:04:33,384 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 1 regions in 1 chunk(s), 6.5ms
2026-04-24 11:04:33,385 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 4: expected='UK', recognized='UK', match=True, conf=82.98%
2026-04-24 11:04:33,385 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:04:33,425 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (234, 845, 3) vs (233, 838, 3)
2026-04-24 11:04:33,428 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (234, 845) -> (55, 200) (scale=0.24)
2026-04-24 11:04:33,429 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.9772, threshold=0.6, match=True, method=TM_CCOEFF_NORMED, total=43.6ms (crop=10.2ms, resize=0.9ms, opt_resize=2.2ms, matching=0.6ms)
2026-04-24 11:04:33,434 - camera_management.utils - INFO - Drew 3 bboxes on frame
2026-04-24 11:04:33,434 - camera_management.utils - INFO - Drew 4 bboxes on frame
2026-04-24 11:04:33,445 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f0_20260424_040433435226_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:04:33,448 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f1_20260424_040433435446_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:04:33,454 - camera_management.result_builder.builder - INFO - Result builder (parallel): 2 frames encoded in 25.0ms, total=25.2ms
2026-04-24 11:04:33,455 - camera_management.pipeline.base - INFO - [Job #3] Pipeline complete: FAIL, time: 0.354s
2026-04-24 11:04:33,455 - camera_management.websocket_client - ERROR - Error sending message: Object of type float32 is not JSON serializable
2026-04-24 11:04:33,456 - camera_management.inference_handler - INFO - [Job #3] Camera 40767173: pass=0, fail=2, error=0
2026-04-24 11:04:33,456 - camera_management.reject_scheduler - INFO - [Group #3] 🕐 Reject scheduled @ T=1777003474.599s (in 1.145s), Reject(DO/D3) + Alarm(DO/D0)
2026-04-24 11:04:33,456 - camera_management.inference_handler - INFO - [Group #3] Reject scheduled (delay_reject=1500.0ms, DO3)
2026-04-24 11:04:33,457 - camera_management.inference_handler - INFO - [Job #3] Completed successfully
2026-04-24 11:04:34,142 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 3, Rejected: 2 (PLC: 0, DIO: 2, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 1
2026-04-24 11:04:34,169 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=2, Groups=3/3 (100.0%), Inferences=3, CaptureFail=0, Active=0g/0t
2026-04-24 11:04:34,281 - camera_management.websocket_client - INFO - Attempting to reconnect in 1s...
2026-04-24 11:04:34,599 - camera_management.reject_scheduler - INFO - [Group #3] 🔴 REJECTING! Method: DIO_OUT, Reject(3) + Alarm(0), Pulse: 50.0ms
2026-04-24 11:04:34,600 - camera_management.reject_scheduler - INFO - [Group #3] Executing parallel DIO pulses (DO3 + DO0)
2026-04-24 11:04:34,719 - camera_management.utils - INFO - DO3 pulse complete (50.0ms)
2026-04-24 11:04:34,763 - camera_management.utils - INFO - DO0 pulse complete (50.0ms)
2026-04-24 11:04:34,764 - camera_management.reject_scheduler - INFO - [Group #3] ✅ DIO parallel pulses successful (Reject DO3 + Alarm DO0)
2026-04-24 11:04:34,765 - camera_management.reject_scheduler - INFO - [Group #3] Reject complete (method: DIO_OUT, duration: 164.43ms)
2026-04-24 11:04:35,284 - camera_management.websocket_client - INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
2026-04-24 11:04:35,295 - camera_management.websocket_client - INFO - WebSocket connected successfully
2026-04-24 11:04:35,296 - camera_management.websocket_client - WARNING - WebSocket connection closed
2026-04-24 11:04:36,297 - camera_management.websocket_client - INFO - Attempting to reconnect in 1s...
2026-04-24 11:04:37,299 - camera_management.websocket_client - INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
2026-04-24 11:04:37,306 - camera_management.websocket_client - INFO - WebSocket connected successfully
2026-04-24 11:04:44,150 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 3, Rejected: 3 (PLC: 0, DIO: 3, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:04:44,178 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=2, Groups=3/3 (100.0%), Inferences=3, CaptureFail=0, Active=0g/0t
2026-04-24 11:04:54,158 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 3, Rejected: 3 (PLC: 0, DIO: 3, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:04:54,187 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=2, Groups=3/3 (100.0%), Inferences=3, CaptureFail=0, Active=0g/0t
2026-04-24 11:04:59,459 - camera_management.trigger_handler - INFO - DI0 edge detected (0→1), triggering 1 camera(s)
2026-04-24 11:04:59,459 - camera_management.trigger_handler - INFO - [Group #4] Creating capture group for 1 camera(s): [('40767173', 500.0)]
2026-04-24 11:04:59,459 - camera_management.trigger_handler - INFO - [Group #4] Scheduling Camera 40767173 capture in 0.50s
2026-04-24 11:04:59,960 - camera_management.trigger_handler - INFO - [Group #4] Camera 40767173 timer FIRED! Capturing now (Thread: CaptureTimer-4-40767173)
2026-04-24 11:04:59,960 - camera_management.camera - INFO - [40767173] Capturing 2 frames with delay_interval=1200.0ms between frames (REVERSED order)
2026-04-24 11:05:01,304 - camera_management.camera - INFO - [40767173] ✅ Captured 2 frames (reordered to match templates)
2026-04-24 11:05:01,304 - camera_management.trigger_handler - INFO - [Group #4] Camera 40767173 captured 2 frames. Progress: 1/1
2026-04-24 11:05:01,305 - camera_management.trigger_handler - INFO - [Group #4] ✅ All 1 cameras completed! Triggering batch inference.
2026-04-24 11:05:01,305 - __main__ - INFO - Frames captured (Group #4), submitting async inference for 1 camera(s)...
2026-04-24 11:05:01,305 - camera_management.inference_handler - INFO - [Job #4] Submitting inference for 1 camera(s) (queue depth: 0)
2026-04-24 11:05:01,306 - camera_management.inference_handler - INFO - [Job #4] Starting inference in InferenceWorker_0 (group_id: 4)
2026-04-24 11:05:01,310 - camera_management.inference_handler - INFO - [Job #4] Using SingleCameraPipeline
2026-04-24 11:05:01,680 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 2 text regions
2026-04-24 11:05:01,681 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: 'BEST BEFORE', 3: '04/2028'}
2026-04-24 11:05:01,681 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3]
2026-04-24 11:05:01,681 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 2 regions (chunked by max_batch=4)
2026-04-24 11:05:01,693 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 2 regions in 1 chunk(s), 12.3ms
2026-04-24 11:05:01,694 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 2: expected='BEST BEFORE', recognized='', match=False, conf=0.00%
2026-04-24 11:05:01,694 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 2: FAIL and similarity=0.00% < 70%, skip augment retry
2026-04-24 11:05:01,694 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 3: expected='04/2028', recognized='', match=False, conf=0.00%
2026-04-24 11:05:01,694 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 3: FAIL and similarity=0.00% < 70%, skip augment retry
2026-04-24 11:05:01,695 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:05:01,753 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (146, 547, 3) vs (1200, 1920, 3)
2026-04-24 11:05:01,756 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (146, 547) -> (53, 200) (scale=0.37)
2026-04-24 11:05:01,757 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.0000, threshold=0.65, match=False, method=TM_CCOEFF_NORMED, total=62.6ms (crop=27.0ms, resize=1.4ms, opt_resize=1.1ms, matching=0.6ms)
2026-04-24 11:05:01,810 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 3: skipping OCR (oversize_crop_605x135)
2026-04-24 11:05:01,811 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 4: skipping OCR (oversize_crop_533x177)
2026-04-24 11:05:01,811 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 3 text regions
2026-04-24 11:05:01,811 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: '297538', 3: 'SAIGON CINNAMON', 4: 'UK'}
2026-04-24 11:05:01,811 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3, 4]
2026-04-24 11:05:01,811 - camera_management.verification.text_verifier - INFO - [40767173] 2/3 bboxes invalid, OCR will skip those
2026-04-24 11:05:01,811 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 1 regions (chunked by max_batch=4)
2026-04-24 11:05:01,818 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 1 regions in 1 chunk(s), 7.0ms
2026-04-24 11:05:01,819 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 2: expected='297538', recognized='', match=False, conf=0.00%
2026-04-24 11:05:01,819 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 2: FAIL and similarity=0.00% < 70%, skip augment retry
2026-04-24 11:05:01,819 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:05:01,845 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (234, 845, 3) vs (611, 662, 3)
2026-04-24 11:05:01,848 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (234, 845) -> (55, 200) (scale=0.24)
2026-04-24 11:05:01,849 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.0000, threshold=0.6, match=False, method=TM_CCOEFF_NORMED, total=29.6ms (crop=5.9ms, resize=1.2ms, opt_resize=2.1ms, matching=0.6ms)
2026-04-24 11:05:01,854 - camera_management.utils - INFO - Drew 3 bboxes on frame
2026-04-24 11:05:01,855 - camera_management.utils - INFO - Drew 4 bboxes on frame
2026-04-24 11:05:01,859 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f0_20260424_040501855151_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:05:01,867 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f1_20260424_040501855425_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:05:01,871 - camera_management.result_builder.builder - INFO - Result builder (parallel): 2 frames encoded in 21.0ms, total=21.2ms
2026-04-24 11:05:01,871 - camera_management.pipeline.base - INFO - [Job #4] Pipeline complete: FAIL, time: 0.560s
2026-04-24 11:05:01,871 - camera_management.inference_handler - INFO - [Job #4] Camera 40767173: pass=0, fail=2, error=0
2026-04-24 11:05:01,873 - camera_management.reject_scheduler - INFO - [Group #4] 🕐 Reject scheduled @ T=1777003502.805s (in 0.939s), Reject(DO/D3) + Alarm(DO/D0)
2026-04-24 11:05:01,873 - camera_management.inference_handler - INFO - [Group #4] Reject scheduled (delay_reject=1500.0ms, DO3)
2026-04-24 11:05:01,873 - camera_management.inference_handler - INFO - [Job #4] Completed successfully
2026-04-24 11:05:02,805 - camera_management.reject_scheduler - INFO - [Group #4] 🔴 REJECTING! Method: DIO_OUT, Reject(3) + Alarm(0), Pulse: 50.0ms
2026-04-24 11:05:02,806 - camera_management.reject_scheduler - INFO - [Group #4] Executing parallel DIO pulses (DO3 + DO0)
2026-04-24 11:05:02,948 - camera_management.utils - INFO - DO0 pulse complete (50.0ms)
2026-04-24 11:05:02,993 - camera_management.utils - INFO - DO3 pulse complete (50.0ms)
2026-04-24 11:05:02,993 - camera_management.reject_scheduler - INFO - [Group #4] ✅ DIO parallel pulses successful (Reject DO3 + Alarm DO0)
2026-04-24 11:05:02,994 - camera_management.reject_scheduler - INFO - [Group #4] Reject complete (method: DIO_OUT, duration: 188.07ms)
2026-04-24 11:05:04,166 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 4, Rejected: 4 (PLC: 0, DIO: 4, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:05:04,194 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=3, Groups=4/4 (100.0%), Inferences=4, CaptureFail=0, Active=0g/0t
2026-04-24 11:05:14,174 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 4, Rejected: 4 (PLC: 0, DIO: 4, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:05:14,204 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=3, Groups=4/4 (100.0%), Inferences=4, CaptureFail=0, Active=0g/0t
2026-04-24 11:05:24,183 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 4, Rejected: 4 (PLC: 0, DIO: 4, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:05:24,215 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=3, Groups=4/4 (100.0%), Inferences=4, CaptureFail=0, Active=0g/0t
2026-04-24 11:05:34,194 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 4, Rejected: 4 (PLC: 0, DIO: 4, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:05:34,222 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=3, Groups=4/4 (100.0%), Inferences=4, CaptureFail=0, Active=0g/0t
2026-04-24 11:05:44,202 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 4, Rejected: 4 (PLC: 0, DIO: 4, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:05:44,232 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=3, Groups=4/4 (100.0%), Inferences=4, CaptureFail=0, Active=0g/0t
2026-04-24 11:05:54,042 - camera_management.trigger_handler - INFO - DI0 edge detected (0→1), triggering 1 camera(s)
2026-04-24 11:05:54,042 - camera_management.trigger_handler - INFO - [Group #5] Creating capture group for 1 camera(s): [('40767173', 500.0)]
2026-04-24 11:05:54,043 - camera_management.trigger_handler - INFO - [Group #5] Scheduling Camera 40767173 capture in 0.50s
2026-04-24 11:05:54,210 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 4, Rejected: 4 (PLC: 0, DIO: 4, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:05:54,242 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=4/5 (80.0%), Inferences=4, CaptureFail=0, Active=1g/2t
2026-04-24 11:05:54,543 - camera_management.trigger_handler - INFO - [Group #5] Camera 40767173 timer FIRED! Capturing now (Thread: CaptureTimer-5-40767173)
2026-04-24 11:05:54,544 - camera_management.camera - INFO - [40767173] Capturing 2 frames with delay_interval=1200.0ms between frames (REVERSED order)
2026-04-24 11:05:55,885 - camera_management.camera - INFO - [40767173] ✅ Captured 2 frames (reordered to match templates)
2026-04-24 11:05:55,886 - camera_management.trigger_handler - INFO - [Group #5] Camera 40767173 captured 2 frames. Progress: 1/1
2026-04-24 11:05:55,886 - camera_management.trigger_handler - INFO - [Group #5] ✅ All 1 cameras completed! Triggering batch inference.
2026-04-24 11:05:55,898 - __main__ - INFO - Frames captured (Group #5), submitting async inference for 1 camera(s)...
2026-04-24 11:05:55,898 - camera_management.inference_handler - INFO - [Job #5] Submitting inference for 1 camera(s) (queue depth: 0)
2026-04-24 11:05:55,899 - camera_management.inference_handler - INFO - [Job #5] Starting inference in InferenceWorker_0 (group_id: 5)
2026-04-24 11:05:55,899 - camera_management.inference_handler - INFO - [Job #5] Using SingleCameraPipeline
2026-04-24 11:05:56,117 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 2: skipping OCR (oversize_crop_485x66)
2026-04-24 11:05:56,123 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 2 text regions
2026-04-24 11:05:56,123 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: 'BEST BEFORE', 3: '04/2028'}
2026-04-24 11:05:56,123 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3]
2026-04-24 11:05:56,123 - camera_management.verification.text_verifier - INFO - [40767173] 1/2 bboxes invalid, OCR will skip those
2026-04-24 11:05:56,123 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 1 regions (chunked by max_batch=4)
2026-04-24 11:05:56,132 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 1 regions in 1 chunk(s), 8.2ms
2026-04-24 11:05:56,135 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 3: expected='04/2028', recognized='04/2028', match=True, conf=99.99%
2026-04-24 11:05:56,136 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:05:56,161 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (146, 547, 3) vs (150, 549, 3)
2026-04-24 11:05:56,164 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (146, 547) -> (53, 200) (scale=0.37)
2026-04-24 11:05:56,165 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.9218, threshold=0.65, match=True, method=TM_CCOEFF_NORMED, total=28.9ms (crop=5.5ms, resize=0.9ms, opt_resize=1.1ms, matching=0.7ms)
2026-04-24 11:05:56,165 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 2: skipping OCR (oversize_crop_314x65)
2026-04-24 11:05:56,165 - camera_management.verification.text_verifier - WARNING - [40767173] Ann 3: skipping OCR (oversize_crop_705x73)
2026-04-24 11:05:56,168 - camera_management.verification.text_verifier - INFO - [40767173] Verifying 3 text regions
2026-04-24 11:05:56,168 - camera_management.verification.text_verifier - INFO - [40767173] Expected texts dict: {2: '297538', 3: 'SAIGON CINNAMON', 4: 'UK'}
2026-04-24 11:05:56,168 - camera_management.verification.text_verifier - INFO - [40767173] Text bbox annotation indices: [2, 3, 4]
2026-04-24 11:05:56,168 - camera_management.verification.text_verifier - INFO - [40767173] 2/3 bboxes invalid, OCR will skip those
2026-04-24 11:05:56,168 - camera_management.verification.text_verifier - INFO - Running BATCH OCR on 1 regions (chunked by max_batch=4)
2026-04-24 11:05:56,177 - camera_management.verification.text_verifier - INFO - Batch OCR complete: 1 regions in 1 chunk(s), 8.6ms
2026-04-24 11:05:56,178 - camera_management.verification.text_verifier - INFO - [40767173] Annotation 4: expected='UK', recognized='UK', match=True, conf=81.57%
2026-04-24 11:05:56,178 - camera_management.verification.template_verifier - INFO - [40767173] Verifying 1 template regions
2026-04-24 11:05:56,212 - camera_management.verification.template_verifier - WARNING - [40767173] Template and target crop size mismatch: (234, 845, 3) vs (231, 831, 3)
2026-04-24 11:05:56,216 - camera_management.verification.template_verifier - INFO - [40767173] Resized for matching: (234, 845) -> (55, 200) (scale=0.24)
2026-04-24 11:05:56,217 - camera_management.verification.template_verifier - INFO - [40767173] Template verification: similarity=0.9760, threshold=0.6, match=True, method=TM_CCOEFF_NORMED, total=38.6ms (crop=5.8ms, resize=1.1ms, opt_resize=2.2ms, matching=0.6ms)
2026-04-24 11:05:56,221 - camera_management.utils - INFO - Drew 3 bboxes on frame
2026-04-24 11:05:56,222 - camera_management.utils - INFO - Drew 4 bboxes on frame
2026-04-24 11:05:56,228 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f0_20260424_040556222200_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:05:56,236 - camera_management.utils - INFO - Queued FAIL frame for async save: inference_results/69eadff5cba186658062a667/2026-04-24/40767173/fail_f1_20260424_040556224714_viz.jpg (full: 1920x1200, display: 640x400)
2026-04-24 11:05:56,238 - camera_management.result_builder.builder - INFO - Result builder (parallel): 2 frames encoded in 20.8ms, total=21.0ms
2026-04-24 11:05:56,238 - camera_management.pipeline.base - INFO - [Job #5] Pipeline complete: FAIL, time: 0.339s
2026-04-24 11:05:56,239 - camera_management.websocket_client - ERROR - Error sending message: Object of type float32 is not JSON serializable
2026-04-24 11:05:56,239 - camera_management.inference_handler - INFO - [Job #5] Camera 40767173: pass=0, fail=2, error=0
2026-04-24 11:05:56,240 - camera_management.reject_scheduler - INFO - [Group #5] 🕐 Reject scheduled @ T=1777003557.387s (in 1.160s), Reject(DO/D3) + Alarm(DO/D0)
2026-04-24 11:05:56,240 - camera_management.inference_handler - INFO - [Group #5] Reject scheduled (delay_reject=1500.0ms, DO3)
2026-04-24 11:05:56,240 - camera_management.inference_handler - INFO - [Job #5] Completed successfully
2026-04-24 11:05:56,407 - camera_management.websocket_client - INFO - Attempting to reconnect in 1s...
2026-04-24 11:05:57,387 - camera_management.reject_scheduler - INFO - [Group #5] 🔴 REJECTING! Method: DIO_OUT, Reject(3) + Alarm(0), Pulse: 50.0ms
2026-04-24 11:05:57,387 - camera_management.reject_scheduler - INFO - [Group #5] Executing parallel DIO pulses (DO3 + DO0)
2026-04-24 11:05:57,409 - camera_management.websocket_client - INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
2026-04-24 11:05:57,416 - camera_management.websocket_client - INFO - WebSocket connected successfully
2026-04-24 11:05:57,417 - camera_management.websocket_client - WARNING - WebSocket connection closed
2026-04-24 11:05:57,510 - camera_management.utils - INFO - DO3 pulse complete (50.0ms)
2026-04-24 11:05:57,554 - camera_management.utils - INFO - DO0 pulse complete (50.0ms)
2026-04-24 11:05:57,554 - camera_management.reject_scheduler - INFO - [Group #5] ✅ DIO parallel pulses successful (Reject DO3 + Alarm DO0)
2026-04-24 11:05:57,555 - camera_management.reject_scheduler - INFO - [Group #5] Reject complete (method: DIO_OUT, duration: 167.49ms)
2026-04-24 11:05:58,418 - camera_management.websocket_client - INFO - Attempting to reconnect in 1s...
2026-04-24 11:05:59,419 - camera_management.websocket_client - INFO - Connecting to WebSocket server: ws://localhost:8000/ws/camera-management
2026-04-24 11:05:59,426 - camera_management.websocket_client - INFO - WebSocket connected successfully
2026-04-24 11:06:04,220 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:06:04,252 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t
2026-04-24 11:06:14,230 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:06:14,262 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t
2026-04-24 11:06:24,238 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:06:24,272 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t
2026-04-24 11:06:34,246 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:06:34,282 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t
2026-04-24 11:06:44,256 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:06:44,292 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t
2026-04-24 11:06:54,266 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:06:54,303 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t
2026-04-24 11:07:04,275 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:07:04,310 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t
2026-04-24 11:07:14,286 - camera_management.reject_scheduler - INFO - 📊 [REJECT STATS] Scheduled: 5, Rejected: 5 (PLC: 0, DIO: 5, Fallback: 0, Reconnect: 0), Cancelled: 0, Missed: 0, Max Queue: 1, Active: 0
2026-04-24 11:07:14,319 - camera_management.trigger_handler - INFO - 📊 [STATS] Triggers=4, Groups=5/5 (100.0%), Inferences=5, CaptureFail=0, Active=0g/0t