import requests
import json
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from ai_services.camera_management.text_recognizer import TextRecognizer

API_URL = "https://suntech-vision-api.ngrok.app/api/inference-results/?skip=0&limit=20&start_date=2026-01-12T06:15:47.000Z&end_date=2026-01-19T06:15:47.000Z&pass_fail=FAIL"
TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc3MTI2ODE0M30.psQknrFm1UH16_hhJM2a8FC-1a3JrTvQ6ovfgAJk_Vc"
BASE_URL = "https://suntech-vision-api.ngrok.app"
DATA_DIR = Path(__file__).parent / "inference_data"
IMAGES_DIR = DATA_DIR / "images"
JSON_FILE = DATA_DIR / "fail_records.json"

DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

recognizer = TextRecognizer(
    model_path='languages/english/rec.onnx',
    dict_path='languages/english/dict.txt',
    use_gpu=True
)

def fetch_and_save_data():
    if JSON_FILE.exists():
        print(f"Loading existing data from {JSON_FILE}")
        with open(JSON_FILE, 'r') as f:
            return json.load(f)

    print(f"Fetching data from API...")
    response = requests.get(API_URL, headers={"Authorization": TOKEN})
    response.raise_for_status()
    data = response.json()

    fail_frames = []
    results = data if isinstance(data, list) else data.get("results", [])
    for record in results:
        for cam_result in record.get("camera_results", []):
            for frame in cam_result.get("frames", []):
                if frame.get("pass_fail") == "FAIL":
                    fail_frames.append({
                        "_id": record["_id"],
                        "recipe_id": record["recipe_id"],
                        "recipe_name": record["recipe_name"],
                        "camera_id": cam_result["camera_id"],
                        "serial_number": cam_result["serial_number"],
                        "frame": frame,
                        "timestamp": record["timestamp"]
                    })

    with open(JSON_FILE, 'w') as f:
        json.dump(fail_frames, f, indent=2)

    print(f"Saved {len(fail_frames)} fail frames to {JSON_FILE}")
    return fail_frames

def download_image(url):
    img_name = url.split('/')[-1]
    img_path = IMAGES_DIR / img_name

    if img_path.exists():
        return cv2.imread(str(img_path))

    full_url = f"{BASE_URL}/api/uploads/{url}"
    print(f"Downloading {full_url}")
    response = requests.get(full_url)
    response.raise_for_status()

    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite(str(img_path), img)

    return img

def visualize_regions(img, detected_regions):
    vis_img = img.copy()

    for region in detected_regions:
        points = np.array(region["points"], dtype=np.int32)

        if region["type"] == "template":
            color = (0, 255, 0)
            label = "Template"
        else:
            color = (255, 0, 0)
            label = f"Text: {region.get('text', 'N/A')}"

        cv2.polylines(vis_img, [points], True, color, 3)
        cv2.putText(vis_img, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return vis_img

def crop_and_recognize_text(img, detected_regions):
    results = []

    for region in detected_regions:
        if region["type"] != "text":
            continue

        points = np.array(region["points"], dtype=np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        width = int(rect[1][0])
        height = int(rect[1][1])

        if width < height:
            width, height = height, width

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped = cv2.warpPerspective(img, M, (width, height))

        text, confidence = recognizer.recognize(cropped)

        results.append({
            "annotation_index": region.get("annotation_index"),
            "expected_text": region.get("text"),
            "recognized_text": text,
            "confidence": confidence
        })

    return results

if __name__ == "__main__":
    fail_records = fetch_and_save_data()

    for idx, record in enumerate(fail_records[:5]):
        print(f"\n=== Record {idx+1}/{len(fail_records)} ===")
        print(f"Recipe: {record['recipe_name']}")
        print(f"Camera: {record['serial_number']}")
        print(f"Frame: {record['frame']['frame_idx']}, Template: {record['frame']['template_name']}")

        img_path = record['frame'].get('image_path')
        if not img_path:
            print("No image path available")
            continue

        img = download_image(img_path)

        detected_regions = record['frame'].get('detected_regions', [])
        vis_img = visualize_regions(img, detected_regions)

        output_vis = IMAGES_DIR / f"vis_{idx}_{img_path.split('/')[-1]}"
        cv2.imwrite(str(output_vis), vis_img)
        print(f"Saved visualization to {output_vis}")

        ocr_results = crop_and_recognize_text(img, detected_regions)
        for result in ocr_results:
            print(f"  Annotation {result['annotation_index']}: '{result['expected_text']}' -> '{result['recognized_text']}' (conf: {result['confidence']:.3f})")
