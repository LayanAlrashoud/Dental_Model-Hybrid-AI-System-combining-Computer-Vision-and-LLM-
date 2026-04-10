import os
import shutil
import cv2
from ultralytics import YOLO

def get_region(x_center, y_center, img_w, img_h):
    ratio = x_center / img_w

    if ratio < 0.20:
        return "far-left"
    elif ratio < 0.40:
        return "left"
    elif ratio < 0.60:
        return "center"
    elif ratio < 0.80:
        return "right"
    else:
        return "far-right"
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def deduplicate_detections_by_iou(detections, iou_threshold=0.4):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []

    for det in detections:
        keep_current = True

        for kept_det in kept:
            same_class = det["class_name"] == kept_det["class_name"]
            iou = calculate_iou(det["bbox"], kept_det["bbox"])

            if same_class and iou >= iou_threshold:
                keep_current = False
                break

        if keep_current:
            kept.append(det)

    return kept


def draw_single_detection(image, detection):
    output = image.copy()
    x1, y1, x2, y2 = detection["bbox"]
    label = f"{detection['class_name']} {detection['confidence']}"

    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
    cv2.putText(
        output,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return output


def draw_all_detections(image, detections):
    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f"{detection['class_name']} {detection['confidence']}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


def run_segmentation(model_path, image_path, output_original_folder, output_predicted_folder):
    os.makedirs(output_original_folder, exist_ok=True)
    os.makedirs(output_predicted_folder, exist_ok=True)

    model = YOLO(model_path)
    file_name = os.path.basename(image_path)
    file_stem, file_ext = os.path.splitext(file_name)

    original_save_path = os.path.join(output_original_folder, file_name)
    shutil.copy2(image_path, original_save_path)

    results = model.predict(source=image_path, save=False, conf=0.25)
    result = results[0]

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        img_h, img_w = result.orig_shape

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cls_id = int(classes[i])
            conf = float(confidences[i])

            if conf < 0.6:
                continue

            class_name = model.names[cls_id]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            region = get_region(x_center, y_center, img_w, img_h)

            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "region": region
            })

    detections = deduplicate_detections_by_iou(detections, iou_threshold=0.4)
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    base_image = cv2.imread(image_path)

    # صورة فيها كل النتائج
    predicted_all_filename = f"{file_stem}_all{file_ext}"
    predicted_all_path = os.path.join(output_predicted_folder, predicted_all_filename)
    all_image = draw_all_detections(base_image, detections)
    cv2.imwrite(predicted_all_path, all_image)

    # صور منفصلة، كل صورة فيها نتيجة وحدة
    single_detection_images = []

    for idx, detection in enumerate(detections):
        single_filename = f"{file_stem}_det_{idx + 1}{file_ext}"
        single_path = os.path.join(output_predicted_folder, single_filename)

        single_image = draw_single_detection(base_image, detection)
        cv2.imwrite(single_path, single_image)

        single_detection_images.append(f"outputs/predicted/{single_filename}")

    return {
        "original_filename": file_name,
        "predicted_all_filename": predicted_all_filename,
        "detections": detections,
        "single_detection_images": single_detection_images
    }