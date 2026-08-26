"""
洪水/低洼积水分割：基于 YOLOv8n-seg 的洪水分割模型

权重来源：v2 项目 FloodNet 训练（flood.pt）
输出：积水面积_m2、淹没占比、灾情等级

旧逻辑（FloodIMG 10 类检测）保留为 detect_objects() 备用方法
"""
import os
import time
import cv2
import numpy as np

from .config import (
    FLOOD_MODEL_WEIGHT, PIXEL_AREA_M2,
    DISASTER_LEVELS, FLOOD_CLASSES, FLOOD_CLASSES_CN,
)


def get_disaster_level(area_m2: float) -> str:
    for low, high, label in DISASTER_LEVELS:
        if low <= area_m2 < high:
            return label
    return "未知"


class FloodDetector:
    """洪水检测/分割器——默认使用分割模型，detect_objects 为备用"""

    def __init__(self, weights: str = None, device: str = "cpu"):
        self.device = device
        if weights is None:
            weights = FLOOD_MODEL_WEIGHT
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"未找到洪水分割权重: {weights}\n"
                f"请确保已下载 flood.pt 到 H:\\dev\\disaster-data\\models\\"
            )
        self._load_model(weights)

    def _load_model(self, weights: str):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.model.to(self.device)

    def predict(self, image_path: str) -> dict:
        return self.infer(image_path)

    def infer(self, image_path: str) -> dict:
        """洪水分割推理（主方法）"""
        t0 = time.time()
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_height, img_width = img.shape[:2]

        results = self.model(
            image_path, conf=0.25, verbose=False, imgsz=320, max_det=10,
        )

        total_mask_pixels = 0
        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            combined = np.zeros((img_height, img_width), dtype=np.uint8)
            for mask in masks:
                mask_resized = cv2.resize(mask, (img_width, img_height))
                combined[mask_resized > 0.5] = 1
            total_mask_pixels = int(combined.sum())

        total_area_m2 = total_mask_pixels * PIXEL_AREA_M2
        total_image_pixels = img_height * img_width
        inundation_ratio = total_mask_pixels / total_image_pixels if total_image_pixels > 0 else 0.0
        disaster_level = get_disaster_level(total_area_m2)
        inference_time_ms = round((time.time() - t0) * 1000, 2)

        return {
            "积水面积_m2": round(total_area_m2, 2),
            "淹没占比": round(inundation_ratio, 6),
            "灾情等级": disaster_level,
            "mask_pixels": total_mask_pixels,
            "image_size": f"{img_width}x{img_height}",
            "inference_time_ms": inference_time_ms,
            "task": "flood",
        }

    def detect_objects(self, image_path: str) -> dict:
        """物体检测（备用方法）"""
        t0 = time.time()
        results = self.model.predict(
            source=image_path, conf=0.25, iou=0.45,
            device=self.device, verbose=False,
        )[0]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = FLOOD_CLASSES.get(cls_id, f"unknown_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "class_name_cn": FLOOD_CLASSES_CN.get(cls_name, cls_name),
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in xyxy],
                })
        inference_time = (time.time() - t0) * 1000
        return {
            "task": "flood_detection",
            "detections": detections,
            "num_detections": len(detections),
            "image_path": image_path,
            "inference_time_ms": round(inference_time, 2),
        }
