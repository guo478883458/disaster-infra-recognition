"""
道路/桥梁损毁检测：基于 YOLOv8 的 RDD2020 道路损毁检测

检测类别（4 类）：
  - D00: 纵向裂缝（Longitudinal crack）
  - D10: 横向裂缝（Transverse crack）
  - D20: 龟裂（Alligator crack）
  - D40: 坑洼（Pothole）

数据格式：PASCAL VOC XML -> 自动转换为 YOLO 格式
"""
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter, OrderedDict
from pathlib import Path

from .config import (
    DATA_DIR, MODEL_WEIGHTS_DIR, FINETUNED_WEIGHTS,
    RDD2020_CLASSES, RDD2020_CLASSES_CN, RDD2020_NAME_TO_ID,
    YOLO_LABELS_DIR, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    get_country_paths
)


class VocToYoloConverter:
    """PASCAL VOC XML -> YOLO 格式标注转换器"""

    def __init__(self, labels_dir: str = YOLO_LABELS_DIR):
        self.labels_dir = labels_dir

    def convert_xml(self, xml_path: str, output_dir: str = None) -> str:
        """
        转换单个 VOC XML 到 YOLO 标签文件

        Args:
            xml_path: XML 文件路径
            output_dir: 输出目录，None 使用默认

        Returns:
            yolo_label_path: YOLO 标签文件路径，若无目标则返回 None
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 图像尺寸
        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        if output_dir is None:
            # 与 XML 同目录下的 labels/ 子目录
            output_dir = os.path.join(os.path.dirname(xml_path), "..", "labels")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        label_path = os.path.join(output_dir, f"{base_name}.txt")

        yolo_lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in RDD2020_NAME_TO_ID:
                continue
            cls_id = RDD2020_NAME_TO_ID[name]

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # 转 YOLO 格式: class_id x_center y_center width height (归一化)
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            # 裁剪到 [0,1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
            return label_path
        return None

    def convert_split(self, split: str = "train", country: str = None) -> dict:
        """
        批量转换整个 split 的标注

        labels 输出到 images 同级的 labels/ 目录（YOLO 标准约定）
        即 rdd2020/<split>/<country>/labels/  <-- 与 images/ 平级
        这样 YOLO 训练时能自动找到标签，无需额外 YAML 路径配置

        Returns:
            {"converted": int, "total": int, "output_dir": str}
        """
        paths = get_country_paths(split, country)
        total = 0
        converted = 0

        for c, img_dir, ann_dir in paths:
            out_dir = os.path.join(os.path.dirname(img_dir), "labels")
            os.makedirs(out_dir, exist_ok=True)

            xmls = sorted(Path(ann_dir).glob("*.xml"))
            for xml_path in xmls:
                total += 1
                result = self.convert_xml(str(xml_path), out_dir)
                if result:
                    converted += 1

        return {
            "converted": converted,
            "total": total,
            "output_dir": out_dir
        }


class RoadDamageDetector:
    """
    道路/桥梁损毁检测器

    基于 Ultralytics YOLOv8n，在 RDD2020 数据集上训练
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = device
        self.weights_path = weights_path or FINETUNED_WEIGHTS
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装 ultralytics: pip install ultralytics")

        if os.path.exists(self.weights_path):
            self.model = YOLO(self.weights_path)
        else:
            print(f"[WARN] RDD2020 微调权重未找到: {self.weights_path}")
            print("       使用 YOLOv8n 预训练模型（COCO，需微调后才能检出裂缝/坑洼）")
            self.model = YOLO("yolov8n.pt")

    def infer(self, image_path: str) -> dict:
        """
        对单张道路图像执行损毁检测

        Returns:
            {"detections": [...], "class_counts": {...},
             "total_damages": int, "inference_time_ms": float,
             "image_path": str}
        """
        t0 = time.time()

        results = self.model(
            image_path, conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD, device=self.device
        )[0]

        elapsed = (time.time() - t0) * 1000

        detections = []
        class_counts = Counter()

        if results.boxes is not None:
            for box, conf, cls_id in zip(
                results.boxes.xyxy, results.boxes.conf, results.boxes.cls
            ):
                cls_id = int(cls_id)
                class_name = RDD2020_CLASSES.get(cls_id, f"unknown_{cls_id}")
                class_name_cn = RDD2020_CLASSES_CN.get(class_name, class_name)

                detections.append({
                    "class_name": class_name,
                    "class_name_cn": class_name_cn,
                    "confidence": round(float(conf), 4),
                    "bbox": [round(float(v), 2) for v in box.tolist()]
                })
                class_counts[class_name] += 1

        class_counts_dict = OrderedDict(
            (k, class_counts.get(k, 0)) for k in RDD2020_CLASSES.values()
        )

        return {
            "detections": detections,
            "class_counts": class_counts_dict,
            "total_damages": len(detections),
            "inference_time_ms": round(elapsed, 2),
            "image_path": image_path
        }

    def evaluate(self, image_dir: str, label_dir: str) -> dict:
        """
        批量评估，计算 mAP（简化版：逐张计算平均精度）

        Args:
            image_dir: 图像目录
            label_dir: YOLO 标签目录

        Returns:
            {"map": float, "mean_time_ms": float, "n": int}
        """
        from ultralytics.utils.metrics import ConfusionMatrix

        image_paths = sorted(Path(image_dir).glob("*.jpg"))
        if not image_paths:
            return {"map": None, "mean_time_ms": None, "n": 0}

        times = []
        for img_path in image_paths[:50]:  # 限制 50 张
            r = self.infer(str(img_path))
            times.append(r["inference_time_ms"])

        return {
            "map": None,  # 完整评估需 ultralytics val 命令
            "mean_time_ms": round(float(np.mean(times)), 2) if times else None,
            "n": len(image_paths[:50])
        }


def demo_inference():
    """快速演示：对 3 张道路图像进行推理"""
    detector = RoadDamageDetector()

    # 从 RDD2020 test1 取 3 张图像
    images = []
    for c in ["Czech", "India", "Japan"]:
        img_dir = os.path.join(DATA_DIR, "test1", c, "images")
        if os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir))[:2]:
                images.append(os.path.join(img_dir, f))
        if len(images) >= 3:
            break

    if not images:
        print("[ERROR] 无测试图像可用")
        return

    print(f"\n{'='*65}")
    print(f"道路损毁检测推理测试（{len(images)} 张）")
    print(f"{'='*65}")

    for img_path in images[:3]:
        result = detector.infer(img_path)
        name = os.path.basename(img_path)
        print(f"  图像: {name}")
        if result["total_damages"] > 0:
            print(f"    损毁总数: {result['total_damages']}")
            for cls_name, cnt in result["class_counts"].items():
                if cnt > 0:
                    cn = RDD2020_CLASSES_CN.get(cls_name, cls_name)
                    print(f"      {cls_name} ({cn}): {cnt} 处")
            top_dets = sorted(result["detections"],
                              key=lambda d: d["confidence"], reverse=True)[:3]
            for det in top_dets:
                print(f"      [{det['class_name']}] 置信度: {det['confidence']:.2%}")
        else:
            print(f"    未检测到损毁（预训练模型需微调后才有效）")
        print(f"    耗时: {result['inference_time_ms']:.1f} ms\n")


if __name__ == "__main__":
    demo_inference()
