"""
水位尺数字框标注生成（弱监督：EasyOCR）

管线：
  1. EasyOCR 检测水位尺图像中的数字区域 → 取 bbox
  2. 过滤低置信度框（< 0.3）
  3. 输出 YOLO 标准格式 labels + images（按日期目录 8:2 切分 train/val）
  4. 抽检可视化

用法：
    python tools/prepare_water_level_labels.py                     # 全量生成
    python tools/prepare_water_level_labels.py --sample 30         # 仅抽检 30 张可视化
    python tools/prepare_water_level_labels.py --max_images 500    # 只处理 500 张（测试用）
"""
import os
import sys
import argparse
import shutil
import random
import cv2
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.water_level.config import IMAGES_DIR, DATA_DIR

# 输出目录
YOLO_BASE = os.path.join(DATA_DIR, "yolo_labels")
TRAIN_IMAGES = os.path.join(YOLO_BASE, "train", "images")
TRAIN_LABELS = os.path.join(YOLO_BASE, "train", "labels")
VAL_IMAGES = os.path.join(YOLO_BASE, "val", "images")
VAL_LABELS = os.path.join(YOLO_BASE, "val", "labels")
SAMPLE_VIS_DIR = os.path.join(YOLO_BASE, "sample_vis")

# EasyOCR 参数
OCR_CONF_THRESHOLD = 0.3
TRAIN_RATIO = 0.8


def get_all_images():
    """扫描所有日期目录下的图像"""
    image_paths = []
    for date_dir in sorted(os.listdir(IMAGES_DIR)):
        img_dir = os.path.join(IMAGES_DIR, date_dir, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(img_dir, fname))
    return image_paths


def split_by_date():
    """按日期目录 8:2 切分 train/val，避免同日期泄漏"""
    date_dirs = sorted([
        d for d in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, d, "images"))
    ])
    random.seed(42)
    random.shuffle(date_dirs)
    split_idx = max(1, int(len(date_dirs) * TRAIN_RATIO))
    train_dates = set(date_dirs[:split_idx])
    val_dates = set(date_dirs[split_idx:])

    train_images, val_images = [], []
    for dd in date_dirs:
        img_dir = os.path.join(IMAGES_DIR, dd, "images")
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(img_dir, fname)
                if dd in train_dates:
                    train_images.append(path)
                else:
                    val_images.append(path)
    return train_images, val_images, train_dates, val_dates


def easyocr_digits(image_path: str, reader) -> list:
    """
    用 EasyOCR 检测数字区域，返回 YOLO 格式标注列表

    Returns:
        [(class_id, x_center, y_center, w, h), ...]  # class_id 始终为 0
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    img_h, img_w = img.shape[:2]

    try:
        results = reader.readtext(image_path)
    except Exception:
        # 跳过损坏图片
        return []
    yolo_boxes = []
    for bbox, text, conf in results:
        if conf < OCR_CONF_THRESHOLD:
            continue
        # bbox 是 4 个角点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 转 YOLO 格式
        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        # 裁剪到 [0,1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        yolo_boxes.append((0, x_center, y_center, w, h, text, conf))
    return yolo_boxes


def generate_labels(image_paths: list, output_images_dir: str,
                    output_labels_dir: str, reader, max_images: int = None):
    """
    批量生成 YOLO 标签

    Returns:
        (processed, with_detections, total_boxes)
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    processed = 0
    with_detections = 0
    total_boxes = 0

    if max_images:
        image_paths = image_paths[:max_images]

    for img_path in image_paths:
        processed += 1
        if processed % 100 == 0:
            print(f"    已处理 {processed}/{len(image_paths)} ...")

        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        dst_img = os.path.join(output_images_dir, fname)
        label_path = os.path.join(output_labels_dir, f"{base_name}.txt")

        # 断点续跑：图像已复制过则跳过（无论上次是否检出数字框）
        if os.path.exists(dst_img):
            if os.path.exists(label_path):
                with_detections += 1
                with open(label_path) as f:
                    total_boxes += len(f.read().splitlines())
            continue

        # 复制图像
        shutil.copy2(img_path, dst_img)

        # OCR 检测（损坏图跳过，不中断整个批次）
        try:
            boxes = easyocr_digits(img_path, reader)
        except Exception as e:
            print(f"    [跳过损坏图] {fname}: {e}")
            continue
        if boxes:
            with_detections += 1
            total_boxes += len(boxes)
            lines = [f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                     for cls_id, x, y, w, h, _, _ in boxes]
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

    return processed, with_detections, total_boxes


def visualize_samples(image_paths: list, reader, n: int = 30):
    """抽检 n 张图，画框保存到 sample_vis 目录"""
    os.makedirs(SAMPLE_VIS_DIR, exist_ok=True)
    sampled = random.sample(image_paths, min(n, len(image_paths)))

    total_detected = 0
    for i, img_path in enumerate(sampled):
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = easyocr_digits(img_path, reader)
        if boxes:
            total_detected += 1

        for cls_id, xc, yc, w, h, text, conf in boxes:
            img_h, img_w = img.shape[:2]
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{text} ({conf:.2f})"
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(SAMPLE_VIS_DIR, f"{i:02d}_{fname}")
        cv2.imwrite(out_path, img)
        status = f" {len(boxes)} boxes" if boxes else " no detections"
        print(f"  [{i+1}/{len(sampled)}] {fname}:{status}")

    print(f"\n抽检 {len(sampled)} 张，有检出: {total_detected}/{len(sampled)}"
          f" ({total_detected/len(sampled)*100:.1f}%)")
    print(f"可视化保存到: {SAMPLE_VIS_DIR}")
    return total_detected, len(sampled)


def create_dataset_yaml():
    """创建 YOLO 训练用的 dataset_water_level.yaml"""
    yaml_path = os.path.join(YOLO_BASE, "dataset_water_level.yaml")
    data = {
        "path": YOLO_BASE,
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["digit"],
    }
    import yaml as ymlib
    with open(yaml_path, "w") as f:
        ymlib.dump(data, f, default_flow_style=False)
    print(f"  YAML 已创建: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="水位尺数字框标注生成（EasyOCR 弱监督）")
    parser.add_argument("--sample", type=int, default=0,
                        help="仅抽检 N 张可视化，不生成完整标签")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最多处理 N 张（测试用）")
    args = parser.parse_args()

    print("=" * 60)
    print("水位尺数字框标注生成（EasyOCR 弱监督）")
    print("=" * 60)

    all_images = get_all_images()
    print(f"  总图像数: {len(all_images)}")

    # 按日期划分 train/val
    train_imgs, val_imgs, train_dates, val_dates = split_by_date()
    print(f"  Train 日期: {len(train_dates)} 个目录 ({len(train_imgs)} 张)")
    print(f"  Val 日期: {len(val_dates)} 个目录 ({len(val_imgs)} 张)")

    # 初始化 EasyOCR（只加载一次）
    print("\n  初始化 EasyOCR ...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)

    if args.sample > 0:
        print(f"\n  抽检可视化: {args.sample} 张")
        visualize_samples(train_imgs + val_imgs, reader, args.sample)
        return

    # 生成训练集标签
    print("\n  生成训练集标签 ...")
    train_processed, train_det, train_boxes = generate_labels(
        train_imgs, TRAIN_IMAGES, TRAIN_LABELS, reader, args.max_images
    )

    # 生成验证集标签（val 不做截断，保证 8:2 划分完整）
    print("\n  生成验证集标签 ...")
    val_processed, val_det, val_boxes = generate_labels(
        val_imgs, VAL_IMAGES, VAL_LABELS, reader, None
    )

    # 汇总
    total_processed = train_processed + val_processed
    total_det = train_det + val_det
    total_boxes = train_boxes + val_boxes
    print(f"\n  {'='*50}")
    print(f"  标注完成")
    print(f"    处理图像: {total_processed}")
    print(f"    有检出: {total_det} ({total_det/total_processed*100:.1f}%)")
    print(f"    总框数: {total_boxes}")
    print(f"    平均框数/有检出图: {total_boxes/max(total_det,1):.1f}")
    print(f"  Train labels: {TRAIN_LABELS}")
    print(f"  Val labels: {VAL_LABELS}")

    # 创建 YAML
    yaml_path = create_dataset_yaml()
    print(f"\n  训练命令: python tools/train.py --model water_level --data_yaml {yaml_path}")

    # 抽检 30 张可视化
    print(f"\n  抽检 30 张可视化 ...")
    visualize_samples(val_imgs, reader, 30)

    print(f"\n  {'='*50}")
    print(f"  结论: 请检查 {SAMPLE_VIS_DIR} 下的可视化结果")
    print(f"  若检出率 > 70% 则直接训练，否则改用人工标注方案")
    print(f"  {'='*50}")


if __name__ == "__main__":
    main()