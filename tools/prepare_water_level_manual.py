"""
水位尺人工标注 → YOLO 格式转换

输入：labelme JSON（rectangle 框，类名 digit）
  H:\dev\disaster-data\infra_datasets\water_level\extracted\SAM_water_level_Dataset\yolo_labels_manual\train\images\

输出：YOLO 格式（class_id=0, 归一化坐标），8:2 train/val 划分
  yolo_labels_manual/dataset/images/{train,val}/
  yolo_labels_manual/dataset/labels/{train,val}/
  yolo_labels_manual/dataset_water_level_manual.yaml
"""
import os
import json
import random
import shutil
import yaml

# 数据路径（只读 H 盘）
MANUAL_DIR = r"H:\dev\disaster-data\infra_datasets\water_level\extracted\SAM_water_level_Dataset\yolo_labels_manual"
INPUT_DIR = os.path.join(MANUAL_DIR, "train", "images")
OUTPUT_DIR = os.path.join(MANUAL_DIR, "dataset")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
CLASS_NAME = "digit"


def convert_labelme_to_yolo(labelme_path: str, img_w: int, img_h: int) -> list:
    """
    将单个 labelme JSON 转为 YOLO 检测格式行

    labelme rectangle 格式：
      shape_type="rectangle", points=[[x1,y1],[x2,y2]]  (左上, 右下)

    返回: [class_id x_center y_center width height, ...]
    """
    with open(labelme_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    yolo_lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape.get("label", "")
        if label != CLASS_NAME:
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # YOLO 归一化
        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        bw = (x_max - x_min) / img_w
        bh = (y_max - y_min) / img_h

        # 过滤无效框
        if bw <= 0 or bh <= 0 or bw > 1 or bh > 1:
            continue

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    return yolo_lines


def main():
    random.seed(RANDOM_SEED)

    # 收集所有 JSON 文件
    json_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])
    print(f"找到 {len(json_files)} 个 labelme JSON 文件")

    # 统计总标注数
    total_boxes = 0
    for jf in json_files:
        with open(os.path.join(INPUT_DIR, jf), "r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            if shape.get("label") == CLASS_NAME and shape.get("shape_type") == "rectangle":
                total_boxes += 1
    print(f"总计 {total_boxes} 个数字框标注")

    # 随机打乱并划分
    random.shuffle(json_files)
    n = len(json_files)
    n_train = int(n * TRAIN_RATIO)
    train_files = json_files[:n_train]
    val_files = json_files[n_train:]

    print(f"划分: train={len(train_files)}, val={len(val_files)}")

    # 创建输出目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # 转换并复制
    stats = {"train": {"converted": 0, "boxes": 0}, "val": {"converted": 0, "boxes": 0}}

    for split, files in [("train", train_files), ("val", val_files)]:
        for jf in files:
            base = os.path.splitext(jf)[0]
            json_path = os.path.join(INPUT_DIR, jf)
            img_path = os.path.join(INPUT_DIR, f"{base}.jpg")

            if not os.path.exists(img_path):
                # 尝试 .png
                img_path = os.path.join(INPUT_DIR, f"{base}.png")
                if not os.path.exists(img_path):
                    print(f"  [跳过] 找不到图像: {base}")
                    continue

            # 读取 JSON 获取图像尺寸
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            img_w = data.get("imageWidth", 0)
            img_h = data.get("imageHeight", 0)
            if img_w == 0 or img_h == 0:
                print(f"  [跳过] 无效图像尺寸: {base}")
                continue

            # 转换
            yolo_lines = convert_labelme_to_yolo(json_path, img_w, img_h)
            if not yolo_lines:
                print(f"  [跳过] 无有效标注: {base}")
                continue

            # 复制图像
            out_img = os.path.join(OUTPUT_DIR, "images", split, f"{base}.jpg")
            shutil.copy2(img_path, out_img)

            # 写入标签
            txt_path = os.path.join(OUTPUT_DIR, "labels", split, f"{base}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))

            stats[split]["converted"] += 1
            stats[split]["boxes"] += len(yolo_lines)

    print(f"\n转换完成:")
    print(f"  train: {stats['train']['converted']} 张, {stats['train']['boxes']} 框")
    print(f"  val:   {stats['val']['converted']} 张, {stats['val']['boxes']} 框")

    # 创建 dataset YAML
    yaml_path = os.path.join(MANUAL_DIR, "dataset_water_level_manual.yaml")
    data_yaml = {
        "path": OUTPUT_DIR,
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [CLASS_NAME],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"\nYAML 已创建: {yaml_path}")
    print(f"  内容: path={OUTPUT_DIR}, train=images/train, val=images/val, nc=1, names=[digit]")

    return yaml_path


if __name__ == "__main__":
    main()