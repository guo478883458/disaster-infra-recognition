"""
FloodIMG: labelme JSON → YOLO 检测格式转换

将 25 张标注图像的 labelme JSON 多边形标注转为 YOLO 格式 txt
输出目录: H:\dev\disaster-data\image_datasets\floodimg\yolo_labels\
"""
import os
import json
import glob
import numpy as np


# ---- 配置 ----
IMG_DIR = r"H:\dev\disaster-data\image_datasets\floodimg\Flood Images"
ANN_DIR = r"H:\dev\disaster-data\image_datasets\floodimg\Annotation"
OUT_DIR = r"H:\dev\disaster-data\image_datasets\floodimg\yolo_labels"

# 类别映射（按名称排序，保持稳定）
CLASSES = sorted([
    "Person", "Tree", "House", "Builbing", "Traffic Sign",
    "Bridge", "Forest", "Car", "Truck", "Road"
])
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

print(f"  类别: {CLASSES}")


def polygon_to_bbox(points):
    """多边形顶点 → 归一化 YOLO bbox (cx, cy, w, h)"""
    pts = np.array(points, dtype=np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def convert_json(json_path, img_w, img_h):
    """转换单个 JSON → YOLO txt 行列表"""
    with open(json_path, "r") as f:
        data = json.load(f)

    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in CLASS_TO_ID:
            print(f"    [警告] 未知标签 '{label}'，跳过")
            continue
        class_id = CLASS_TO_ID[label]
        points = shape["points"]  # [[x1,y1], [x2,y2], ...]

        # 多边形 → bbox → 归一化
        cx, cy, bw, bh = polygon_to_bbox(points)
        cx_norm = cx / img_w
        cy_norm = cy / img_h
        bw_norm = bw / img_w
        bh_norm = bh / img_h

        # 裁剪到 [0,1]
        cx_norm = max(0, min(1, cx_norm))
        cy_norm = max(0, min(1, cy_norm))
        bw_norm = max(0, min(1, bw_norm))
        bh_norm = max(0, min(1, bh_norm))

        lines.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}")

    return lines


def main():
    print("=" * 60)
    print("FloodIMG: labelme JSON → YOLO 检测格式")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # 查找所有 JSON
    jsons = sorted(glob.glob(os.path.join(ANN_DIR, "*.json")))
    print(f"  找到 JSON 标注: {len(jsons)} 个")

    converted = 0
    skipped = 0

    for jp in jsons:
        # 读取 JSON 获取图像信息
        with open(jp, "r") as f:
            data = json.load(f)

        img_name = data.get("imagePath", "")
        img_w = data.get("imageWidth", 0)
        img_h = data.get("imageHeight", 0)

        if img_w == 0 or img_h == 0:
            print(f"  [跳过] {jp}: 无图像尺寸信息")
            skipped += 1
            continue

        # 写入 YOLO txt
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(OUT_DIR, txt_name)

        lines = convert_json(jp, img_w, img_h)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        print(f"  [转换] {img_name} → {txt_name} ({len(lines)} 个标注)")
        converted += 1

    print(f"\n  转换完成: {converted} 张, 跳过: {skipped} 张")
    print(f"  输出目录: {OUT_DIR}")

    # 统计
    total_labels = 0
    for f in os.listdir(OUT_DIR):
        if f.endswith(".txt"):
            with open(os.path.join(OUT_DIR, f)) as fh:
                total_labels += len(fh.readlines())
    print(f"  总标注数: {total_labels}")
    print(f"  类别数: {len(CLASSES)}")


if __name__ == "__main__":
    main()
