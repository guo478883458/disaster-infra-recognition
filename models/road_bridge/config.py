"""
道路/桥梁损毁检测模型配置

数据源：RDD2020（三国道路损毁数据集）
  - train/: Czech(2829) + India(7706) + Japan(10506) = 21041 张
  - test1/: Czech(349) + India(969) + Japan(1313) = 2631 张
  - test2/: Czech(360) + India(990) + Japan(1314) = 2664 张
  - 标注: <country>/annotations/xmls/*.xml (PASCAL VOC 格式)
  - 类别: D00 纵向裂缝, D10 横向裂缝, D20 龟裂, D40 坑洼
"""
import os

# 数据集根目录
DATA_DIR = r"H:\dev\disaster-data\infra_datasets\rdd2020"

# 权重保存路径（不进 git）
MODEL_WEIGHTS_DIR = r"H:\dev\disaster-data\models\infra\road_bridge"

# RDD2020 类别映射（4 类，与 RDD2022 一致）
RDD2020_CLASSES = {
    0: "D00",   # 纵向裂缝（Longitudinal crack）
    1: "D10",   # 横向裂缝（Transverse crack）
    2: "D20",   # 龟裂（Alligator crack）
    3: "D40",   # 坑洼（Pothole）
}
RDD2020_CLASSES_CN = {
    "D00": "纵向裂缝",
    "D10": "横向裂缝",
    "D20": "龟裂",
    "D40": "坑洼",
}
# 反向映射 name → id
RDD2020_NAME_TO_ID = {v: k for k, v in RDD2020_CLASSES.items()}

# YOLO 格式标签输出目录（在数据目录下创建，只读数据不修改，写 YOLO 标签到此处）
YOLO_LABELS_DIR = os.path.join(DATA_DIR, "yolo_labels")

# 微调后的权重路径
FINETUNED_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, "rdd2020_yolov8n.pt")

# 推理参数
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def get_country_paths(split: str = "train", country: str = None):
    """
    获取指定 split 和 country 的目录路径

    Args:
        split: "train" / "test1" / "test2"
        country: "Czech" / "India" / "Japan" / None（全部）

    Returns:
        [(country, image_dir, annotation_dir), ...]
    """
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        return []

    countries = [country] if country else [
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ]

    result = []
    for c in countries:
        img_dir = os.path.join(split_dir, c, "images")
        ann_dir = os.path.join(split_dir, c, "annotations", "xmls")
        if os.path.isdir(img_dir) and os.path.isdir(ann_dir):
            result.append((c, img_dir, ann_dir))
    return result