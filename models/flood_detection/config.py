"""
洪水/低洼积水分割模型配置

主模型：v2 项目 RescueNet 训练的洪水分割模型（flood_rescuenet.pt，mAP50 0.834）
备用数据：FloodIMG（仅 25 张标注，10 类通用物体，无 Flood 类）
          RescueNet（无人机洪水分割，2941 对，CC BY-NC-ND）
          STURM-Flood（遥感洪水，S1 完整 S2 不完整，本次不训练）

许可备注：
  - RescueNet: CC BY-NC-ND（非商用，训练前需确认用途）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from path_config import (
    MODELS_DIR, FLOODIMG_DIR, RESCUENET_DIR,
    INFRA_FLOOD_DIR, FLOOD_MODEL_WEIGHT,
)

# ==================== 分割模型权重 ====================
FLOOD_MODEL_WEIGHT = FLOOD_MODEL_WEIGHT

# ==================== 面积换算常量 ====================
PIXEL_RESOLUTION_M = 0.05
PIXEL_AREA_M2 = PIXEL_RESOLUTION_M ** 2  # 0.0025 m²/px

# 灾情等级阈值（积水面积 m²）
DISASTER_LEVELS = [
    (0, 100, "无"),
    (100, 500, "轻度"),
    (500, 2000, "中度"),
    (2000, 5000, "重度"),
    (5000, float("inf"), "严重"),
]

# ==================== 数据目录 ====================
DATA_DIR = FLOODIMG_DIR
IMAGES_DIR = os.path.join(DATA_DIR, "Flood Images")
YOLO_LABELS_DIR = os.path.join(DATA_DIR, "yolo_labels")
RESCUE_DIR = RESCUENET_DIR
RESCUE_YOLO_LABELS_DIR = os.path.join(RESCUE_DIR, "yolo_seg_labels")
MODEL_WEIGHTS_DIR = INFRA_FLOOD_DIR

# ==================== 备用检测类别（FloodIMG 10 类） ====================
FLOOD_CLASSES = {
    0: "Bridge", 1: "Builbing", 2: "Car", 3: "Forest", 4: "House",
    5: "Person", 6: "Road", 7: "Traffic Sign", 8: "Tree", 9: "Truck",
}
FLOOD_CLASSES_CN = {
    "Bridge": "桥梁", "Builbing": "建筑物", "Car": "汽车",
    "Forest": "森林", "House": "房屋", "Person": "行人",
    "Road": "道路", "Traffic Sign": "交通标志", "Tree": "树木", "Truck": "卡车",
}
FLOOD_NAME_TO_ID = {v: k for k, v in FLOOD_CLASSES.items()}
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45