"""
路径配置模块（基础设施灾损识别项目）
====================================
统一管理所有数据/权重路径，实现路径可移植化。

优先级（三级回退）：
  1. 环境变量 DISASTER_DATA_DIR（显式覆盖）
  2. 原 H 盘路径（本机兼容，零影响）
  3. 包内 data/ 目录（新机器）

使用方式：
    from path_config import (
        DATA_ROOT, MODELS_DIR,
        WATER_LEVEL_DATA_DIR, RDD2020_DIR, ...
    )
"""

import os
import sys

# ── 项目根目录 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 数据根目录：三级回退 ──
_ORIG_H = r"H:\dev\disaster-data"
_env_data = os.environ.get("DISASTER_DATA_DIR")
if _env_data:
    DATA_ROOT = _env_data
elif os.path.exists(_ORIG_H):
    DATA_ROOT = _ORIG_H
else:
    # 发布包中，基础设施在 infra_recognition/，data 在上级
    DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "data")

# ==================== 模型权重路径 ====================
MODELS_DIR = os.path.join(DATA_ROOT, "models")

# 基础设施模型权重子目录
INFRA_WATER_LEVEL_DIR = os.path.join(MODELS_DIR, "infra", "water_level")
INFRA_ROAD_BRIDGE_DIR = os.path.join(MODELS_DIR, "infra", "road_bridge")
INFRA_FLOOD_DIR = os.path.join(MODELS_DIR, "infra", "flood_detection")

# 权重文件
WATER_LEVEL_DIGIT_PT = os.path.join(INFRA_WATER_LEVEL_DIR, "water_level_digit_detector.pt")
ROAD_BRIDGE_PT = os.path.join(INFRA_ROAD_BRIDGE_DIR, "rdd2020_yolov8n.pt")
FLOOD_MODEL_WEIGHT = os.path.join(MODELS_DIR, "flood_rescuenet.pt")

# ==================== 基础设施数据目录 ====================
WATER_LEVEL_DATA_DIR = os.path.join(
    DATA_ROOT, "infra_datasets", "water_level", "extracted", "SAM_water_level_Dataset"
)
WATER_LEVEL_IMAGES_DIR = os.path.join(WATER_LEVEL_DATA_DIR, "Staff gauge images")
WATER_LEVEL_XLSX = os.path.join(
    WATER_LEVEL_DATA_DIR, "In-situ water levels", "In-situ & simulated water levels.xlsx"
)
RDD2020_DIR = os.path.join(DATA_ROOT, "infra_datasets", "rdd2020")

# ==================== 训练数据集 ====================
FLOODIMG_DIR = os.path.join(DATA_ROOT, "image_datasets", "floodimg")
RESCUENET_DIR = os.path.join(DATA_ROOT, "image_datasets", "rescuenet", "segmentation-trainset")