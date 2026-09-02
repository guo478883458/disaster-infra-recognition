# 基础设施灾损视觉评估

从现场图片识别基础设施灾损：河道水位、道路损毁、洪水分割。本项目专注视觉识别模型，识别结果通过 v2 灾害链推理引擎映射为贝叶斯网络证据，驱动灾害链推理。

---

## 项目结构

```
.
├── models/
│   ├── water_level/          # 河道水位识别
│   │   ├── __init__.py
│   │   ├── config.py         # 配置（路径从 infra_path_config 导入）
│   │   └── inference.py      # 推理管线：YOLO 数字检测 → EasyOCR 读数 → 水位计算
│   ├── road_bridge/          # 道路/桥梁损毁检测
│   │   ├── __init__.py
│   │   ├── config.py         # 配置
│   │   └── inference.py      # YOLOv8n 推理（RDD2020 训练）
│   └── flood_detection/      # 洪水/低洼积水分割
│       ├── __init__.py
│       ├── config.py         # 配置
│       └── inference.py      # YOLOv8n-seg 分割推理（flood_rescuenet.pt）
├── tools/
│   ├── infer_api.py          # 统一推理接口（water_level/road/flood）
│   ├── train.py              # 训练入口
│   ├── self_test.py          # 自测
│   └── prepare_*.py          # 数据集准备脚本
├── infra_path_config.py      # 路径配置（三级回退，与 v2 的 path_config.py 独立）
├── requirements.txt
└── README.md
```

---

## 子模型

### 1. 河道水位识别 (`water_level`)

参考 [GaoKangYu/Water-Level-Recognition-With-OCR](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR) 方案，标注数据为 labelme 人工标注 167 张/846 框后 GPU 重训：

- **数字检测**：YOLOv8n 检测水位尺数字区域（mAP50 0.966）
- **OCR 读数**：**EasyOCR**（非 PaddleOCR）识别数字值
- **水位计算**：`水位 = 1.0 + N × 0.1 米`（N = 最小可见数字，过滤 0 与两位数误检）
- **精度**：MAE 9.57cm

### 2. 道路损毁检测 (`road_bridge`)

基于 Ultralytics YOLOv8n，在 RDD2020 数据集训练：

| 类别 | 代码 | 说明 |
|------|------|------|
| D00 | Longitudinal crack | 纵向裂缝 |
| D10 | Transverse crack | 横向裂缝 |
| D20 | Alligator crack | 龟裂 |
| D40 | Pothole | 坑洼 |

### 3. 洪水分割 (`flood_detection`)

基于 YOLOv8n-seg，在 RescueNet 数据集训练（mAP50 0.834）：

- **模型权重**：`flood_rescuenet.pt`
- **输出**：积水面积（m²）、淹没占比、灾情等级（轻度/中度/重度/未知）

---

## 统一推理接口

```python
from tools.infer_api import process_infra_image

# 水位识别
result = process_infra_image("image.jpg", task="water_level")

# 道路损毁检测
result = process_infra_image("image.jpg", task="road")

# 洪水分割
result = process_infra_image("image.jpg", task="flood")
```

命令行：

```bash
python tools/infer_api.py image.jpg --task water_level
python tools/infer_api.py image.jpg --task road
python tools/infer_api.py image.jpg --task flood
```

---

## 配合 v2 引擎使用

本项目的识别结果通过 v2 灾害链推理引擎的 `fuse_infer.py` 接入，映射为贝叶斯网络证据：

| 识别任务 | 输出字段 | BN 证据节点 |
|----------|----------|-------------|
| 水位读数 | `water_level_cm` | 河道水位、内涝深度 |
| 道路损毁 | `damage_counts` | 道路积水历史频率 |
| 洪水分割 | `flood_area_m2` | 内涝深度 |
| 滑坡分割（v2 侧复用） | `landslide_area_m2` | 滑坡历史密度 |

使用方式：

```bash
# 在 v2 项目目录下，通过 fuse_infer 统一调用
python -m tools.fuse_infer --tasks "图片.jpg:water_level"
python -m tools.fuse_infer --tasks "图片.jpg:flood"
python -m tools.fuse_infer --tasks "图片.jpg:landslide"
```

---

## 数据

数据只读引用 H 盘，不拷贝到项目目录。路径通过 `infra_path_config.py` 三级回退解析：

1. **环境变量** `DISASTER_DATA_DIR`
2. **原 H 盘路径** `H:\dev\disaster-data`
3. **包内 data/ 目录**（回退选项）

权重保存至 `H:\dev\disaster-data\models\infra\`（不进 git）：

- `water_level_digit_detector.pt`（水位数字检测）
- `rdd2020_yolov8n.pt`（道路损毁检测）
- `flood_rescuenet.pt`（洪水分割）

---

## 环境

```bash
conda create -n disasterlex python=3.10
conda activate disasterlex
pip install -r requirements.txt
```

> **注意**：`tifffile` 必须装入 conda 环境而非用户目录（`PYTHONNOUSERSITE=1` 下用户级包不可见，会导致后台线程 easyocr 导入失败）。