# 基础设施灾损视觉评估

从现场图片识别基础设施灾损：河道水位、道路/桥梁损毁。

## 项目结构

```
.
├── models/
│   ├── water_level/          # 河道水位识别
│   │   ├── __init__.py
│   │   ├── config.py         # 配置
│   │   └── inference.py      # 推理管线：检测→OCR→水位计算
│   └── road_bridge/          # 道路/桥梁损毁检测
│       ├── __init__.py
│       ├── config.py         # 配置
│       └── inference.py      # YOLOv8 推理
├── tools/
│   ├── infer_api.py          # 统一推理接口
│   └── train.py              # 训练入口
├── requirements.txt
└── README.md
```

## 子模型

### 1. 河道水位识别 (`water_level`)

参考 [GaoKangYu/Water-Level-Recognition-With-OCR](https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR) 方案：

- **数字检测**：YOLOv8 检测水位尺数字区域
- **OCR 读数**：PaddleOCR 识别数字值
- **刻度线检测**：霍夫变换检测刻度线（精度 0.01）
- **水位计算**：`L1 - (10 - L2 * 0.1) - 0.25 * L3`

### 2. 道路损毁检测 (`road_bridge`)

基于 Ultralytics YOLOv8n，在 RDD2022 数据集训练：

| 类别 | 代码 | 说明 |
|------|------|------|
| D00 | Longitudinal crack | 纵向裂缝 |
| D10 | Transverse crack | 横向裂缝 |
| D20 | Alligator crack | 龟裂 |
| D40 | Pothole | 坑洼 |

## 统一推理接口

```python
from tools.infer_api import process_infra_image

# 水位识别
result = process_infra_image("image.jpg", task="water_level")

# 道路损毁检测
result = process_infra_image("image.jpg", task="road")
```

命令行：
```bash
python tools/infer_api.py image.jpg --task water_level
python tools/infer_api.py image.jpg --task road
```

## 数据

数据只读引用 H 盘，不拷贝到项目目录：

- 水位尺：`H:\dev\disaster-data\infra_datasets\water_level\`
- 道路损毁：`H:\dev\disaster-data\infra_datasets\rdd2022\`

权重保存至 `H:\dev\disaster-data\models\infra\`（不进 git）。

## 环境

```bash
D:\ana\envs\disasterlex\python.exe -m pip install -r requirements.txt
```