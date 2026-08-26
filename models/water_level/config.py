"""
河道水位识别模型配置

数据源：SAM_water_level_Dataset（中文水位尺场景，3315 张图像）
  - 图像: Staff gauge images/<YYYYMMDD>/images/<Image_name>.jpg
  - 真值: In-situ water levels/In-situ & simulated water levels.xlsx
    - Sheet "Total", 2691 行 × 13 列
    - Col B: Image_name（如 P23052700112410.jpg）
    - Col M: Gauge water level (cm) — 现场水位真值

参考 GaoKangYu/Water-Level-Recognition-With-OCR 方案：
  https://github.com/GaoKangYu/Water-Level-Recognition-With-OCR
  管线: YOLO 数字检测 → LSTM-OCR 读数 → 霍夫变换刻度线检测 → 水位计算
  水位公式: L1 - (10 - L2 * 0.1) - 0.25 * L3
"""
import os

# 数据集根目录（只读引用 H 盘）
DATA_DIR = r"H:\dev\disaster-data\infra_datasets\water_level\extracted\SAM_water_level_Dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "Staff gauge images")
XLSX_PATH = os.path.join(DATA_DIR, "In-situ water levels",
                         "In-situ & simulated water levels.xlsx")

# 权重保存路径（不进 git）
MODEL_WEIGHTS_DIR = r"H:\dev\disaster-data\models\infra\water_level"

# 预训练权重路径
PRETRAINED_WEIGHTS = os.path.join(MODEL_WEIGHTS_DIR, "water_level_digit_detector.pt")

# 推理参数
CONFIDENCE_THRESHOLD = 0.5
OCR_CONFIDENCE_THRESHOLD = 0.3


def get_image_path(image_name: str) -> str:
    """
    根据图像文件名定位完整路径

    文件名格式: P23052700112410.jpg -> YYMMDD = 230527
    目录结构: Staff gauge images/<YYYYMMDD>/images/<image_name>
    """
    import glob
    # 搜索所有日期目录
    pattern = os.path.join(IMAGES_DIR, "**", "images", image_name)
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    # 若文件名不含日期前缀，尝试从文件名解析
    # P23052700112410.jpg → 20230527
    if image_name.startswith("P") and len(image_name) >= 8:
        yymmdd = image_name[1:7]
        yyyymmdd = f"20{yymmdd}"
        direct = os.path.join(IMAGES_DIR, yyyymmdd, "images", image_name)
        if os.path.exists(direct):
            return direct
    return image_name  # 返回原路径，让调用方处理