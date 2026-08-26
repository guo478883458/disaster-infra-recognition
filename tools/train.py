"""
训练入口

支持三种子模型的训练：
  1. water_level: 水位尺数字检测训练（YOLOv8）
  2. road_bridge: 道路损毁检测训练（YOLOv8 + RDD2020 数据准备）
  3. flood_detection: 洪水场景物体检测训练（YOLOv8 + FloodIMG）

用法:
    python tools/train.py --model water_level --data_yaml dataset.yaml
    python tools/train.py --model road_bridge --epochs 50 --prepare_data
    python tools/train.py --model flood_detection --epochs 100
    python tools/train.py --model flood_detection --resume  # 断点续训

数据准备:
    road_bridge 训练前先运行 --prepare_data 将 VOC XML 转为 YOLO 格式
    flood_detection 训练前先运行 tools/prepare_floodimg.py 转换 labelme→YOLO
"""
import os
import sys
import argparse
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.water_level.config import (
    IMAGES_DIR as WL_IMAGES_DIR,
    MODEL_WEIGHTS_DIR as WL_WEIGHTS_DIR,
)
from models.road_bridge.config import (
    DATA_DIR as RB_DATA_DIR,
    MODEL_WEIGHTS_DIR as RB_WEIGHTS_DIR,
    YOLO_LABELS_DIR as RB_LABELS_DIR,
    RDD2020_CLASSES, get_country_paths
)
from models.road_bridge.inference import VocToYoloConverter
from models.flood_detection.config import (
    DATA_DIR as FD_DATA_DIR,
    IMAGES_DIR as FD_IMAGES_DIR,
    YOLO_LABELS_DIR as FD_LABELS_DIR,
    MODEL_WEIGHTS_DIR as FD_WEIGHTS_DIR,
    FLOOD_CLASSES,
)


def prepare_rdd2020_data():
    """将 RDD2020 的 VOC XML 标注批量转换为 YOLO 格式"""
    print("=" * 60)
    print("RDD2020 数据准备：VOC XML → YOLO 格式")
    print("=" * 60)

    converter = VocToYoloConverter(labels_dir=RB_LABELS_DIR)
    total_converted = 0
    total_total = 0

    for split in ["train", "test1", "test2"]:
        result = converter.convert_split(split)
        total_converted += result["converted"]
        total_total += result["total"]
        print(f"  {split}: {result['converted']}/{result['total']} 转换完成")
        print(f"    输出目录: {result['output_dir']}")

    print(f"\n  总计: {total_converted}/{total_total} 标注转换完成")
    return total_converted


def create_rdd2020_yaml(output_path: str):
    """创建 RDD2020 的 YOLOv8 训练数据集 YAML 文件"""
    data = {
        "train": [os.path.join(RB_DATA_DIR, "train", c, "images")
                  for c in ["Czech", "India", "Japan"]],
        "val": [os.path.join(RB_DATA_DIR, "test1", c, "images")
                for c in ["Czech", "India", "Japan"]],
        "nc": len(RDD2020_CLASSES),
        "names": list(RDD2020_CLASSES.values()),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  YAML 已创建: {output_path}")
    return output_path


def create_floodimg_yaml(output_path: str):
    """创建 FloodIMG 的 YOLOv8 训练数据集 YAML 文件"""
    # 训练集：全部 25 张有标注图像（样本量小，全部用于训练）
    images_dir = FD_IMAGES_DIR
    labels_dir = FD_LABELS_DIR

    data = {
        "train": images_dir,
        "val": images_dir,  # 样本太少，训测同一批
        "nc": len(FLOOD_CLASSES),
        "names": list(FLOOD_CLASSES.values()),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  YAML 已创建: {output_path}")
    return output_path


def _resume_or_init(model_cls, weights_dir, project_name, args):
    """
    处理断点续训逻辑

    Args:
        model_cls: YOLO 类
        weights_dir: 权重目录
        project_name: 项目名（子目录）
        args: 命令行参数

    Returns:
        YOLO 模型实例
    """
    if args.resume:
        last_pt = os.path.join(weights_dir, project_name, "weights", "last.pt")
        if not os.path.exists(last_pt):
            raise FileNotFoundError(
                f"[ERROR] --resume 但未找到断点权重: {last_pt}\n"
                f"       请先正常训练生成 last.pt，或去掉 --resume 从头开始训练"
            )
        print(f"  [续训] 从断点恢复: {last_pt}")
        model = model_cls(last_pt)
        # resume=True 自动从断点恢复 epoch/optimizer 状态
        return model, dict(resume=True)
    else:
        model = model_cls("yolov8n.pt")
        return model, {}


def train_road_bridge(args):
    """训练道路损毁检测模型"""
    print("=" * 60)
    print("训练道路损毁检测模型")
    print(f"  数据: {RB_DATA_DIR}")
    print(f"  权重保存: {RB_WEIGHTS_DIR}")
    print(f"  类别: {RDD2020_CLASSES}")
    print("=" * 60)

    os.makedirs(RB_WEIGHTS_DIR, exist_ok=True)
    project_name = "rdd2020_yolov8n"

    if args.prepare_data:
        prepare_rdd2020_data()

    data_yaml = args.data_yaml or os.path.join(RB_DATA_DIR, "dataset_rdd2020.yaml")
    if not os.path.exists(data_yaml):
        data_yaml = create_rdd2020_yaml(data_yaml)

    from ultralytics import YOLO
    model, resume_kwargs = _resume_or_init(YOLO, RB_WEIGHTS_DIR, project_name, args)

    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=RB_WEIGHTS_DIR,
        name=project_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=20,
        val=True,
    )
    train_kwargs.update(resume_kwargs)

    results = model.train(**train_kwargs)
    print(f"\n训练完成！权重保存至: {RB_WEIGHTS_DIR}")
    return results


def train_water_level(args):
    """训练水位尺数字检测模型"""
    print("=" * 60)
    print("训练水位尺数字检测模型")
    print(f"  数据: {WL_IMAGES_DIR}")
    print(f"  权重保存: {WL_WEIGHTS_DIR}")
    print("=" * 60)

    os.makedirs(WL_WEIGHTS_DIR, exist_ok=True)
    project_name = "water_level_digit"

    data_yaml = args.data_yaml
    if not data_yaml or not os.path.exists(data_yaml):
        print("[ERROR] 水位尺训练需要 dataset.yaml 文件")
        print("       请先准备水位尺数字检测标注数据并创建 YAML")
        return

    from ultralytics import YOLO
    model, resume_kwargs = _resume_or_init(YOLO, WL_WEIGHTS_DIR, project_name, args)

    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=WL_WEIGHTS_DIR,
        name=project_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        amp=True,  # 跳过 AMP 检查（网络不通无法下载 yolo26n.pt，不影响训练）
    )
    train_kwargs.update(resume_kwargs)

    results = model.train(**train_kwargs)
    print(f"\n训练完成！权重保存至: {WL_WEIGHTS_DIR}")
    return results


def train_flood_detection(args):
    """训练洪水场景物体检测模型"""
    print("=" * 60)
    print("训练洪水场景物体检测模型")
    print(f"  数据: {FD_DATA_DIR}")
    print(f"  权重保存: {FD_WEIGHTS_DIR}")
    print(f"  类别: {FLOOD_CLASSES}")
    print("=" * 60)

    os.makedirs(FD_WEIGHTS_DIR, exist_ok=True)
    project_name = "flood_detection"

    # 检查 YOLO 标签是否存在
    if not os.path.isdir(FD_LABELS_DIR):
        print(f"[ERROR] 未找到 YOLO 标签目录: {FD_LABELS_DIR}")
        print(f"       请先运行: python tools/prepare_floodimg.py")
        sys.exit(1)

    # 创建数据集 YAML
    data_yaml = args.data_yaml or os.path.join(FD_DATA_DIR, "dataset_floodimg.yaml")
    if not os.path.exists(data_yaml):
        data_yaml = create_floodimg_yaml(data_yaml)

    from ultralytics import YOLO
    model, resume_kwargs = _resume_or_init(YOLO, FD_WEIGHTS_DIR, project_name, args)

    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=FD_WEIGHTS_DIR,
        name=project_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=20,
        val=True,
    )
    train_kwargs.update(resume_kwargs)

    results = model.train(**train_kwargs)
    print(f"\n训练完成！权重保存至: {FD_WEIGHTS_DIR}")
    return results


def main():
    parser = argparse.ArgumentParser(description="基础设施灾损识别 - 模型训练")
    parser.add_argument("--model", "-m", required=True,
                        choices=["water_level", "road_bridge", "flood_detection"],
                        help="要训练的模型")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--imgsz", "-s", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda"], help="训练设备")
    parser.add_argument("--data_yaml", type=str, default=None,
                        help="数据集 YAML 配置文件路径")
    parser.add_argument("--prepare_data", action="store_true",
                        help="[road_bridge] 先转换 VOC→YOLO 格式")
    parser.add_argument("--resume", action="store_true",
                        help="从断点续训（自动查找 last.pt）")

    args = parser.parse_args()

    if args.model == "water_level":
        train_water_level(args)
    elif args.model == "road_bridge":
        train_road_bridge(args)
    elif args.model == "flood_detection":
        train_flood_detection(args)


if __name__ == "__main__":
    main()
