"""
统一推理接口

提供 process_infra_image() 函数，根据任务类型调用对应的子模型进行推理。

用法:
    from tools.infer_api import process_infra_image

    # 水位识别
    result = process_infra_image("image.jpg", task="water_level")

    # 道路损毁检测
    result = process_infra_image("image.jpg", task="road")

    # 洪水分割
    result = process_infra_image("洪灾现场.jpg", task="flood")
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.water_level.inference import WaterLevelDetector
from models.road_bridge.inference import RoadDamageDetector
from models.flood_detection.inference import FloodDetector

_water_level_detector = None
_road_damage_detector = None
_flood_detector = None


def get_water_level_detector(device: str = "cpu") -> WaterLevelDetector:
    global _water_level_detector
    if _water_level_detector is None:
        _water_level_detector = WaterLevelDetector(device=device)
    return _water_level_detector


def get_road_damage_detector(device: str = "cpu") -> RoadDamageDetector:
    global _road_damage_detector
    if _road_damage_detector is None:
        _road_damage_detector = RoadDamageDetector(device=device)
    return _road_damage_detector


def get_flood_detector(device: str = "cpu") -> FloodDetector:
    global _flood_detector
    if _flood_detector is None:
        _flood_detector = FloodDetector(device=device)
    return _flood_detector


def process_infra_image(image_path: str, task: str = "water_level",
                        device: str = "cpu") -> dict:
    """
    统一推理接口：对基础设施灾损图像执行识别/检测

    Args:
        image_path: 图像文件路径
        task: "water_level"/"water" / "road"/"road_bridge" / "flood"/"flood_seg"/"flood_detection"
        device: "cpu" 或 "cuda"

    Returns:
        dict: 结构化结果

    Raises:
        FileNotFoundError: 图像文件不存在
        ValueError: 不支持的任务类型
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    task = task.lower().replace(" ", "_")
    if task in ("water_level", "water"):
        detector = get_water_level_detector(device=device)
        result = detector.infer(image_path)
        result["task"] = "water_level"
        return result

    elif task in ("road", "road_bridge"):
        detector = get_road_damage_detector(device=device)
        result = detector.infer(image_path)
        result["task"] = "road_bridge"
        return result

    elif task in ("flood", "flood_seg", "flood_detection"):
        detector = get_flood_detector(device=device)
        result = detector.infer(image_path)
        result["task"] = "flood"
        return result

    else:
        raise ValueError(
            f"不支持的任务类型: '{task}'。"
            f"支持: 'water_level'/'water', 'road'/'road_bridge', "
            f"'flood'/'flood_seg'/'flood_detection'"
        )


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="基础设施灾损视觉评估 - 统一推理接口")
    parser.add_argument("image_path", help="输入图像路径")
    parser.add_argument("--task", "-t", default="water_level",
                        choices=["water_level", "water", "road", "road_bridge",
                                 "flood", "flood_detection", "flood_seg"],
                        help="任务类型")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda"], help="推理设备")
    args = parser.parse_args()

    result = process_infra_image(args.image_path, task=args.task, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
