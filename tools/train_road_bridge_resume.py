import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    ckpt = r"I:\models\infra\road_bridge\rdd2020_yolov8n\weights\last.pt"
    model = YOLO(ckpt)
    model.train(resume=True)
    print("Training complete!")


if __name__ == "__main__":
    main()
