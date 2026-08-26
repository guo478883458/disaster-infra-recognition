import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    data_yaml = r"I:\models\infra\road_bridge\dataset_rdd2020.yaml"
    out_dir = r"I:\models\infra\road_bridge"
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project=out_dir,
        name="rdd2020_yolov8n",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=0,
        val=False,
        seed=42,
        workers=4,
        amp=True,
    )
    print("Training complete!")


if __name__ == "__main__":
    main()
