import os, yaml

DATA_DIR = r"H:\dev\disaster-data\infra_datasets\rdd2020"
OUT_DIR = r"H:\dev\disaster-data\models\infra\road_bridge"

data_yaml = os.path.join(OUT_DIR, "dataset_rdd2020.yaml")
yaml_data = {
    "train": [
        os.path.join(DATA_DIR, "train", "Czech", "images"),
        os.path.join(DATA_DIR, "train", "India", "images"),
        os.path.join(DATA_DIR, "train", "Japan", "images"),
    ],
    "val": [
        os.path.join(DATA_DIR, "train", "Czech", "images"),
        os.path.join(DATA_DIR, "train", "India", "images"),
        os.path.join(DATA_DIR, "train", "Japan", "images"),
    ],
    "nc": 4,
    "names": ["D00", "D10", "D20", "D40"],
}
with open(data_yaml, "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False)
print("YAML updated:", data_yaml)
