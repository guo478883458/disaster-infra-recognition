import os, json, glob
from collections import Counter

def audit_floodimg():
    base = r"H:\dev\disaster-data\image_datasets\floodimg"
    ann_dir = os.path.join(base, "Annotation")
    img_dir = os.path.join(base, "Flood Images")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    n_jpg = len(glob.glob(os.path.join(img_dir, "*.jpg")))
    n_png = len(glob.glob(os.path.join(img_dir, "*.png")))
    print(f"  图像总数: {len(images)} (jpg {n_jpg} + png {n_png})")
    jsons = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    print(f"  标注 JSON 数: {len(jsons)}")
    label_counter = Counter()
    annotated_image_ids = set()
    total_shapes = 0
    multi_shape_jsons = 0
    for jp in jsons:
        with open(jp, "r") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        total_shapes += len(shapes)
        if len(shapes) > 1:
            multi_shape_jsons += 1
        for s in shapes:
            label_counter[s["label"]] += 1
        img_name = data.get("imagePath", "")
        annotated_image_ids.add(img_name)
    print(f"  有标注图像: {len(annotated_image_ids)} 张")
    print(f"  标注对象总数: {total_shapes} 个")
    print(f"  含多个标注的 JSON: {multi_shape_jsons} 个")
    print()
    print(f"  标注类别分布:")
    for label, cnt in sorted(label_counter.items(), key=lambda x: -x[1]):
        print(f"    {label}: {cnt}")
    print()
    print(f"  注意: 数据集中无Flood标签，仅有通用场景标注")
    print(f"  >> 结论: 仅 {len(jsons)} 张有标注，其余 {len(images)-len(jsons)} 张为无标注图像")
    print(f"  >> 建议: 训练YOLO检测模型，对洪水场景中的物体进行检测")

def audit_rescuenet():
    base = r"H:\dev\disaster-data\image_datasets\rescuenet\segmentation-trainset"
    org_dir = os.path.join(base, "train-org-img")
    lbl_dir = os.path.join(base, "train-label-img")
    org_files = sorted(os.listdir(org_dir))
    lbl_files = sorted(os.listdir(lbl_dir))
    print(f"  原图 (train-org-img): {len(org_files)} 张")
    print(f"  标签 (train-label-img): {len(lbl_files)} 张")
    org_basenames = set(os.path.splitext(f)[0] for f in org_files)
    lbl_basenames = set()
    for f in lbl_files:
        name = os.path.splitext(f)[0]
        if name.endswith("_lab"):
            name = name[:-4]
        lbl_basenames.add(name)
    matched = org_basenames & lbl_basenames
    org_only = org_basenames - lbl_basenames
    lbl_only = lbl_basenames - org_basenames
    print(f"\n  文件名对应:")
    print(f"    双向匹配: {len(matched)} 张")
    print(f"    仅原图有: {len(org_only)} 张")
    print(f"    仅标签有: {len(lbl_only)} 张")
    import numpy as np
    from PIL import Image
    print(f"\n  标签 PNG 像素值抽样(前5张):")
    sample_lbls = sorted(lbl_files)[:5]
    unique_vals = set()
    for f in sample_lbls:
        path = os.path.join(lbl_dir, f)
        img = Image.open(path)
        arr = np.array(img)
        vals = sorted(np.unique(arr).tolist())
        unique_vals.update(vals)
        print(f"    {f}: 尺寸 {img.size}, 唯一值 {vals}")
    print(f"  >> 所有抽样标签中像素值集合: {sorted(unique_vals)}")
    print(f"  >> 推测: 0=背景, 1=洪水, 可能还有其他类别")
    print(f"  >> 许可: CC BY-NC-ND (非商用，训练前需确认用途)")
    print(f"  >> 可用配对数: {len(matched)}")

def audit_sturm():
    base = r"H:\dev\disaster-data\image_datasets\sturm_flood\Dataset"
    s1_img = os.path.join(base, "Sentinel1", "S1")
    s1_fm = os.path.join(base, "Sentinel1", "Floodmaps")
    s1_img_count = len(os.listdir(s1_img))
    s1_fm_count = len(os.listdir(s1_fm))
    s2_img = os.path.join(base, "Sentinel2", "S2")
    s2_fm = os.path.join(base, "Sentinel2", "Floodmaps")
    s2_img_count = len(os.listdir(s2_img)) if os.path.isdir(s2_img) else 0
    s2_fm_count = len(os.listdir(s2_fm)) if os.path.isdir(s2_fm) else 0
    print(f"  Sentinel1: S1 瓦片 {s1_img_count} 张, Floodmaps {s1_fm_count} 张")
    print(f"  Sentinel2: S2 瓦片 {s2_img_count} 张, Floodmaps {s2_fm_count} 张")
    s1_names = set(os.listdir(s1_img))
    fm_names = set(os.listdir(s1_fm))
    matched_s1 = s1_names & fm_names
    print(f"  S1+Floodmaps 同名匹配: {len(matched_s1)}/{s1_img_count}")
    import numpy as np
    from PIL import Image
    print(f"\n  Floodmaps 像素值抽样(前3张S1):")
    sample_fm = sorted(os.listdir(s1_fm))[:3]
    for f in sample_fm:
        path = os.path.join(s1_fm, f)
        img = Image.open(path)
        arr = np.array(img)
        vals = sorted(np.unique(arr).tolist())
        print(f"    {f}: 尺寸 {img.size}, 唯一值 {vals}")
    print(f"\n  >> 结论: S1完整({s1_img_count}对), S2不完整({s2_img_count}张)")
    print(f"  >> 遥感域差异大，本次不训练，仅做数据说明")

def main():
    print("=" * 60)
    print("新数据集审计报告(只读统计)")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("【FloodIMG】")
    print("=" * 60)
    audit_floodimg()
    print("\n" + "=" * 60)
    print("【RescueNet】")
    print("=" * 60)
    audit_rescuenet()
    print("\n" + "=" * 60)
    print("【STURM-Flood】")
    print("=" * 60)
    audit_sturm()
    print("\n" + "=" * 60)
    print("审计完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
