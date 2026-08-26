"""
RescueNet: 分割 PNG → YOLO 分割格式转换

将 2941 对匹配的 org/label 图像转为 YOLO 分割格式
输出目录: H:\dev\disaster-data\image_datasets\rescuenet\yolo_seg_labels\
"""
import os
import glob
import numpy as np
from PIL import Image
import cv2


# ---- 配置 ----
ORG_DIR = r"H:\dev\disaster-data\image_datasets\rescuenet\segmentation-trainset\train-org-img"
LBL_DIR = r"H:\dev\disaster-data\image_datasets\rescuenet\segmentation-trainset\train-label-img"
OUT_DIR = r"H:\dev\disaster-data\image_datasets\rescuenet\yolo_seg_labels"
IMG_OUT_DIR = r"H:\dev\disaster-data\image_datasets\rescuenet\yolo_seg_labels\images"

# RescueNet 类别映射（根据审计结果）
# 0=背景, 1=洪水, 2-10=其他类别
# 先按像素值定义，训练时再决定哪些类别合并
CLASS_NAMES = {
    0: "background",
    1: "flood_water",
    2: "class_2",
    3: "class_3",
    4: "class_4",
    5: "class_5",
    6: "class_6",
    8: "class_8",
    9: "class_9",
    10: "class_10",
}
# 只保留非背景类别用于训练
VALID_CLASSES = [k for k in CLASS_NAMES if k != 0]
# 重新映射到连续 ID（0=flood_water, 1=class_2, ...）
PIXEL_TO_YOLO_ID = {pix: idx for idx, pix in enumerate(sorted(VALID_CLASSES))}

print(f"  类别映射（像素值→YOLO ID）: {PIXEL_TO_YOLO_ID}")


def mask_to_polygons(mask, epsilon=1.0):
    """
    将二值掩膜转为多边形顶点列表（YOLO seg 格式）

    Args:
        mask: 二值掩膜 (H, W) uint8, 前景=255
        epsilon: 多边形简化精度（像素）

    Returns:
        polygons: [[[x1,y1],[x2,y2],...], ...]
    """
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        # 简化轮廓
        epsilon_px = epsilon
        approx = cv2.approxPolyDP(cnt, epsilon_px, True)
        if len(approx) >= 3:  # 至少三角形
            pts = approx.reshape(-1, 2).astype(np.float32)
            polygons.append(pts)
    return polygons


def convert_label_image(label_path, img_h, img_w):
    """
    转换单个标签 PNG → YOLO seg 行列表

    Args:
        label_path: 标签 PNG 路径
        img_h, img_w: 图像尺寸

    Returns:
        lines: YOLO 格式行列表
    """
    lbl = Image.open(label_path)
    arr = np.array(lbl)

    lines = []
    # 按类别处理
    for pix_val, yolo_id in PIXEL_TO_YOLO_ID.items():
        # 提取该类别的二值掩膜
        mask = (arr == pix_val).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue

        # 转为多边形
        polygons = mask_to_polygons(mask)
        for poly in polygons:
            # 归一化
            poly[:, 0] = poly[:, 0] / img_w
            poly[:, 1] = poly[:, 1] / img_h
            # 裁剪到 [0,1]
            poly = np.clip(poly, 0, 1)
            # 展平为 x1 y1 x2 y2 ...
            flat = poly.flatten()
            coords = " ".join(f"{v:.6f}" for v in flat)
            lines.append(f"{yolo_id} {coords}")

    return lines


def main():
    print("=" * 60)
    print("RescueNet: 分割 PNG → YOLO 分割格式")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_OUT_DIR, exist_ok=True)

    # 找匹配配对
    lbl_files = {os.path.splitext(f)[0].replace("_lab", ""): f
                 for f in os.listdir(LBL_DIR) if f.endswith(".png")}
    org_files = sorted([f for f in os.listdir(ORG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    matched = 0
    skipped = 0
    total_polygons = 0

    for org_name in org_files:
        base = os.path.splitext(org_name)[0]
        if base not in lbl_files:
            skipped += 1
            continue

        lbl_name = lbl_files[base]
        org_path = os.path.join(ORG_DIR, org_name)
        lbl_path = os.path.join(LBL_DIR, lbl_name)

        # 获取图像尺寸
        org_img = Image.open(org_path)
        img_w, img_h = org_img.size

        # 转换标签
        lines = convert_label_image(lbl_path, img_h, img_w)
        total_polygons += len(lines)

        # 写入 YOLO seg txt
        txt_name = f"{base}.txt"
        txt_path = os.path.join(OUT_DIR, txt_name)
        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        # 复制原图到 yolo 目录（YOLO 训练需要图像和标签同级）
        import shutil
        dst_img_path = os.path.join(IMG_OUT_DIR, org_name)
        if not os.path.exists(dst_img_path):
            shutil.copy2(org_path, dst_img_path)

        if matched % 200 == 0:
            print(f"  [进度] {matched}/{len(org_files)} 已转换...")

        matched += 1

    # 统计
    print(f"\n  转换完成: {matched} 张")
    print(f"  跳过(无标签): {skipped} 张")
    print(f"  总多边形数: {total_polygons}")
    print(f"  YOLO 标签: {OUT_DIR}")
    print(f"  图像副本: {IMG_OUT_DIR}")
    print(f"  类别数: {len(PIXEL_TO_YOLO_ID)}")
    print(f"  >> 注意: 原始图像尺寸约 4000x3000，训练时建议 imgsz=640/1280 自动缩放")


if __name__ == "__main__":
    main()
