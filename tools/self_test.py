"""
自测脚本：验证项目结构和代码正确性

阶段 1: 代码导入、类定义、函数签名验证（无需网络/权重）
阶段 2: 真实数据推理测试（水位3张 + 道路3张 + 洪水3张）
阶段 3: 指标报告
"""
import os
import sys
import inspect
import importlib
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PASS = 0
FAIL = 0
SKIP = 0


def check(description: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {description}")
        PASS += 1
    else:
        print(f"  [FAIL] {description} - {detail}")
        FAIL += 1


def check_import(module_name: str, attr: str = None):
    try:
        mod = importlib.import_module(module_name)
        if attr:
            obj = getattr(mod, attr, None)
            check(f"from {module_name} import {attr}", obj is not None)
            return obj
        check(f"import {module_name}", True)
        return mod
    except Exception as e:
        check(f"import {module_name}", False, str(e))
        return None


def stage_1_code_structure():
    """stage 1: code structure check"""
    print("\n" + "=" * 60)
    print("Stage 1: Code Structure Check")
    print("=" * 60)

    print("\n  [1.1] Project Directory Structure")
    for d in ["models", "models/water_level", "models/road_bridge", "models/flood_detection", "tools"]:
        path = os.path.join(PROJECT_ROOT, d)
        check(f"dir exists: {d}", os.path.isdir(path))

    print("\n  [1.2] Module Import")
    WL = check_import("models.water_level", "WaterLevelDetector")
    RD = check_import("models.road_bridge", "RoadDamageDetector")
    FD = check_import("models.flood_detection", "FloodDetector")
    VC = check_import("models.road_bridge.inference", "VocToYoloConverter")
    check_import("models.water_level.config")
    check_import("models.road_bridge.config")
    check_import("models.flood_detection.config")

    print("\n  [1.3] Class Definition and Method Signature")
    if WL:
        check("WaterLevelDetector has infer()", hasattr(WL, "infer"))
        check("WaterLevelDetector has evaluate()", hasattr(WL, "evaluate"))
    if RD:
        check("RoadDamageDetector has infer()", hasattr(RD, "infer"))
    if FD:
        check("FloodDetector has predict()", hasattr(FD, "predict"))
        check("FloodDetector has infer()", hasattr(FD, "infer"))
    if VC:
        check("VocToYoloConverter has convert_xml()", hasattr(VC, "convert_xml"))
        check("VocToYoloConverter has convert_split()", hasattr(VC, "convert_split"))

    print("\n  [1.4] Unified Inference API")
    infer_api = check_import("tools.infer_api", "process_infra_image")
    if infer_api:
        sig = inspect.signature(infer_api)
        params = list(sig.parameters.keys())
        check("process_infra_image accepts image_path", "image_path" in params)
        check("process_infra_image accepts task", "task" in params)

    print("\n  [1.5] Config Constants")
    from models.water_level.config import DATA_DIR as WL_D, XLSX_PATH
    from models.road_bridge.config import DATA_DIR as RB_D, RDD2020_CLASSES
    from models.flood_detection.config import DATA_DIR as FD_D, FLOOD_CLASSES, FLOOD_MODEL_WEIGHT
    check("water_level DATA_DIR defined", bool(WL_D))
    check("water_level XLSX_PATH defined", bool(XLSX_PATH))
    check("road_bridge DATA_DIR defined", bool(RB_D))
    check("RDD2020_CLASSES has 4 classes", len(RDD2020_CLASSES) == 4)
    check("flood_detection DATA_DIR defined", bool(FD_D))
    check("FLOOD_CLASSES has 10 classes", len(FLOOD_CLASSES) == 10)
    check("FLOOD_MODEL_WEIGHT defined", bool(FLOOD_MODEL_WEIGHT))

    print("\n  [1.6] Training Entry Point")
    train_mod = check_import("tools.train")
    if train_mod:
        check("tools.train has main()", hasattr(train_mod, "main"))
        check("tools.train has prepare_rdd2020_data()",
              hasattr(train_mod, "prepare_rdd2020_data"))
        check("tools.train has train_flood_detection()",
              hasattr(train_mod, "train_flood_detection"))


def stage_2_inference():
    """stage 2: real data inference test"""
    print("\n" + "=" * 60)
    print("Stage 2: Real Data Inference Test")
    print("=" * 60)

    results = {}

    print("\n  [2.1] Water Level Detection")
    from models.water_level.inference import WaterLevelDetector
    from models.water_level.config import IMAGES_DIR
    try:
        detector = WaterLevelDetector()
        date_dirs = sorted(os.listdir(IMAGES_DIR))
        test_images = []
        for dd in date_dirs:
            img_dir = os.path.join(IMAGES_DIR, dd, "images")
            if os.path.isdir(img_dir):
                for f in sorted(os.listdir(img_dir)):
                    if f.lower().endswith(".jpg"):
                        test_images.append(os.path.join(img_dir, f))
            if len(test_images) >= 3:
                break
        if test_images:
            print(f"  test images: {len(test_images[:3])}")
            wl_results = []
            for img_path in test_images[:3]:
                r = detector.infer(img_path)
                wl_results.append(r)
                name = r["image_name"]
                if r["water_level_cm"] is not None:
                    print(f"    {name}: {r['water_level_cm']:.1f}cm time: {r['inference_time_ms']:.0f}ms")
                else:
                    print(f"    {name}: {r['details'].get('error', 'fail')} time: {r['inference_time_ms']:.0f}ms")
            errors = [r.get("absolute_error_cm") for r in wl_results if r.get("absolute_error_cm") is not None]
            times = [r["inference_time_ms"] for r in wl_results]
            results["water_level"] = {
                "mae_cm": round(float(np.mean(errors)), 2) if errors else None,
                "mean_time_ms": round(float(np.mean(times)), 2) if times else None,
                "n": len(wl_results),
                "n_with_gt": len(errors)
            }
            if errors:
                print(f"    >> MAE: {results['water_level']['mae_cm']} cm")
            print(f"    >> avg time: {results['water_level']['mean_time_ms']} ms")
        else:
            print("  [SKIP] no test images")
            SKIP += 1
    except Exception as e:
        print(f"  [ERROR] water level test failed: {e}")
        FAIL += 1

    print("\n  [2.2] Road/Bridge Damage Detection")
    from models.road_bridge.inference import RoadDamageDetector
    from models.road_bridge.config import DATA_DIR as RB_DATA_DIR
    try:
        detector = RoadDamageDetector()
        test_images = []
        for c in ["Czech", "India", "Japan"]:
            img_dir = os.path.join(RB_DATA_DIR, "test1", c, "images")
            if os.path.isdir(img_dir):
                for f in sorted(os.listdir(img_dir))[:2]:
                    test_images.append(os.path.join(img_dir, f))
            if len(test_images) >= 3:
                break
        if test_images:
            print(f"  test images: {len(test_images[:3])}")
            rd_results = []
            for img_path in test_images[:3]:
                r = detector.infer(img_path)
                rd_results.append(r)
                name = os.path.basename(img_path)
                if r["total_damages"] > 0:
                    classes_str = ", ".join(f"{k}={v}" for k, v in r["class_counts"].items() if v > 0)
                    print(f"    {name}: {r['total_damages']} damages ({classes_str}, time: {r['inference_time_ms']:.0f}ms)")
                else:
                    print(f"    {name}: no damage (time: {r['inference_time_ms']:.0f}ms)")
            times = [r["inference_time_ms"] for r in rd_results]
            results["road_bridge"] = {
                "mean_time_ms": round(float(np.mean(times)), 2) if times else None,
                "n": len(rd_results)
            }
            print(f"    >> avg time: {results['road_bridge']['mean_time_ms']} ms")
        else:
            print("  [SKIP] no test images")
            SKIP += 1
    except Exception as e:
        print(f"  [ERROR] road detection test failed: {e}")
        FAIL += 1

    print("\n  [2.3] Flood Segmentation")
    from models.flood_detection.inference import FloodDetector
    from models.flood_detection.config import IMAGES_DIR as FD_IMG_DIR
    try:
        detector = FloodDetector()
        test_images = []
        if os.path.isdir(FD_IMG_DIR):
            for f in sorted(os.listdir(FD_IMG_DIR))[:3]:
                if f.lower().endswith((".jpg", ".png")):
                    test_images.append(os.path.join(FD_IMG_DIR, f))
        if test_images:
            print(f"  test images: {len(test_images)}")
            fd_results = []
            for img_path in test_images:
                r = detector.infer(img_path)
                fd_results.append(r)
                name = os.path.basename(img_path)
                assert "积水面积_m2" in r, "missing field: 积水面积_m2"
                assert "淹没占比" in r, "missing field: 淹没占比"
                assert "灾情等级" in r, "missing field: 灾情等级"
                assert "inference_time_ms" in r, "missing field: inference_time_ms"
                print(f"    {name}: area={r['积水面积_m2']}m2 ratio={r['淹没占比']:.4f} level={r['灾情等级']} time={r['inference_time_ms']:.0f}ms")
            times = [r["inference_time_ms"] for r in fd_results]
            results["flood_detection"] = {
                "mean_time_ms": round(float(np.mean(times)), 2) if times else None,
                "n": len(fd_results)
            }
            print(f"    >> avg time: {results['flood_detection']['mean_time_ms']} ms")
        else:
            print("  [SKIP] no test images")
            SKIP += 1
    except Exception as e:
        print(f"  [ERROR] flood segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        FAIL += 1

    return results


def stage_3_report(perf_results: dict):
    """stage 3: performance report"""
    print("\n" + "=" * 60)
    print("Stage 3: Performance Report")
    print("=" * 60)

    print("\n  [Water Level Detection]")
    wl = perf_results.get("water_level", {})
    if wl:
        print(f"    samples: {wl.get('n', 0)}")
        print(f"    with gt: {wl.get('n_with_gt', 0)}")
        if wl.get("mae_cm") is not None:
            print(f"    MAE: {wl['mae_cm']} cm")
        else:
            print(f"    MAE: N/A")
        print(f"    avg inference time: {wl.get('mean_time_ms', 'N/A')} ms")
    else:
        print("    not tested")

    print("\n  [Flood Segmentation]")
    fd = perf_results.get("flood_detection", {})
    if fd:
        print(f"    samples: {fd.get('n', 0)}")
        print(f"    avg inference time: {fd.get('mean_time_ms', 'N/A')} ms")
    else:
        print("    not tested")

    print("\n  [Road Damage Detection]")
    rd = perf_results.get("road_bridge", {})
    if rd:
        print(f"    samples: {rd.get('n', 0)}")
        print(f"    avg inference time: {rd.get('mean_time_ms', 'N/A')} ms")
    else:
        print("    not tested")


def main():
    print("=" * 60)
    print("Infrastructure Disaster Visual Assessment - Self Test")
    print("=" * 60)
    print(f"  Python: {sys.executable}")
    print(f"  Project Root: {PROJECT_ROOT}")

    global PASS, FAIL, SKIP

    stage_1_code_structure()
    perf_results = stage_2_inference()
    stage_3_report(perf_results)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 60)

    if FAIL > 0:
        print("\n[WARN] some checks failed")
        sys.exit(1)
    else:
        print("\n[OK] all checks passed!")


if __name__ == "__main__":
    main()
