#!/usr/bin/env python3
"""在单张图片上测试 YOLOv8 对 person 的检出效果。

用途：验证「背对镜头 + 坐姿」这个 case，预训练模型到底能不能检出来；
      顺带对比 .pt / OpenVINO 两种格式的精度和速度，决定要不要自训练。

前提：
    source yolo_env/bin/activate

用法：
    # 1) 看环境诊断（搞清楚之前配的 CPU 加速到底是什么）
    python snap_test.py --info

    # 2) 测单张照片
    python snap_test.py --image seat.jpg

    # 3) 对比多个模型 / 格式（推荐这么跑）
    python snap_test.py --image seat.jpg --models n s m yolov8n_openvino_model/

    # 4) 限定检测区域（归一化 0~1：x1,y1,x2,y2）
    #    --zone  推荐：不裁剪，全图检测后只保留框中心在该区域的 person（= 摄像头 Activity Zone）
    #    --roi   会把图片真裁掉再检测（结果图会变小），一般只用于对比
    python snap_test.py --image seat.jpg --zone 0.62,0.44,0.88,0.84
    python snap_test.py --image seat.jpg --roi 0.55,0.38,0.96,1.00

    # 5) 指定推理设备（OpenVINO 下可试 intel:gpu / cpu）
    python snap_test.py --image seat.jpg --models yolov8n_openvino_model/ --device cpu

输出：每张图存一张画了框的 <原图名>_<模型名>.jpg，方便肉眼确认。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PERSON_CLASS = 0
DEFAULT_CONF = 0.25
REPEAT = 3  # 计时重复次数（预热后取多次，避免偶然值）

SEP = "=" * 68


# ------------------------------ 环境诊断 ------------------------------
def print_info() -> None:
    """打印环境诊断，看清楚之前配的「CPU 加速」到底是什么。"""
    print(SEP)
    print("环境诊断")
    print(SEP)
    print(f"Python       : {sys.version.split()[0]}")

    def safe(label: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # 任何一项查不到都不影响其它项
            print(f"{label}: 查询失败（{type(exc).__name__}: {exc}）")

    def _torch():
        import torch

        print(f"torch        : {torch.__version__}")
        print(f"  推理线程数 : {torch.get_num_threads()}")
        cuda = torch.cuda.is_available()
        print(f"  CUDA 可用  : {cuda}")

    def _ipex():
        import intel_extension_for_pytorch as ipex

        print(f"IPEX         : {ipex.__version__}   <-- Intel CPU 加速扩展")

    def _openvino():
        import openvino as ov

        core = ov.Core()
        print(f"OpenVINO     : {ov.__version__}")
        print(f"  可用设备   : {core.available_devices}")

    def _ultra():
        import ultralytics

        print(f"ultralytics  : {ultralytics.__version__}")

    def _ort():
        import onnxruntime as ort

        print(f"onnxruntime  : {ort.__version__}")
        print(f"  providers  : {ort.get_available_providers()}")

    def _cv2():
        import cv2

        print(f"opencv       : {cv2.__version__}")

    safe("torch", _torch)
    try:
        _ipex()
    except ImportError:
        print("IPEX         : 未安装")
    try:
        _openvino()
    except ImportError:
        print("OpenVINO     : 未安装（yolo export format=openvino 需要它）")
    safe("ultralytics", _ultra)
    try:
        _ort()
    except ImportError:
        print("onnxruntime  : 未安装")
    safe("opencv", _cv2)

    print()
    print("怎么看这份输出：")
    print("  - OpenVINO 有输出且可用设备含 CPU/GPU  -> 之前配的加速就是它，走 openvino 格式最划算")
    print("  - IPEX 有输出                        -> 之前配的是 Intel PyTorch 扩展，.pt 直跑就有加速")
    print("  - 两者都没有                          -> 只有原生 torch，建议装 openvino 再导出")


# ------------------------------ 工具 ------------------------------
def configure_device(device: str | None) -> None:
    """按请求配置推理设备。**必须在加载模型之前调用**。

    ultralytics 各版本选 Intel 核显的方式不统一，这里两条路都铺上：
      - 新版支持 predict(device="intel:gpu")
      - 老版读环境变量 OPENVINO_DEVICE
    两者同时设置，哪个生效都能跑。
    """
    if not device:
        return
    if "gpu" in device.lower() or "intel" in device.lower():
        os.environ["OPENVINO_DEVICE"] = "GPU"
        print(f"已设置 OPENVINO_DEVICE=GPU（请求设备：{device}）")
        print("  注意：核显只对 OpenVINO 模型（*_openvino_model/）生效。")
        print("        若当前是 .pt 模型，ultralytics 会把 intel:gpu 丢给 torch 解析而报错，")
        print("        本脚本会自动改用环境变量方式，.pt 仍跑 CPU。想用核显请先：")
        print("        yolo export model=yolov8s.pt format=openvino half=True")
    elif "cpu" in device.lower():
        os.environ["OPENVINO_DEVICE"] = "CPU"


def resolve_model(spec: str) -> str:
    """'n'/'s'/'m' 简写补全成 yolov8n.pt；已存在的路径原样返回。"""
    if Path(spec).exists():
        return spec
    if len(spec) <= 3 and spec.isalnum():
        return f"yolov8{spec}.pt"
    return spec


def model_label(path: str) -> str:
    """给输出图起个短名字。"""
    p = Path(path.rstrip("/"))
    name = p.name or p.parent.name
    return name.replace(".pt", "")


def load_image(path: Path):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"读不到图片：{path}")
    return img


def crop_roi(img, roi):
    """roi = (x1, y1, x2, y2)，归一化 0~1。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    return img[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def parse_roi(text: str):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--roi 需要 4 个数：x1,y1,x2,y2（归一化 0~1），例如 0.35,0.45,0.65,0.95")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit("--roi 必须是数字")
    if not all(0.0 <= v <= 1.0 for v in vals):
        raise SystemExit("--roi 取值必须在 0~1 之间")
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        raise SystemExit("--roi 必须是 x1<x2 且 y1<y2")
    return vals


# ------------------------------ 推理 ------------------------------
def run_one(model_path: str, img, conf: float, device: str | None, half: bool = False):
    """返回 (结果对象, 加载耗时s, 预热耗时s, 单次推理ms)。"""
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(model_path)
    load_s = time.perf_counter() - t0

    kw = {"conf": conf, "classes": [PERSON_CLASS], "verbose": False}
    # intel:gpu 这类设备字符串，ultralytics 只在 OpenVINO 后端认；
    # 传给 .pt 会抛 "Invalid device string"。而 OpenVINO 后端实测是靠
    # OPENVINO_DEVICE 环境变量选设备的（configure_device 里已设好），
    # 所以 GPU 请求一律不传 device，避免误伤 .pt 模型。
    is_gpu_req = bool(device) and ("gpu" in device.lower() or "intel" in device.lower())
    if device and not is_gpu_req:
        kw["device"] = device
    if half:
        kw["half"] = True

    def predict():
        return model(img, **kw)

    # 预热：首次推理含 GPU kernel 编译 / 初始化，必须单独计时
    try:
        t = time.perf_counter()
        predict()
        warm_s = time.perf_counter() - t
    except Exception as exc:
        print(f"  以 {device or '默认设备'} 推理失败（{exc}），回退到 cpu")
        kw["device"] = "cpu"
        os.environ["OPENVINO_DEVICE"] = "CPU"
        t = time.perf_counter()
        predict()
        warm_s = time.perf_counter() - t

    best = None
    total = 0.0
    for _ in range(REPEAT):
        t = time.perf_counter()
        res = predict()
        total += (time.perf_counter() - t) * 1000
        best = res[0]

    return best, load_s, warm_s, total / REPEAT


def extract(result, img):
    """提取 person 检出框，附带占比信息。"""
    h, w = img.shape[:2]
    dets = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return dets
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append(
            {
                "conf": float(b.conf[0]),
                "xyxy": (round(x1), round(y1), round(x2), round(y2)),
                "area_pct": round((x2 - x1) * (y2 - y1) / (w * h) * 100, 1),
            }
        )
    dets.sort(key=lambda d: -d["conf"])
    return dets


def filter_by_zone(dets, zone, img):
    """只保留「框中心落在 zone 内」的检出。

    zone = (x1, y1, x2, y2)，归一化 0~1。
    图片不裁剪，检测照常跑全图，只是事后把区域外的框扔掉
    ——这才是「限定检测区域」的正确做法（对应摄像头的 Activity Zone）。
    """
    h, w = img.shape[:2]
    zx1, zy1, zx2, zy2 = zone
    kept, dropped = [], []
    for i, d in enumerate(dets):
        bx1, by1, bx2, by2 = d["xyxy"]
        nx = ((bx1 + bx2) / 2) / w
        ny = ((by1 + by2) / 2) / h
        d["idx"] = i
        d["center"] = (round(nx, 3), round(ny, 3))
        d["h_pct"] = round((by2 - by1) / h, 3)  # 框高占比，可用来区分坐/站
        (kept if zx1 <= nx <= zx2 and zy1 <= ny <= zy2 else dropped).append(d)
    return kept, dropped


def parse_zone(text: str):
    """格式同 --roi，只是报错信息不同。"""
    try:
        return parse_roi(text)
    except SystemExit:
        raise SystemExit("--zone 格式：x1,y1,x2,y2（归一化 0~1），例如 0.62,0.44,0.88,0.84")


def report(label: str, dets, load_s: float, warm_s: float, ms: float, img) -> None:
    h, w = img.shape[:2]
    print(f"\n---- {label} ----")
    print(f"输入尺寸     : {w}x{h}")
    print(f"模型加载     : {load_s:.2f}s")
    print(f"首次推理     : {warm_s * 1000:.1f}ms   （含 GPU 编译，只看一次，不代表常态）")
    print(f"稳定推理     : {ms:.1f}ms   （预热后 {REPEAT} 次平均，这才是真实速度）")

    if not dets:
        print("检出结果     : 没有检出 person ❌")
        print("  -> 这张图没检出来。可以试 --conf 0.15 看看低分框，或换更大的模型（s/m/l）")
        return

    top = dets[0]
    verdict = "✅ 检出" if top["conf"] >= 0.5 else ("⚠️ 勉强（可配合运动先验使用）" if top["conf"] >= 0.25 else "❌ 太低")
    print(f"检出 person  : {len(dets)} 个，最高置信度 {top['conf']:.3f}  {verdict}")
    for i, d in enumerate(dets[:5], 1):
        x1, y1, x2, y2 = d["xyxy"]
        print(
            f"  {i}. conf={d['conf']:.3f}  box=({x1},{y1})-({x2},{y2})  "
            f"尺寸={x2 - x1}x{y2 - y1}  占画面={d['area_pct']}%"
        )


def save_annotated(result, out_path: Path) -> bool:
    try:
        import cv2

        plotted = result.plot()
        cv2.imwrite(str(out_path), plotted)
        return True
    except Exception as exc:
        print(f"  保存标注图失败：{exc}")
        return False


# ------------------------------ 主流程 ------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="在单张图片上测试 YOLOv8 的 person 检出效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", help="待测试的图片路径")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["n"],
        metavar="M",
        help="要测的模型，默认 n。可写简写 n/s/m/l/x，或完整路径/目录，"
        "如：--models n s m yolov8n_openvino_model/",
    )
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"置信度阈值，默认 {DEFAULT_CONF}")
    ap.add_argument("--roi", help="【裁剪】把图片裁到该区域再检测，归一化 0~1：x1,y1,x2,y2")
    ap.add_argument(
        "--zone",
        help="【过滤】不裁剪，全图检测后只保留框中心落在该区域内的 person，"
        "归一化 0~1：x1,y1,x2,y2。这才是「限定检测区域」，对应摄像头的 Activity Zone",
    )
    ap.add_argument(
        "--device",
        help="推理设备：cpu / intel:gpu / gpu。带 gpu 的会同时设置 OPENVINO_DEVICE=GPU；"
        "核显不可用时自动回退 cpu",
    )
    ap.add_argument(
        "--half",
        action="store_true",
        help="FP16 半精度推理（核显更快）。注意：OpenVINO IR 模型的精度在导出时就固定了，"
        "此开关只对 .pt 真正生效",
    )
    ap.add_argument("--outdir", default=".", help="标注图输出目录，默认当前目录")
    ap.add_argument("--info", action="store_true", help="只打印环境诊断后退出")
    args = ap.parse_args(argv)

    if args.info:
        print_info()
        return 0

    if not args.image:
        ap.error("需要 --image（或改用 --info 只看环境诊断）")

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"图片不存在：{img_path}")

    img = load_image(img_path)
    if args.roi:
        roi = parse_roi(args.roi)
        img = crop_roi(img, roi)
        print(f"已按 ROI {roi} 裁剪，实际输入 {img.shape[1]}x{img.shape[0]}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 设备配置必须在加载模型之前
    configure_device(args.device)

    print(SEP)
    print(f"测试图片：{img_path}")
    print(f"置信度阈值：{args.conf}")
    print(f"请求设备  ：{args.device or '默认'}")
    print(SEP)

    results = []
    for spec in args.models:
        path = resolve_model(spec)
        label = model_label(path)
        try:
            result, load_s, warm_s, ms = run_one(
                path, img, args.conf, args.device, args.half
            )
        except Exception as exc:
            print(f"\n---- {label} ----\n运行失败：{type(exc).__name__}: {exc}")
            continue

        dets = extract(result, img)
        if args.zone:
            zone = parse_zone(args.zone)
            kept, dropped = filter_by_zone(dets, zone, img)
            print(f"\n[区域过滤] zone={zone}")
            print(f"  全图检出 {len(dets)} 个 person -> 区域内 {len(kept)} 个")
            for d in dropped:
                print(f"    [排除] conf={d['conf']:.3f} 中心={d['center']} 框高占={d['h_pct']:.0%}")
            dets = kept
            # 标注图也只画区域内的框，方便肉眼确认
            try:
                result.boxes = result.boxes[[d["idx"] for d in kept]]
            except Exception:
                pass
        report(label, dets, load_s, warm_s, ms, img)

        out = outdir / f"{img_path.stem}_{label}.jpg"
        if save_annotated(result, out):
            print(f"标注图     : {out}")

        results.append((label, dets[0]["conf"] if dets else 0.0, ms, len(dets)))

    if len(results) > 1:
        print("\n" + SEP)
        print("汇总对比")
        print(SEP)
        print(f"{'模型':<28}{'最高置信度':>12}{'推理耗时':>12}{'检出数':>8}")
        for label, conf, ms, n in results:
            flag = "✅" if conf >= 0.5 else ("⚠️" if conf >= 0.25 else "❌")
            print(f"{label:<28}{conf:>10.3f} {flag}{ms:>10.1f}ms{n:>8}")

    print("\n下一步：")
    print("  置信度 >= 0.5        -> 预训练够用，直接搭 seat-watcher，不用自训练")
    print("  置信度 0.25~0.5      -> 配合运动先验勉强可用；想要稳就自训练")
    print("  置信度 < 0.25 或检不出 -> 走自训练路线（先采集数据）")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
