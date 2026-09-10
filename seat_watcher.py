#!/usr/bin/env python3
"""餐厅座位占用检测（seat-watcher）。

流程：
    抓帧（go2rtc 或本地图片）
      -> YOLOv8s(OpenVINO / Intel iGPU) 全图检测 person
      -> 过滤 1：框中心必须落在「座位区 zone」内   ← 只认这个位置
      -> 过滤 2：框高占画面比例 < 阈值              ← 排除「站着」，只认「坐着」
      -> 过滤 3：置信度 >= 阈值
      -> 去抖（连续命中 N 次才算在座，连续未命中 M 次才算离开）
      -> 状态变化时写入 Home Assistant 的 binary_sensor

用法（先 source yolo_env/bin/activate）：

    # 单张图测试，不推 HA —— 用来校准 zone 坐标
    python seat_watcher.py --image test.png --once
    python seat_watcher.py --image empty.png --once      # 负样本，验证不误报

    # 常驻运行（读 seat_config.yml）
    python seat_watcher.py

    # 常驻但不推 HA，只在控制台看判定过程
    python seat_watcher.py --dry-run

    # 把判定区画出来存图，肉眼确认 zone 框得对不对
    python seat_watcher.py --image test.png --once --draw zone.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PERSON_CLASS = 0

# 默认值：配置文件缺失时兜底，也方便直接命令行覆盖
DEFAULTS = {
    "source": {"url": "http://127.0.0.1:1984/api/frame.jpeg?src=dining", "timeout": 5},
    "model": {
        "path": "yolov8s_openvino_model/",
        "device": "intel:gpu",
        "imgsz": 640,
        "conf": 0.5,
    },
    # 座位判定区（归一化 0~1）：[x1, y1, x2, y2]
    "zone": {"rect": [0.62, 0.44, 0.88, 0.84], "max_box_h_pct": 0.5},
    "debounce": {
        "hit_needed": 2,
        "miss_needed": 2,
        "hold_minutes": 20,
        "interval": 1.0,
    },
    "ha": {"url": "", "token": "", "entity_id": "binary_sensor.dining_seat_occupied"},
}

log = logging.getLogger("seat")


# ------------------------------ 配置 ------------------------------
def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # 深拷贝，避免改到默认值
    if path is None:
        return cfg
    if not path.is_file():
        log.warning("配置文件不存在，全部使用默认/命令行参数：%s", path)
        return cfg
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "需要 PyYAML 才能读配置文件：pip install pyyaml（或不指定 -c 用默认值）"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"配置文件必须是键值结构：{path}")
    return deep_merge(cfg, data)


# ------------------------------ 抓帧 ------------------------------
def fetch_frame(url: str, timeout: float):
    """从 go2rtc 的快照接口取一帧，返回 BGR 图像。"""
    import cv2
    import numpy as np

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"图片解码失败（收到 {len(data)} 字节）")
    return img


def load_image(path: Path):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"读不到图片：{path}")
    return img


# ------------------------------ 检测器 ------------------------------
class SeatDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        imgsz: int,
        conf: float,
        rect: list,
        max_box_h_pct: float,
    ):
        from ultralytics import YOLO

        # 设备配置必须在加载模型之前（ultralytics 老版本靠这个环境变量选核显）
        if device and "gpu" in str(device).lower():
            os.environ["OPENVINO_DEVICE"] = "GPU"

        t0 = time.perf_counter()
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.rect = tuple(rect)
        self.max_box_h_pct = max_box_h_pct
        log.info(
            "模型已加载 %s（%.2fs），device=%s",
            model_path,
            time.perf_counter() - t0,
            device,
        )

    def warmup(self, img) -> None:
        """首次推理含 GPU kernel 编译（实测约 5 秒），必须在服务启动阶段吃掉。"""
        t0 = time.perf_counter()
        try:
            self._infer(img)
        except Exception as exc:
            if self.device and "gpu" in str(self.device).lower():
                log.warning("设备 %s 不可用（%s），回退 cpu", self.device, exc)
                self.device = "cpu"
                os.environ["OPENVINO_DEVICE"] = "CPU"
                self._infer(img)
            else:
                raise
        log.info(
            "预热完成，耗时 %.1fs（一次性，之后每帧约几十毫秒）",
            time.perf_counter() - t0,
        )

    def _infer(self, img):
        kw = {
            "conf": self.conf,
            "classes": [PERSON_CLASS],
            "imgsz": self.imgsz,
            "verbose": False,
        }
        # intel:gpu 这类字符串 ultralytics 只在 OpenVINO 后端认，传给 .pt 会抛
        # "Invalid device string"；而 OpenVINO 后端是靠 OPENVINO_DEVICE 环境变量
        # 选设备的（构造时已设好），所以 GPU 请求一律不传 device。
        is_gpu = bool(self.device) and (
            "gpu" in self.device.lower() or "intel" in self.device.lower()
        )
        if self.device and not is_gpu:
            kw["device"] = self.device
        return self.model(img, **kw)[0]

    def detect(self, img) -> tuple[bool, list, list]:
        """返回 (是否命中座位, 保留的框, 被排除的框)。

        保留条件：框中心在 zone 内 且 框高占画面比例 < 阈值（坐着而非站着）。
        """
        h, w = img.shape[:2]
        zx1, zy1, zx2, zy2 = self.rect
        result = self._infer(img)
        boxes = getattr(result, "boxes", None)

        kept, dropped = [], []
        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                nx, ny = cx / w, cy / h
                box_h_pct = (y2 - y1) / h
                item = {
                    "conf": conf,
                    "xyxy": (round(x1), round(y1), round(x2), round(y2)),
                    "center": (round(nx, 3), round(ny, 3)),
                    "box_h_pct": round(box_h_pct, 3),
                }
                in_zone = zx1 <= nx <= zx2 and zy1 <= ny <= zy2
                seated = box_h_pct < self.max_box_h_pct
                if in_zone and seated:
                    kept.append(item)
                else:
                    reason = (
                        "不在座位区" if not in_zone else f"站着(框高{box_h_pct:.0%})"
                    )
                    item["reason"] = reason
                    dropped.append(item)
        kept.sort(key=lambda d: -d["conf"])
        return bool(kept), kept, dropped


# ------------------------------ 状态机 ------------------------------
class Occupancy:
    """把逐帧的命中/未命中，去抖成稳定的「在座 / 空」状态。"""

    def __init__(self, hit_needed: int, miss_needed: int, hold_minutes: int):
        self.hit_needed = max(1, hit_needed)
        self.miss_needed = max(1, miss_needed)
        self.hold = timedelta(minutes=hold_minutes)
        self.occupied = False
        self.hit_streak = 0
        self.miss_streak = 0
        self.last_confirm: datetime | None = None

    def update(self, hit: bool) -> str | None:
        """返回 'on' / 'off' 表示状态发生变化，None 表示无变化。"""
        now = datetime.now()
        if hit:
            self.hit_streak += 1
            self.miss_streak = 0
            self.last_confirm = now
            if not self.occupied and self.hit_streak >= self.hit_needed:
                self.occupied = True
                return "on"
        else:
            self.miss_streak += 1
            self.hit_streak = 0
            if self.occupied and self.miss_streak >= self.miss_needed:
                self.occupied = False
                return "off"

        # 兜底：长时间没有任何确认，强制释放（防止状态卡死）
        if self.occupied and self.last_confirm and now - self.last_confirm > self.hold:
            self.occupied = False
            log.warning("超过 %s 没有新的确认，兜底释放", self.hold)
            return "off"
        return None


# ------------------------------ HA 上报 ------------------------------
def push_ha(cfg: dict, state: str, attrs: dict, dry_run: bool) -> None:
    url = cfg.get("url", "").rstrip("/")
    token = cfg.get("token") or os.environ.get("HA_TOKEN", "")
    entity = cfg.get("entity_id", "binary_sensor.dining_seat_occupied")

    if dry_run:
        log.info("[dry-run] 将写入 %s = %s  attrs=%s", entity, state, attrs)
        return
    if not url or not token:
        log.warning(
            "未配置 HA（url/token 为空），跳过上报。可用环境变量 HA_TOKEN 提供令牌"
        )
        return

    endpoint = f"{url}/api/states/{entity}"
    payload = json.dumps({"state": state, "attributes": attrs}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        log.info("已上报 HA：%s = %s", entity, state)
    except urllib.error.HTTPError as exc:
        log.error(
            "HA 返回 %s：%s", exc.code, exc.read().decode("utf-8", "replace")[:200]
        )
    except Exception as exc:
        log.error("上报 HA 失败：%s", exc)


# ------------------------------ 可视化 ------------------------------
def draw(img, rect, kept, dropped, out_path: Path) -> None:
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    cv2.rectangle(
        img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 2
    )
    cv2.putText(
        img,
        "seat zone",
        (int(x1 * w) + 4, int(y1 * h) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    for d in kept:
        bx1, by1, bx2, by2 = d["xyxy"]
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 200, 0), 2)
        cv2.putText(
            img,
            f"SEATED {d['conf']:.2f}",
            (bx1, max(by1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
        )
    for d in dropped:
        bx1, by1, bx2, by2 = d["xyxy"]
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 220), 1)
        cv2.putText(
            img,
            f"skip: {d.get('reason', '')}",
            (bx1, max(by1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 220),
            1,
        )
    cv2.imwrite(str(out_path), img)
    log.info("判定图已保存：%s", out_path)


# ------------------------------ 主流程 ------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="餐厅座位占用检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).with_name("seat_config.yml"),
        help="配置文件，默认同目录 seat_config.yml",
    )
    ap.add_argument("--image", type=Path, help="用本地图片代替摄像头（测试用）")
    ap.add_argument(
        "--once", action="store_true", help="只检测一次就退出（配合 --image 校准坐标）"
    )
    ap.add_argument("--draw", type=Path, help="把座位区和判定框画出来存成图片")
    ap.add_argument("--dry-run", action="store_true", help="不推 HA，只打印")
    ap.add_argument("--conf", type=float, help="覆盖配置里的置信度阈值")
    ap.add_argument("--zone", help="覆盖配置里的座位区，格式 x1,y1,x2,y2（归一化）")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每次检测的明细")
    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config if args.config.is_file() else None)

    mcfg, zcfg, dcfg, hacfg = cfg["model"], cfg["zone"], cfg["debounce"], cfg["ha"]
    if args.conf is not None:
        mcfg["conf"] = args.conf
    if args.zone:
        try:
            zcfg["rect"] = [float(x) for x in args.zone.split(",")]
        except ValueError:
            raise SystemExit("--zone 格式：x1,y1,x2,y2（归一化 0~1）")
    if len(zcfg["rect"]) != 4:
        raise SystemExit("zone.rect 必须是 4 个数：[x1, y1, x2, y2]")

    log.info(
        "座位区 zone=%s，框高上限=%.0f%%，置信度=%.2f",
        zcfg["rect"],
        zcfg["max_box_h_pct"] * 100,
        mcfg["conf"],
    )

    detector = SeatDetector(
        mcfg["path"],
        mcfg["device"],
        int(mcfg["imgsz"]),
        float(mcfg["conf"]),
        zcfg["rect"],
        float(zcfg["max_box_h_pct"]),
    )

    # 取一帧用于预热（GPU 首次推理要编译 kernel，实测约 5 秒）
    first = (
        load_image(args.image)
        if args.image
        else fetch_frame(cfg["source"]["url"], cfg["source"]["timeout"])
    )
    detector.warmup(first)

    if args.once:
        hit, kept, dropped = detector.detect(first)
        print("\n" + "=" * 60)
        print(
            f"图片        : {args.image or cfg['source']['url']}  {first.shape[1]}x{first.shape[0]}"
        )
        print(f"座位区      : {zcfg['rect']}")
        print(f"命中（在座）: {len(kept)} 个")
        for d in kept:
            print(
                f"  ✅ conf={d['conf']:.3f} box={d['xyxy']} 中心={d['center']} 框高占={d['box_h_pct']:.0%}"
            )
        print(f"被排除      : {len(dropped)} 个")
        for d in dropped:
            print(f"  ⛔ conf={d['conf']:.3f} 中心={d['center']} -> {d.get('reason')}")
        print(f"\n判定结果    : {'【在座】' if hit else '【空座】'}")
        print("=" * 60)
        if args.draw:
            draw(first, zcfg["rect"], kept, dropped, args.draw)
        return 0

    occ = Occupancy(
        int(dcfg["hit_needed"]), int(dcfg["miss_needed"]), int(dcfg["hold_minutes"])
    )
    interval = float(dcfg["interval"])
    log.info(
        "开始常驻检测，间隔 %.1fs（hit>=%d 置为在座，miss>=%d 释放，兜底 %d 分钟）",
        interval,
        occ.hit_needed,
        occ.miss_needed,
        int(dcfg["hold_minutes"]),
    )

    while True:
        try:
            img = (
                load_image(args.image)
                if args.image
                else fetch_frame(cfg["source"]["url"], cfg["source"]["timeout"])
            )
            t0 = time.perf_counter()
            hit, kept, dropped = detector.detect(img)
            cost = (time.perf_counter() - t0) * 1000

            if args.verbose:
                log.debug(
                    "本帧 %d 命中 / %d 排除，耗时 %.0fms", len(kept), len(dropped), cost
                )
                for d in dropped:
                    log.debug(
                        "  排除 conf=%.2f 中心=%s -> %s",
                        d["conf"],
                        d["center"],
                        d.get("reason"),
                    )

            changed = occ.update(hit)
            if changed:
                log.info(
                    "状态变化 -> %s（命中 %d 次 / 未命中 %d 次）",
                    changed,
                    occ.hit_streak,
                    occ.miss_streak,
                )
                push_ha(
                    hacfg,
                    changed,
                    {
                        "friendly_name": "餐座位 有人",
                        "device_class": "occupancy",
                        "confidence": kept[0]["conf"] if kept else 0,
                        "box_center": kept[0]["center"] if kept else None,
                        "updated": datetime.now().isoformat(timespec="seconds"),
                    },
                    args.dry_run,
                )

        except KeyboardInterrupt:
            log.info("已停止")
            return 0
        except Exception as exc:
            log.warning("本轮失败，跳过：%s: %s", type(exc).__name__, exc)

        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
