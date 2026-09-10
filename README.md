# 视觉感知服务（vision）

用摄像头 + YOLOv8 感知房间里「谁在哪、在干什么」，把结果作为**状态**发布给 Home Assistant。
**本服务只输出事实，不做决策**——调灯光、播报等动作一律交给 HA 自动化。

```
摄像头 ──> go2rtc ──> 本服务（YOLO 推理）──> Home Assistant ──> 自动化动作
             1 路流                          只发「事实」         （灯光/播报…）
```

---

## 1. 设计原则

### 1.1 只输出事实，不做决策

| | 内容 | 谁负责 |
|---|---|---|
| **事实** | `seat_occupied=on`<br>`people_count=2`<br>`objects=[bowl, cup]`<br>`facing_monitor=on` | **本服务** |
| **决策** | `if people>=2 and bowl → 暖光`<br>`if seat_occupied and facing_monitor → 冷光` | **HA 自动化** |

好处：加新场景 = 视觉侧加一个 detector + HA 侧加一条自动化，**两边互不相干**。

### 1.2 一个服务 + 插件化 detector，不是一个场景一个服务

摄像头只有 **2 路并发流**，N 个服务各拉一路会直接爆掉；且每个都要重复加载模型、
重复 5 秒 GPU 编译、重复写推 HA 的代码。所以：**一个进程抓一帧，喂给所有 detector**。

### 1.3 归一化坐标

所有区域坐标都是 `0~1`，与分辨率无关。**前提是机位固定**（见 §9 坑 9）。

---

## 2. 硬件与环境

| 项 | 型号 / 值 |
|---|---|
| 摄像头 | **TP-Link Tapo C216**（2K 3MP，Pan/Tilt 云台机） |
| 服务器 | Intel N100（4 核 + 核显） |
| 推理 | **OpenVINO + Intel 核显（iGPU）+ FP16** |
| 取帧 | **go2rtc** |
| Python | venv `~/yolo_env`（Python 3.12） |

### 为什么不用摄像头自带的人形侦测

| 相机能力 | 结论 |
|---|---|
| 自带 Person Detection | ❌ **背对镜头坐姿检出很差**，实测不可用 |
| 自带 Motion Detection | ✅ 灵敏，但**事件推不到 HA**（见 §9 坑 6） |
| 自带 Activity Zone | 概念可用，但依赖上面两个，一并放弃 |

→ 最终：**相机只当"网络摄像头"用，检测全交给自己的 YOLO。**

---

## 3. 模型选型（实测基准）

测试图：「背对镜头坐在桌前」，置信度阈值 0.25。

| 模型 | 设备 | 置信度 | 推理耗时 |
|---|---|---|---|
| yolov8n (.pt) | CPU | 0.642 | 78.2 ms |
| yolov8s (.pt) | CPU | 0.886 | 193.9 ms |
| yolov8m (.pt) | CPU | 0.842 | 483.2 ms |
| yolov8n_openvino (FP32) | — | 0.630 | 136.4 ms |
| **yolov8s_openvino (FP16)** | **iGPU** | **0.882** | **83.2 ms** ← 选定 |

**结论**：

1. **预训练完全够用，不需要自训练**（全部远超 0.5 阈值）。
   相机自己检不出「背对坐姿」是它内置 AI 太弱，YOLO 对这个 case 毫无压力。
2. **m 比 s 差**（0.842 < 0.886）——**大模型在特定 case 上不一定更好，别盲目选大**。
3. OpenVINO + FP16 精度几乎无损（0.886 → 0.882），速度提升 2.3 倍。
4. 最终：`yolov8s_openvino_model/`（FP16）+ iGPU，83ms，1Hz 轮询约 8% 占用。

---

## 4. 状态管理（核心设计）

### 4.1 为什么必须是常驻进程，不能是 cron

去抖需要「记住前几次结果」——**连续 2 次命中才算在座**。
cron 跑完就退出，下次什么都不记得，做不到。

> 对比：mail-service 是 cron 模式，状态靠外部（邮件已读标志 + `state/once_per_day.json`）。
> 本项目是常驻模式，状态在内存。**两种模式在同一个系统里共存，按需选择。**

### 4.2 非对称去抖（关键）

```
确认在座的延迟 = hit_needed  × interval
判定离开的延迟 = miss_needed × interval
```

**两个数字故意不对称**：

| 参数 | 推荐值 | 理由 |
|---|---|---|
| `hit_needed` | **2**（2 秒） | 进来要快，坐下立刻有反馈 |
| `miss_needed` | **30**（30 秒） | 出去要慢，防「低头/转身/被挡」导致状态闪断 |

> ⚠️ 早期版本 `miss_needed=2`（2 秒）**太敏感**：低头捡东西 YOLO 掉 1~2 秒 → 状态翻成空 →
> 灯灭 → 抬头又亮 → 疯狂闪烁。已修正为 30。

### 4.3 持久化策略：三层

| 层 | 存哪 | 丢了怎么办 |
|---|---|---|
| 去抖计数器 | 进程内存 | 无所谓，1 秒重新累计 |
| 当前是否在座 | **推给 HA**（HA 负责持久化） | HA 存着 |
| 模型 | 每次重新加载 | 5 秒 GPU 编译，无所谓 |

**唯一需要防的：进程挂了 HA 永不知情**（见 §9 坑 12）。

---

## 5. 目录结构

```
vision/
├── seat_watcher.py       # 主服务：抓帧 → 检测 → 去抖 → 推 HA
├── seat_config.yml       # 配置（区域/模型/去抖/HA）
├── snap_test.py          # 单图调试工具：测模型置信度、对比模型、验证区域过滤
├── go2rtc.yaml           # go2rtc 配置模板 → /etc/go2rtc/go2rtc.yaml
├── go2rtc.service        # go2rtc 的 systemd 单元
└── README.md             # 本文档
```

### 计划中的重构（多场景时）

```
vision/
├── vision_service.py       # 主循环（原 seat_watcher.py 改名）
├── frame_source.py         # 抓帧 + 缓存：一帧喂所有 detector
├── detectors/
│   ├── base.py             # Detector 接口：detect(frame) -> dict
│   ├── person_zone.py      # 座位占用（当前）
│   ├── table_objects.py    # 餐盘/杯子（计划）
│   └── monitor_gaze.py     # 看显示器（计划，需 pose）
└── publishers/ha.py
```

Detector 接口约定：

```python
class Detector:
    name = "person_zone"
    def setup(self, cfg): ...        # 加载模型
    def warmup(self, frame): ...     # 吃掉 GPU 编译
    def detect(self, frame) -> dict: # 返回「事实」，绝不含动作
```

---

## 6. 安装部署

### 6.1 先验证 RTSP（不通就别往下装）

```bash
sudo apt update && sudo apt install -y ffmpeg curl

ffprobe -rtsp_transport tcp "rtsp://<账号>:<密码>@192.168.1.50:554/stream1"
# 期望：Stream #0:0: Video: h264, 2304x1296
```

摄像头侧前置条件：

- Tapo App → 高级设置 → **摄像头账号**（建用户名/密码）
- App 里画质设为 **Best Quality**（否则 stream1 分辨率低）
- **关闭移动追踪 / 自动巡航**（锁定机位）
- 路由器给摄像头 MAC 绑静态 IP

### 6.2 装 go2rtc

```bash
cd /tmp
wget https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_amd64
chmod +x go2rtc_linux_amd64
sudo mv go2rtc_linux_amd64 /usr/local/bin/go2rtc
go2rtc -version
```

配置：

```bash
sudo mkdir -p /etc/go2rtc
sudo cp go2rtc.yaml /etc/go2rtc/go2rtc.yaml
sudo vi /etc/go2rtc/go2rtc.yaml    # 填账号/密码/IP
sudo cp go2rtc.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now go2rtc
sudo systemctl status go2rtc
```

验证取帧：

```bash
curl -s -o /tmp/t.jpg "http://127.0.0.1:1984/api/frame.jpeg?src=dining"
file /tmp/t.jpg      # 必须是 JPEG，且分辨率 ≥1080p
```

### 6.3 Python 环境

```bash
python3 -m venv ~/yolo_env
source ~/yolo_env/bin/activate
pip install ultralytics openvino pyyaml opencv-python

# 导出模型（FP16，核显用）
yolo export model=yolov8s.pt format=openvino half=True
```

---

## 7. 使用

```bash
source ~/yolo_env/bin/activate
```

### 单张测试（校准用，不推 HA）

```bash
# 你坐着的画面 -> 应判定【在座】
python seat_watcher.py --image /tmp/t.jpg --once

# 画区域图：黄=座位区，绿=判定在座，红=被排除（标原因）
python seat_watcher.py --image /tmp/t.jpg --once --draw /tmp/zone.png
```

### 常驻运行

```bash
python seat_watcher.py                 # 读 seat_config.yml
python seat_watcher.py --dry-run       # 不推 HA，只看判定
python seat_watcher.py -v              # 打印每帧明细
```

### 模型/区域调试工具

```bash
python snap_test.py --info                       # 环境诊断（确认核显可用）
python snap_test.py --image t.jpg --device intel:gpu --models n s m
python snap_test.py --image t.jpg --zone 0.62,0.44,0.88,0.84 --device intel:gpu
```

`--zone`（推荐）与 `--roi` 的区别：

| | 行为 | 用途 |
|---|---|---|
| `--zone` | **不裁剪**，全图检测后按框中心过滤 | 正式逻辑（= 摄像头 Activity Zone） |
| `--roi` | 真把图片裁掉再检测 | 仅对比用 |

---

## 8. 配置参考（seat_config.yml）

| 段 | 键 | 说明 |
|---|---|---|
| `source` | `url` | go2rtc 快照地址 |
| `model` | `path` / `device` / `conf` | `yolov8s_openvino_model/` / `intel:gpu` / `0.5` |
| `zone` | `rect` | 座位区 `[x1,y1,x2,y2]` 归一化 |
| `zone` | `max_box_h_pct` | **框高占画面上限**，超过判为「站着」<br>实测坐姿 0.39、站姿约 0.68，取 0.5 |
| `debounce` | `hit_needed` / `miss_needed` | 见 §4.2，推荐 2 / 30 |
| `debounce` | `interval` | 检测间隔（秒），1.0 |
| `ha` | `url` / `token` / `entity_id` | token 留空则读环境变量 `HA_TOKEN` |

### 判定逻辑（三层过滤）

1. 框中心落在 `zone.rect` 内 → **只认这个位置**
2. `框高/画面高 < max_box_h_pct` → **坐着而非站着**
3. 置信度 ≥ `conf`

---

## 9. 已解决的坑（务必阅读）

| # | 坑 | 说明 / 对策 |
|---|---|---|
| 1 | **`half=True` 放错地方** | OpenVINO IR 的精度在**导出时**烘焙，预测时传 `half` 是空操作。<br>必须 `yolo export ... format=openvino half=True` |
| 2 | **`intel:gpu` 只对 OpenVINO 模型有效** | 传给 `.pt` 会抛 `Invalid device string`（torch 不认）。<br>本项目已改为：GPU 请求只设 `OPENVINO_DEVICE` 环境变量，不传 `device=` |
| 3 | **首次推理 5.1 秒** | GPU kernel 编译，一次性。**服务必须预热**，否则第一次检测卡 5 秒 |
| 4 | **摄像头只有 2 路并发流** | HA 预览 + 本服务各拉一路就满。go2rtc 单源多路复用 |
| 5 | **裁剪不提速** | YOLO 无论输入多大都 resize 到 640×640，计算量恒定。<br>裁剪只改变「人在输入里的相对大小」（影响精度）。想提速只能降 `imgsz` 或换小模型 |
| 6 | **检测结果推不进 HA** | Tapo 集成的 `binary_sensor` 是**动态创建**的，只在收到第一个事件后才注册；<br>且事件靠 webhook（HA 必须 HTTP 不能 HTTPS）、ONVIF 2020 端口，部分固件 pullpoint 是坏的。<br>**→ 已完全绕开，不依赖相机事件** |
| 7 | **没有"只检测某区域"的参数** | ultralytics 无此功能。**检测后按框中心过滤**是唯一做法 |
| 8 | **SD卡 / Tapo Care / RTSP 三选二** | 前两个同时开会禁用第三方 RTSP |
| 9 | **机位不能动** | zone 是画面坐标，云台一转全废。**必须关移动追踪、锁预置位** |
| 10 | **RTSP 密码特殊字符** | 必须 URL 编码：`@`→`%40` `:`→`%3A` `#`→`%23` `/`→`%2F` |
| 11 | **go2rtc 默认无鉴权** | API 绑 `127.0.0.1:1984`；要对局域网开放必须配 `username/password` |
| 12 | **服务挂了 HA 永不知情** | sensor 会永远停在 `on`。HA 侧必须加新鲜度判断：<br>`(now() - last_updated).total_seconds() < 120` |
| 13 | **判断是否真在核显上** | 看日志 `Using OpenVINO LATENCY mode ... on GPU.0`；<br>別只看耗时（大模型跑 GPU ≈ 小模型跑 CPU，数字会骗人） |

---

## 10. Roadmap

### 近期

- [ ] systemd 常驻：开机自启 + `Restart=always` + 启动时从 HA 读回状态（避免重启瞬间闪断）
- [ ] 空座位负样本验证（必须判定【空座】）
- [ ] 多时段验证：白天 / 傍晚开灯 / **深夜红外** / 阴天

### 多场景扩展

| 场景 | 需要 | HA 动作 |
|---|---|---|
| 座位占用 | ✅ 已完成 | 播报 / 灯光 |
| 餐桌有餐盘 | COCO `bowl`/`cup` 类 | 暖光 |
| 看显示器 | **需 pose 或 gaze 模型** | 冷光 |
| 在场人数 | 数 person 框 | 亮度档位 |

每个场景：新增 `detectors/xxx.py` + 一条 HA 自动化，**主循环不用改**。

---

## 11. 与其他系统的关系

本服务与 `mail-service` **完全解耦**，仅通过 HA 交集：

```
vision  ──写──> HA: binary_sensor.dining_seat_occupied
                        ↑
mail-service ──读───────┘（播报前检查是否有人）
```

mail-service 侧计划加通用的 `actions/ha-condition-action.py`：
读 stdin 文本原样透传，查 HA 实体状态，不满足则 `exit 2`（约定：优雅跳过，不算失败）。
这样「人不在 → 跳过 → 邮件保持未读 → 下轮 cron 自动重试」，无需额外重试机制。
