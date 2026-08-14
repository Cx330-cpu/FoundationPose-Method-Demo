# FoundationPose 实时姿态估计项目交接文档

> 交接对象：下一位开发人员  
> 更新时间：2026-08-14  
> 项目路径：`/home/brandon/projects/FoundationPose-main`  
> 当前状态：PC 端实时架构与 FPFRAME v1 网络链路已验证收口；Unity/iPhone sender 脚本包已新增，等待导入 Unity Editor 做真机编译与 Phase A-E 验证。

---

## 1. 当前阶段总览

本项目目标是用 FoundationPose 在 PC GPU 上估计真实物体 6DoF 姿态，后续由 Unity/iPhone 提供 RGB、LiDAR depth、camera intrinsics 和首帧 object mask。

当前已经完成两条主线：

1. **PC FoundationPose live 主链**

```text
FrameProvider
  -> FramePacket
  -> first frame register()
  -> following frames track_one()
  -> PC GUI visualization
  -> pose txt output
```

2. **FPFRAME v1 TCP 网络输入**

```text
Unity/iPhone or local sender
  -> FPFRAME v1 TCP
  -> NetworkFrameProvider
  -> FoundationPose
```

当前 Unity/iPhone 阶段只实现单向 frame streaming：

```text
iPhone / Unity
  -> RGB + raw LiDAR depth + K + first-frame mask
  -> FPFRAME v1 TCP
  -> PC NetworkFrameProvider
  -> FoundationPose
  -> PC GUI 3D box / axis
```

本阶段暂不做：

- pose 回传 Unity；
- Unity / ARKit 坐标转换；
- tracking-loss recovery；
- iPhone async queue / pose ACK；
- YOLO-Seg / SAM 等真 segmentation；
- 改 FoundationPose 内核。

---

## 2. 当前仓库结构

关键文件：

```text
FoundationPose-main/
├── run_demo.py                         # 官方 baseline，保留不改
├── run_live.py                         # 新 live loop，支持 recorded / network provider
├── network_frame_protocol.py           # FPFRAME v1 编解码、TCP recv_exact、协议校验
├── send_recorded_frames.py             # 本地 recorded TCP sender，用 mustard0 做网络回归
├── check_network_packets.py            # recorded vs network packet exactness 检查
├── inspect_fpframe_stream.py           # Unity synthetic sender / FPFRAME 诊断接收器
├── datareader.py                       # 官方 reader，保留不改
├── estimater.py                        # FoundationPose 内核，保留不改
├── Utils.py                            # 官方工具，保留不改
├── demo_data/
│   └── mustard0/                       # 当前 regression 数据源
└── unity_sender/
    ├── README.md
    └── Assets/Scripts/FoundationPose/
        ├── FPFrameProtocol.cs
        ├── FoundationPoseTcpSender.cs
        ├── FoundationPoseFrameStreamer.cs
        ├── FoundationPoseSyntheticSender.cs
        └── YoloBBoxToMaskAdapter.cs
```

注意：

- 本仓库当前不是 Unity 工程，没有 `Packages/manifest.json`、`ProjectSettings/` 或现有 `.cs` 项目脚本。
- `unity_sender/` 是可移植 Unity 脚本包，需要拷入真实 Unity 项目后编译。
- `PointCloudCaptureDemo.cs` 未在本仓库内找到；后续若真实 Unity 工程中存在它，只应做最小挂接，不要大范围重构。

---

## 3. PC 端实时架构

### 3.1 `FramePacket`

定义位置：`run_live.py`

字段：

```python
FramePacket(
  rgb: np.ndarray,          # H x W x 3, uint8, RGB
  depth: np.ndarray,        # H x W, meters
  K: np.ndarray,            # 3 x 3 camera intrinsics
  mask: Optional[np.ndarray],
  timestamp: Optional[float],
  frame_id: str,
  index: int,
)
```

约束：

- `rgb` 必须是 RGB，不是 BGR。
- `depth` 单位是 meters。
- `depth.shape == rgb.shape[:2]`。
- `K` 必须对应当前 `rgb/depth/mask` 的最终像素分辨率。
- `mask` 只要求注册帧提供，后续 tracking 帧为 `None`。

### 3.2 `FrameProvider`

定义位置：`run_live.py`

接口：

```python
class FrameProvider:
  def get_frame(self) -> Optional[FramePacket]:
    ...
```

当前实现：

- `RecordedFoundationPoseProvider`
  - 读取 `demo_data/mustard0`。
  - 复刻官方 `YcbineoatReader(video_dir=..., shorter_side=None, zfar=np.inf)`。
  - 第 0 帧提供 mask，后续 mask 为 `None`。

- `NetworkFrameProvider`
  - TCP server。
  - 接收 FPFRAME v1。
  - clean EOF 返回 `None`。
  - frame 内 EOF 或协议错误抛异常。

### 3.3 Live loop 行为

`run_live.py` 保持官方 FoundationPose 调用参数：

```text
set_seed(0)
est_refine_iter = 5
track_refine_iter = 2
```

状态机：

```text
UNREGISTERED
  -> first packet must have mask
  -> register(K, rgb, depth, ob_mask, iteration=5)
  -> TRACKING
  -> track_one(rgb, depth, K, iteration=2)
```

GUI：

- 窗口名：`FoundationPose Live`
- 显示：
  - 3D box；
  - XYZ axis；
  - frame id；
  - state；
  - operation；
  - processing FPS。
- overlay 画在 `packet.rgb.copy()` 上，不修改原始 `FramePacket.rgb`。

pose 输出：

```text
{debug_dir}/ob_in_cam/{frame_id}.txt
```

---

## 4. PC CLI 参数文档

### 4.1 `run_live.py`

```bash
python run_live.py [options]
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mesh_file` | `demo_data/mustard0/mesh/textured_simple.obj` | 目标物 mesh |
| `--test_scene_dir` | `demo_data/mustard0` | recorded provider 数据目录 |
| `--est_refine_iter` | `5` | 首帧 register refinement iterations |
| `--track_refine_iter` | `2` | 后续 tracking refinement iterations |
| `--debug` | `1` | `>=1` 时显示 GUI |
| `--debug_dir` | `debug_live` | pose / 可视化输出目录 |
| `--save_track_vis` | false | 保存每帧可视化图 |
| `--provider` | `recorded` | `recorded` 或 `network` |
| `--host` | `0.0.0.0` | network provider TCP listen host |
| `--port` | `5000` | network provider TCP listen port |

常用命令：

```bash
python run_live.py --provider recorded --debug 0 --debug_dir debug_live_recorded_check
```

```bash
python run_live.py --provider network --host 0.0.0.0 --port 5000 --debug 1 --debug_dir debug_live_iphone
```

### 4.2 `send_recorded_frames.py`

用途：本地模拟 TCP client，读取 `demo_data/mustard0`，发送 FPFRAME v1 给 PC server。

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--host` | `127.0.0.1` | PC server host |
| `--port` | `5000` | PC server port |
| `--test_scene_dir` | `demo_data/mustard0` | recorded 数据源 |
| `--rgb_codec` | `png` | `png` 或 `jpeg` |
| `--jpeg_quality` | `95` | JPEG 质量 |
| `--fps` | `0` | `0` 表示不限速 |

示例：

```bash
python send_recorded_frames.py --host 127.0.0.1 --port 5000 --rgb_codec png
```

### 4.3 `check_network_packets.py`

用途：验证 recorded provider 与 network 解码后的 packet 是否 exact。

检查内容：

- RGB exact；
- depth exact；
- K exact；
- mask exact；
- 对应 max diff。

### 4.4 `inspect_fpframe_stream.py`

用途：接 Unity sender 的 FPFRAME v1，不进入 FoundationPose 主链，用于 Phase A-D 输入诊断。

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--host` | `0.0.0.0` | listen host |
| `--port` | `5000` | listen port |
| `--num_frames` | `1` | 接收帧数 |
| `--save_dir` | `debug_fpframe_inspect` | 保存 RGB/depth/mask/K |
| `--expect_synthetic` | false | 与 Unity synthetic sender 的期望值逐项 exact 比较 |

示例：

```bash
python3 inspect_fpframe_stream.py --host 0.0.0.0 --port 5000 --num_frames 3 --expect_synthetic
```

---

## 5. FPFRAME v1 协议文档

定义位置：

- PC：`network_frame_protocol.py`
- Unity：`unity_sender/Assets/Scripts/FoundationPose/FPFrameProtocol.cs`

### 5.1 消息结构

```text
4 bytes big-endian uint32 header_len
JSON header bytes, UTF-8
RGB blob
Depth blob
Optional mask blob
```

### 5.2 Header 字段

```json
{
  "magic": "FPFRAME",
  "version": 1,
  "frame_id": "string",
  "index": 0,
  "timestamp": 0.0,
  "width": 256,
  "height": 192,
  "fx": 100.0,
  "fy": 100.0,
  "cx": 128.0,
  "cy": 96.0,
  "rgb_format": "png",
  "depth_format": "uint16_png_mm",
  "mask_format": "uint8_png",
  "rgb_len": 12345,
  "depth_len": 12345,
  "mask_len": 1234
}
```

字段约束：

- `magic == "FPFRAME"`。
- `version == 1`。
- `index` 必须是非负 int。
- `width/height` 必须是正 int。
- `fx/fy` 必须是有限正数。
- `cx/cy` 必须是有限数字。
- `timestamp` 允许 `None`，否则必须是有限数字。
- `rgb_len/depth_len > 0`。
- `mask_len >= 0`。
- `rgb_format in ["png", "jpeg"]`。
- `depth_format == "uint16_png_mm"`。
- `mask_len == 0` 时 `mask_format == "none"`。
- `mask_len > 0` 时 `mask_format == "uint8_png"`。
- `header_len <= 1 MB`。

### 5.3 Blob 编码

RGB：

- Unity / sender 语义是 RGB；
- PNG 用于 exact correctness；
- JPEG 只用于 smoke / 性能测试；
- PC 端 OpenCV 解码后转换回 RGB。

Depth：

```text
meters -> round(meters * 1000) -> uint16 millimeters -> PNG
```

PC 端恢复：

```python
depth_mm = cv2.imdecode(depth_bytes, cv2.IMREAD_UNCHANGED)
depth = depth_mm / 1e3
```

Mask：

- `uint8 PNG`；
- object area 为 `255`；
- background 为 `0`；
- PC 端转成 bool。

### 5.4 TCP EOF 语义

PC receiver 已区分：

- 读取下一帧 `4-byte header_len` 前 socket 正常关闭：正常 stream end。
- 已开始读取某一帧 header/blob 后 EOF：`ProtocolError`。

---

## 6. Unity/iPhone Sender 脚本说明

Unity 脚本包位置：

```text
unity_sender/Assets/Scripts/FoundationPose/
```

### 6.1 `FPFrameProtocol.cs`

职责：

- 生成 FPFRAME v1 header；
- 写入 4-byte big-endian header length；
- 编码 RGB PNG/JPEG；
- 编码 depth uint16 PNG/mm；
- 编码 optional mask PNG；
- 生成完整 TCP message bytes。

核心接口：

```csharp
FPFrameProtocol.BuildFrameMessage(
  byte[] rgb24,
  ushort[] depthMillimeters,
  byte[] maskU8,
  int width,
  int height,
  FPCameraIntrinsics intrinsics,
  string frameId,
  int index,
  double timestamp,
  FPRgbCodec rgbCodec,
  int jpegQuality)
```

约束：

- `rgb24.Length == width * height * 3`。
- `depthMillimeters.Length == width * height`。
- `maskU8 == null` 或 `maskU8.Length == width * height`。
- `fx/fy` 必须有限且大于 0。

### 6.2 `FoundationPoseTcpSender.cs`

职责：

- TCP client；
- 连接 PC `NetworkFrameProvider`；
- 后台线程发送；
- REGISTER 首帧不可丢；
- TRACKING 帧 latest-frame-wins；
- 本阶段不接收 pose。

状态机：

```text
Stopped
Connecting
WaitingForRegistrationFrame
SendingRegistrationFrame
TrackingStream
Error
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `host` | `127.0.0.1` | PC server IP |
| `port` | `5000` | PC server port |
| `connectOnStart` | true | Start 时自动连接 |
| `connectTimeoutMs` | `3000` | TCP connect timeout |
| `sendTimeoutMs` | `3000` | TCP write timeout |
| `verboseLogging` | true | 输出发送日志 |

关键行为：

- `EnqueueRegistrationFrame(...)` 仅在 `WaitingForRegistrationFrame` 接收；
- register packet 写入 socket 成功后才切到 `TrackingStream`；
- `EnqueueTrackingFrame(...)` 在 `TrackingStream` 中覆盖未发送旧 tracking frame；
- `StopSender()` 先关闭 socket，再 join worker，避免阻塞 IO 残留线程。

### 6.3 `FoundationPoseFrameStreamer.cs`

职责：

- 从 AR Foundation 获取 RGB；
- 从 AR Foundation 获取 raw environment depth；
- 每帧获取 intrinsics；
- 校验 RGB/depth timestamp；
- 使用 raw depth native resolution 作为最终输出分辨率；
- RGB downsample 到 depth resolution；
- K 缩放到最终 resolution；
- 首帧生成 mask，后续不发送 mask；
- 把 frame 交给 TCP sender。

使用 API：

```csharp
ARCameraManager.TryAcquireLatestCpuImage(out XRCpuImage rgbImage)
AROcclusionManager.TryAcquireRawEnvironmentDepthCpuImage(out XRCpuImage depthImage)
ARCameraManager.TryGetIntrinsics(out XRCameraIntrinsics intrinsics)
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `maxRgbDepthDeltaMs` | `50.0` | RGB/depth timestamp 最大差值 |
| `maxAspectRatioDelta` | `0.01` | RGB/depth 宽高比最大差值 |
| `targetFps` | `5.0` | 采集目标 FPS |
| `rgbCodec` | `Png` | RGB 编码 |
| `jpegQuality` | `95` | JPEG 质量 |
| `verboseLogging` | true | 输出采集/丢帧日志 |

分辨率策略：

```text
finalWidth = rawDepthWidth
finalHeight = rawDepthHeight
RGB -> downsample to finalWidth x finalHeight
Depth -> keep raw native resolution
Mask -> finalWidth x finalHeight
K -> scale to finalWidth x finalHeight
```

K 缩放：

```csharp
scaleX = finalWidth / intrinsics.resolution.x
scaleY = finalHeight / intrinsics.resolution.y

fx = intrinsics.focalLength.x * scaleX
fy = intrinsics.focalLength.y * scaleY
cx = intrinsics.principalPoint.x * scaleX
cy = intrinsics.principalPoint.y * scaleY
```

丢帧原因会在 Unity log 中输出，例如：

- `sender_not_ready`
- `missing_intrinsics`
- `missing_rgb_cpu_image`
- `missing_raw_depth_cpu_image`
- `rgb_depth_timestamp_delta_too_large`
- `rgb_depth_aspect_mismatch`
- `registration_mask_unavailable`

### 6.4 `YoloBBoxToMaskAdapter.cs`

职责：

- 接收现有 CoreML / Vision YOLO bbox；
- 转换为最终 RGB 像素坐标；
- rasterize rectangle binary mask；
- 仅用于首帧 register 验证。

核心接口：

```csharp
SetBoundingBox(Rect newBBox, double timestamp, FPBBoxCoordinateSpace space)
```

坐标空间：

```csharp
FinalImagePixelsTopLeft
NormalizedTopLeft
NormalizedBottomLeft
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `maxMaskAgeMs` | `50.0` | YOLO bbox 与 RGB frame 最大 timestamp 差 |
| `bboxInflation` | `1.0` | bbox 膨胀比例 |

注意：

- rectangle mask 只用于验证数据链和 register/track 是否能跑。
- 不要用 bbox mask 的姿态精度评判 FoundationPose 最终效果。
- 后续应升级 YOLO-Seg / SAM / 真 object segmentation。

### 6.5 `FoundationPoseSyntheticSender.cs`

职责：

- 不依赖 AR Foundation；
- 发送 synthetic RGB/depth/K/mask；
- 用于 Phase A 验证 Unity FPFRAME v1 是否被 PC exact decode。

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `width` | `256` | synthetic frame width |
| `height` | `192` | synthetic frame height |
| `fps` | `2.0` | synthetic 发送 FPS |
| `rgbCodec` | `Png` | RGB 编码 |
| `jpegQuality` | `95` | JPEG 质量 |

---

## 7. 已完成验证记录

### 7.1 PC recorded/live regression

此前完成：

```text
Recorded:
737/737
allclose_all_frames = True
max_abs_diff = 0
```

说明：

- `run_live.py --provider recorded` 与官方 `run_demo.py` 数值等价。
- `run_demo.py` 继续作为 regression baseline，保留不改。

### 7.2 Network PNG regression

此前完成：

```text
Network PNG:
737/737
allclose_all_frames = True
max_abs_diff = 0
```

Packet exactness：

```text
RGB exact = True
Depth exact = True
K exact = True
Mask exact = True
```

说明：

- recorded 数据绕 TCP/FPFRAME 一圈后，FoundationPose pose 仍完全一致。
- 证明 PC network layer 没有改变 FoundationPose 输入。

### 7.3 JPEG smoke test

此前完成：

```text
Network JPEG smoke:
737 pose files generated
stream ended normally
```

说明：

- JPEG 仅验证网络收发和推理链路可跑；
- 不要求数值等价。

### 7.4 当前静态检查

已通过：

```bash
PYTHONPYCACHEPREFIX=/tmp/fp_pycache python3 -m py_compile \
  run_live.py \
  network_frame_protocol.py \
  send_recorded_frames.py \
  check_network_packets.py \
  inspect_fpframe_stream.py
```

已检查：

```bash
grep -RIn "[[:blank:]]$" unity_sender inspect_fpframe_stream.py
```

无行尾空白。

---

## 8. Unity/iPhone Phase A-E 验证计划

### Phase A：Synthetic FPFRAME 验证

PC：

```bash
python3 inspect_fpframe_stream.py \
  --host 0.0.0.0 \
  --port 5000 \
  --num_frames 3 \
  --expect_synthetic
```

Unity：

- 添加 `FoundationPoseTcpSender`；
- 添加 `FoundationPoseSyntheticSender`；
- 设置 PC IP / port；
- 点击 Play 或真机运行。

预期：

```text
RGB exact = True
Depth exact = True
K exact = True
Mask exact = True
```

如果 depth exact 失败，优先检查 Unity `ImageConversion.EncodeArrayToPNG` 对 `GraphicsFormat.R16_UNorm` 的实际编码结果。

### Phase B：真实 RGB + K 验证

目标：

- 验证 `ARCameraManager.TryAcquireLatestCpuImage` 输出；
- 验证 RGB 方向；
- 验证最终分辨率；
- 验证 K 缩放。

PC：

```bash
python3 inspect_fpframe_stream.py --host 0.0.0.0 --port 5000 --num_frames 10
```

检查：

- RGB 没有明显旋转/翻转；
- header width/height 与图像一致；
- K 数值合理；
- Unity log 中 `final=WxH` 与 depth native resolution 一致。

### Phase C：Raw depth 同步验证

目标：

- 验证 `TryAcquireRawEnvironmentDepthCpuImage`；
- 验证 RGB/depth timestamp delta；
- 验证 invalid depth；
- 验证 RGB/depth 对齐。

检查：

- Unity log 中 `delta_ms <= 50`；
- `invalid_depth` 计数合理；
- PC 保存的 `*_depth_mm.png` 值范围合理；
- depth 边缘与 RGB 物体边缘大致对应。

### Phase D：YOLO bbox mask 验证

目标：

- 验证现有 CoreML / Vision YOLO bbox 能正确进入 `YoloBBoxToMaskAdapter`；
- 验证 bbox 坐标转换；
- 验证首帧 mask。

检查：

- PC 保存的 mask 覆盖目标；
- mask 无旋转、翻转、缩放错位；
- Unity log 中 register frame 带 mask；
- 后续 tracking frame 不带 mask。

### Phase E：FoundationPose 真机 smoke test

PC：

```bash
python run_live.py \
  --provider network \
  --host 0.0.0.0 \
  --port 5000 \
  --debug 1 \
  --debug_dir debug_live_iphone
```

Unity/iPhone：

- 发送真实 RGB + raw depth + K + first-frame mask。

预期：

- PC 首帧进入 `REGISTER`；
- 后续进入 `TRACKING`；
- GUI 显示 3D box 和 XYZ axis；
- pose txt 持续输出。

注意：

- bbox rectangle mask 带背景、桌面、手是预期风险；
- 此阶段只判断链路、同步、坐标和 register/track 是否能跑；
- 不用 bbox mask 的姿态精度评价最终效果。

---

## 9. 开发约束

除非发现明确 bug，不要再动：

```text
run_demo.py
estimater.py
datareader.py
Utils.py
FoundationPose 内核
已验证的 PC live loop
已验证的 FPFRAME v1 PC receiver
```

不要在当前阶段扩展：

- pose 回传；
- Unity 坐标转换；
- iPhone/Unity ACK；
- async latest-frame server；
- tracking lost recovery；
- segmentation 模型替换；
- depth upsample 到 640。

当前第一版真实 iPhone sender 的核心策略是：

```text
correctness first
raw depth native resolution first
PNG first
REGISTER first frame non-droppable
tracking latest-frame-wins
```

---

## 10. 已知风险与排查优先级

### 10.1 Unity C# 编译风险

本仓库没有 Unity 工程和 C# 编译器，因此 Unity 脚本尚未在 Editor 中编译。

导入 Unity 后第一优先级：

- AR Foundation package 版本是否支持：
  - `TryAcquireRawEnvironmentDepthCpuImage`
  - `XRCpuImage.Format.DepthFloat32`
  - `XRCpuImage.Format.DepthUint16`
  - `ImageConversion.EncodeArrayToPNG`
  - `GraphicsFormat.R16_UNorm`

如果 API 名称不兼容，按当前语义做最小适配，不要改变协议。

### 10.2 Depth PNG exactness

风险：

- Unity `R16_UNorm` PNG 编码是否被 OpenCV exact 解成原始 `uint16 mm` 需要实测。

排查：

- 先跑 Phase A synthetic exactness。
- 如果 depth max diff 非 0，优先替换 Unity depth PNG 编码实现。

### 10.3 RGB/depth aspect ratio

当前策略：

- 如果 RGB/depth 宽高比不一致，直接丢帧。

原因：

- 第一版不做 crop。
- 避免 RGB 被拉伸导致 K/depth 几何不一致。

后续若需要支持不同比例：

- 必须显式 center-crop RGB；
- 同步修正 K 的 `cx/cy`；
- mask/bbox 同步转换。

### 10.4 图像方向 / 垂直翻转

当前策略：

- 不从 screen texture 截图；
- 不为了 GUI 单独旋转 RGB；
- 以最终发送像素数组为统一坐标系。

排查：

- Phase B 看 PC 保存 RGB；
- Phase D 看 mask overlay；
- 如果发现翻转，RGB/depth/mask/K 必须整体一致修正。

### 10.5 YOLO bbox 坐标

风险：

- Vision normalized bbox 可能是 bottom-left origin；
- Unity texture / screen coords 可能和 camera CPU image 不一致。

当前支持：

```csharp
FinalImagePixelsTopLeft
NormalizedTopLeft
NormalizedBottomLeft
```

排查：

- Phase D 只看 mask 是否贴住目标。

### 10.6 REGISTER 首帧

当前保护：

- register packet non-droppable；
- register packet 完整 send 成功后才进入 tracking；
- tracking 阶段才 latest-frame-wins。

限制：

- 当前没有 PC pose ACK；
- sender 只能知道 TCP 完整写入成功，不能知道 FoundationPose register 是否成功。

---

## 11. 下一位开发人员建议路线

建议按下面顺序推进，不要跳步：

1. 把 `unity_sender/Assets/Scripts/FoundationPose/` 拷入 Unity 工程。
2. 解决 Unity Editor 编译错误。
3. 跑 Phase A synthetic exactness。
4. 接 `ARCameraManager`，跑 Phase B RGB + K。
5. 接 `AROcclusionManager` raw depth，跑 Phase C。
6. 把现有 YOLO bbox 接到 `YoloBBoxToMaskAdapter.SetBoundingBox(...)`，跑 Phase D。
7. 启动 PC `run_live.py --provider network`，跑 Phase E。
8. 记录每阶段日志、截图、PC 保存图像和 pose 输出。
9. 若 Phase E 能 register/track，再考虑 segmentation mask 升级。
10. 最后再设计 pose 回传和 Unity 坐标转换。

---

## 12. 当前结论

PC 端已经是稳定 baseline：

```text
run_live.py recorded == run_demo.py
network PNG == recorded
737/737
allclose_all_frames = True
max_abs_diff = 0
```

Unity/iPhone 端已经有第一版 sender 脚本包，但还未经过 Unity Editor / iPhone 真机验证。

下一步的核心不是继续改 PC 端，而是：

```text
Unity compile
-> synthetic FPFRAME exactness
-> real RGB/K
-> raw depth
-> YOLO bbox mask
-> FoundationPose live smoke test
```
