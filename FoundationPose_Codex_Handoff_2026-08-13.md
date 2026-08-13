# FoundationPose 实时姿态估计项目交接文档

> 交接对象：Codex  
> 更新时间：2026-08-13  
> 当前阶段：FoundationPose 官方 model-based demo 已在 RTX 3070 + WSL2 + Docker 环境成功跑通，并已确认 OpenCV/WSLg 可视化窗口正常。下一阶段开始开发实时输入版本，用 iPhone RGB + LiDAR Depth + Camera Intrinsics + Object Mask 驱动 FoundationPose，并最终把 6DoF Pose 回传 Unity。

---

## 1. 项目目标

当前暑期科研项目目标是实现一个 **AR Object Replacement System**：

- 使用摄像头识别真实物体；
- 获取真实物体的 6DoF Pose（位置 + 旋转）；
- 在 Unity / ARKit 中用虚拟模型覆盖或替换真实物体；
- 目标设备为 iPhone，测试机为 iPhone 16 Pro Max；
- iPhone 可提供：
  - RGB Camera Frame；
  - ARKit / LiDAR Depth；
  - Camera Intrinsics；
  - AR Camera Transform；
- FoundationPose 在 Windows PC 的 NVIDIA RTX 3070 上负责 GPU Pose Estimation。

当前决定：

**不再把 iPhone LiDAR Point Cloud 作为主要的几何姿态估计方法。**

新的目标流程是：

```text
iPhone / Unity
│
├── RGB
├── LiDAR Depth
├── Camera Intrinsics K
└── Object Detection / Mask
        │
        │ LAN / Wi-Fi
        ▼
RTX 3070 PC
│
└── FoundationPose
    ├── register()
    └── track_one()
        │
        ▼
4×4 Object-in-Camera Pose
        │
        │ LAN / Wi-Fi
        ▼
Unity / ARKit
        │
        ▼
Coordinate Conversion
        │
        ▼
Virtual Object Replacement
```

---

# 2. 当前机器环境

## 2.1 Windows

操作系统：

```text
Windows 11
```

GPU：

```text
NVIDIA GeForce RTX 3070
VRAM: 8 GB
```

Windows `nvidia-smi` 曾确认：

```text
NVIDIA-SMI 610.88
CUDA UMD 13.3
RTX 3070
```

注意：

Windows 驱动显示 CUDA 13.3 并不意味着 FoundationPose Docker 内必须使用 CUDA 13.3。

官方 Docker 内可以继续使用较老 CUDA / PyTorch 环境，RTX 3070（Ampere / sm_86）兼容。

---

## 2.2 WSL

发行版：

```text
Ubuntu-22.04
WSL2
```

项目路径：

```text
/home/brandon/projects/FoundationPose-main
```

当前正常的 Ubuntu prompt 类似：

```text
brandon@localhost:~/projects/FoundationPose-main$
```

项目应始终尽量放在：

```text
/home/brandon/...
```

而不是：

```text
/mnt/c/...
```

原因：

- Linux 文件系统编译速度更好；
- CUDA/C++ 扩展构建更稳定；
- 权限问题更少。

---

# 3. FoundationPose Repository 状态

当前项目目录：

```text
/home/brandon/projects/FoundationPose-main
```

关键文件已经存在：

```text
FoundationPose-main/
├── LICENSE
├── Utils.py
├── build_all.sh
├── build_all_conda.sh
├── bundlesdf/
├── datareader.py
├── docker/
├── environment.yml
├── estimater.py
├── learning/
├── mycpp/
├── offscreen_renderer.py
├── requirements.txt
├── run_demo.py
├── run_linemod.py
├── run_ycb_video.py
├── weights/
└── demo_data/
```

有一个额外的空目录：

```text
FoundationPose-main/FoundationPose-main
```

执行过：

```bash
ls FoundationPose-main
```

没有输出，因此该子目录是空的。

它不是当前实际 repo root，可以忽略或删除。

---

# 4. Model Weights

权重已经放好：

```text
weights/
├── 2023-10-28-18-33-37
└── 2024-01-11-20-02-45
```

分别对应 FoundationPose 的 refiner / scorer 模型。

Windows 下载文件曾生成类似：

```text
*.zip:Zone.Identifier
```

这类文件不是模型内容，可忽略或删除。

---

# 5. Demo Data

官方 demo data 已经准备好：

```text
demo_data/
├── mustard0
└── kinect_driller_seq
```

主要测试：

```text
demo_data/mustard0
```

官方默认 mesh：

```text
demo_data/mustard0/mesh/textured_simple.obj
```

---

# 6. Docker

官方镜像已经成功下载。

当前：

```bash
docker images | grep foundationpose
```

曾输出：

```text
foundationpose:latest                 22837f81c0cf   31.7GB   11GB
wenbowen123/foundationpose:latest     22837f81c0cf   31.7GB   11GB
```

两个 tag 指向同一个 Docker Image ID：

```text
22837f81c0cf
```

所以不会实际占用两份镜像空间。

---

# 7. Docker GPU 已验证

已经执行过类似：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Docker 内可以正常看到：

```text
NVIDIA GeForce RTX 3070
```

所以：

```text
Windows NVIDIA Driver
       ↓
Docker Desktop
       ↓
WSL2
       ↓
CUDA Container
       ↓
RTX 3070
```

这一链路已确认正常。

---

# 8. 非常重要：build_all.sh 只能在 Docker 内运行

曾错误在 WSL host 直接执行：

```bash
bash build_all.sh
```

结果：

```text
cmake: command not found
cd: /kaolin: No such file or directory
```

这是预期的，因为：

```text
/kaolin
cmake
Python / CUDA dependencies
```

都在 FoundationPose Docker image 内。

**以后不要在 WSL host 直接运行 `build_all.sh`。**

正确关系：

```text
Windows
└── WSL Ubuntu
    └── FoundationPose Docker
        └── bash build_all.sh
```

---

# 9. build_all.sh 状态

FoundationPose C++ extension 已经成功构建。

当前官方 build 脚本主要包括：

```text
mycpp
  ↓
CMake
  ↓
make
  ↓
Python C++ extension

Kaolin
  ↓
pip install -e .
```

BundleSDF CUDA ops 当前不是 model-based demo 的必需项。

官方 demo 已成功运行证明：

```text
mycpp
Kaolin
nvdiffrast
PyTorch
CUDA
FoundationPose
```

主环境均已可用。

---

# 10. WSL 网络问题及解决方案

## 10.1 原始问题

WSL 曾经：

```bash
getent ahostsv4 archive.ubuntu.com
```

解析得到：

```text
11.18.0.45
```

`security.ubuntu.com`：

```text
11.18.0.44
```

apt 因此直接连接 fake IP 超时。

同时 WSL 环境里已经存在：

```text
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
http_proxy=http://127.0.0.1:7897
https_proxy=http://127.0.0.1:7897
```

该端口是 Windows 侧代理服务。

执行：

```bash
curl -I https://archive.ubuntu.com --max-time 10
```

可以成功：

```text
HTTP/1.1 200 Connection established

HTTP/1.1 200 OK
```

所以：

**WSL 网络本身正常，问题是 apt 没有正确走代理。**

---

## 10.2 WSL config

Windows 用户目录 `.wslconfig` 已调整方向为：

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

并执行：

```powershell
wsl --shutdown
```

重新打开 WSL。

当前：

```bash
cat /etc/resolv.conf
```

得到：

```text
nameserver 10.255.255.254
```

这是当前正常状态。

不要随便手动改 `/etc/resolv.conf`。

---

## 10.3 APT 永久代理

已经配置 APT 永久代理。

对应文件：

```text
/etc/apt/apt.conf.d/99proxy
```

内容：

```text
Acquire::http::Proxy "http://127.0.0.1:7897";
Acquire::https::Proxy "http://127.0.0.1:7897";
```

现在：

```bash
sudo apt update
```

应该可以正常使用。

---

# 11. X11 / WSLg 可视化

为了运行 FoundationPose GUI，需要：

```text
xhost
WSLg
DISPLAY
X11 socket
```

已安装：

```bash
sudo apt install -y x11-xserver-utils
```

现在：

```bash
which xhost
```

应得到：

```text
/usr/bin/xhost
```

执行官方 Docker 脚本时已经看到：

```text
access control disabled, clients can connect from any host
```

说明：

```text
xhost +
```

成功。

---

# 12. 官方 Docker 启动脚本的路径坑

非常重要。

官方：

```text
docker/run_container.sh
```

内部使用类似：

```bash
DIR=$(pwd)/../
```

因此：

**必须先进入 `docker/` 再执行。**

正确：

```bash
cd ~/projects/FoundationPose-main/docker
bash run_container.sh
```

不要在 repo root：

```bash
bash docker/run_container.sh
```

否则 `DIR` 会变成：

```text
/home/brandon/projects
```

而不是：

```text
/home/brandon/projects/FoundationPose-main
```

---

# 13. 正确进入 FoundationPose Docker

WSL：

```bash
cd ~/projects/FoundationPose-main/docker
bash run_container.sh
```

成功后 prompt 类似：

```text
(my) root@docker-desktop:/home/brandon/projects/FoundationPose-main#
```

检查：

```bash
pwd
```

应为：

```text
/home/brandon/projects/FoundationPose-main
```

检查 GPU：

```bash
nvidia-smi
```

应看到：

```text
RTX 3070
```

---

# 14. FoundationPose 官方 demo 已成功跑通

执行过：

```bash
python run_demo.py --debug 0
```

成功生成：

```text
debug/ob_in_cam/
```

里面每帧一个：

```text
*.txt
```

保存 4×4 Object-in-Camera Pose Matrix。

第一帧曾得到：

```text
 0.6080549955  -0.2525023818   0.7526696324  -0.4481847286
-0.7742000818  -0.3984626234   0.4917741716   0.1187305748
 0.1757365167  -0.8817427158  -0.4377744794   0.8016615510
 0.0000000000   0.0000000000   0.0000000000   1.0000000000
```

最后一帧曾得到：

```text
-0.9004577994   0.4246312678  -0.0941044092  -0.0631549805
 0.3320833445   0.5315117836  -0.7792394161   0.1000391990
-0.2808723748  -0.7329223156  -0.6196205020   0.6308447123
 0.0000000000   0.0000000000   0.0000000000   1.0000000000
```

说明：

```text
FoundationPose register()
FoundationPose track_one()
4×4 Pose Output
```

全部已经成功。

---

# 15. FoundationPose GUI 已成功

后续重新启用了 WSLg / X11。

现在已确认：

**OpenCV 窗口可以从 Docker 正常显示到 Windows 桌面。**

因此当前可以直接运行：

```bash
python run_demo.py --debug 1
```

并看到实时官方 mustard demo：

```text
RGB
+
3D Bounding Box
+
XYZ Pose Axis
```

这一点已经确认成功。

---

# 16. run_demo.py 官方工作流

当前 `run_demo.py` 的核心逻辑：

```python
reader = YcbineoatReader(
    video_dir=args.test_scene_dir,
    shorter_side=None,
    zfar=np.inf
)

for i in range(len(reader.color_files)):
    color = reader.get_color(i)
    depth = reader.get_depth(i)

    if i == 0:
        mask = reader.get_mask(0).astype(bool)

        pose = est.register(
            K=reader.K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=args.est_refine_iter
        )

    else:
        pose = est.track_one(
            rgb=color,
            depth=depth,
            K=reader.K,
            iteration=args.track_refine_iter
        )
```

因此：

### 第一帧

FoundationPose 需要：

```text
RGB
Depth
Mask
K
Mesh
```

执行：

```python
register()
```

### 后续帧

需要：

```text
RGB
Depth
K
```

执行：

```python
track_one()
```

后续帧不需要 mask。

---

# 17. FoundationPose 输入格式

这是后续实时接口必须严格保持的格式。

## RGB

```python
rgb.shape == (H, W, 3)
rgb.dtype == np.uint8
```

FoundationPose 代码按 RGB 使用。

如果来源是 OpenCV BGR：

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

---

## Depth

```python
depth.shape == (H, W)
depth.dtype == np.float32
```

单位：

```text
meters
```

例如：

```text
0.25 = 25 cm
0.80 = 80 cm
1.20 = 1.2 m
```

官方 demo：

```python
depth = cv2.imread(..., -1) / 1e3
```

因为官方 PNG depth 是毫米。

---

## Camera Intrinsics K

```python
K.shape == (3, 3)
```

格式：

```text
fx   0   cx
0    fy  cy
0    0   1
```

例如：

```python
K = np.array([
    [1450.0,    0.0, 960.0],
    [   0.0, 1450.0, 540.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float32)
```

注意：

**K 必须与实际传给 FoundationPose 的 RGB / Depth 分辨率一致。**

如果图像 resize：

```text
1920×1080
↓ 0.5
960×540
```

那么：

```text
fx
fy
cx
cy
```

全部也要乘 0.5。

---

## Mask

第一帧注册需要：

```python
mask.shape == (H, W)
```

binary mask：

```text
0 = background
1 = object
```

目前用户已有 YOLO Bounding Box 检测系统，但不一定有 segmentation。

第一版可以：

```python
mask = np.zeros((H, W), dtype=np.uint8)
mask[y1:y2, x1:x2] = 1
```

直接把 bbox 区域作为粗 mask，用于验证流程。

后续可以升级到：

```text
YOLO-Seg
SAM / SAM2
```

---

# 18. FoundationPose 为什么仍然需要 Depth

FoundationPose 官方 model-based `register()` 会执行类似：

```python
center = self.guess_translation(
    depth=depth,
    mask=mask,
    K=K
)
```

其逻辑大致：

```text
取 mask bbox 中心
+
取 mask 内有效 depth 的中位数
+
inv(K)
↓
计算 Object Translation
```

因此普通 RGB webcam 并不能完整提供官方 FoundationPose 所需输入。

对于本项目：

**最合理的实时传感器仍然是 iPhone RGB + LiDAR Depth。**

---

# 19. Mesh / CAD

Model-based FoundationPose 需要目标物体 Mesh。

官方 mustard 使用：

```text
demo_data/mustard0/mesh/textured_simple.obj
```

之后识别自己的目标，例如 iPhone，则需要类似：

```text
models/
└── iphone16promax/
    └── iphone16promax.obj
```

加载：

```python
mesh = trimesh.load(mesh_file)
```

然后初始化：

```python
est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx,
    debug=1
)
```

---

# 20. 当前 Unity / iPhone 原有项目流程

原来的 AR Object Replacement Pipeline：

```text
AR Camera RGB
↓
iOS CoreML / Vision YOLO
↓
Bounding Box
↓
LiDAR Depth ROI
↓
ARKit Depth + Intrinsics
↓
Back-project Point Cloud
↓
Point Cloud Cleaning
↓
Downsampling
↓
Outlier Removal
↓
Geometric Pose Estimation
↓
SurfaceObject / FreeObject
↓
Unity Replacement
```

旧流程的问题：

- iPhone LiDAR point cloud 较稀疏；
- 手持物体时手部点严重干扰；
- 很难稳定找到物体几何中心；
- 很难稳定判断物体方向；
- 对平滑 / 金属物体效果尤其差。

---

# 21. 新 FoundationPose Pipeline

准备替换为：

```text
iPhone RGB
+
LiDAR Depth
+
Camera Intrinsics
+
Object Mask
+
Object Mesh
        ↓
FoundationPose
        ↓
4×4 Object-in-Camera Pose
        ↓
Coordinate Conversion
        ↓
ARKit / Unity
        ↓
Virtual Object Replacement
```

因此可以考虑删除 / 降级原来的：

```text
ROI Point Cloud
↓
Point Cloud Cleaning
↓
Geometric Center Estimation
↓
Geometric Orientation Estimation
```

LiDAR 的新角色：

```text
提供 Metric Depth
```

而不是：

```text
直接从 Point Cloud 求 Pose
```

---

# 22. 下一阶段：不要修改 run_demo.py

**保留官方 `run_demo.py` 不动。**

它是环境基准。

建议新增：

```text
run_live.py
```

结构：

```text
FoundationPose-main/
├── run_demo.py
├── run_live.py
├── estimater.py
├── datareader.py
└── ...
```

---

# 23. run_live.py 目标结构

第一版核心：

```python
from estimater import *
from datareader import *
import cv2
import numpy as np
import trimesh
import nvdiffrast.torch as dr


mesh_file = "PATH_TO_OBJECT.obj"

mesh = trimesh.load(mesh_file)

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)

bbox = np.stack(
    [-extents / 2, extents / 2],
    axis=0
).reshape(2, 3)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()

est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx,
    debug=1
)

initialized = False

while True:

    rgb, depth, K, mask = get_live_frame()

    if not initialized:

        pose = est.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=5
        )

        initialized = True

    else:

        pose = est.track_one(
            rgb=rgb,
            depth=depth,
            K=K,
            iteration=2
        )

    center_pose = pose @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(
        K,
        img=rgb,
        ob_in_cam=center_pose,
        bbox=bbox
    )

    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=0.1,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True
    )

    cv2.imshow(
        "FoundationPose Live",
        vis[..., ::-1]
    )

    print(pose)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
```

其中最关键的新接口：

```python
get_live_frame()
```

必须返回：

```python
rgb
depth
K
mask
```

---

# 24. 实时开发建议阶段

不要一次把所有网络 / Unity 功能全写完。

建议严格分阶段。

## Stage 1 — 已完成

```text
FoundationPose official demo
+
RTX 3070
+
Docker
+
visualization
```

状态：

```text
DONE
```

---

## Stage 2 — 下一步

建立：

```text
run_live.py
```

要求：

- 保留 FoundationPose 初始化；
- 建立实时 frame loop；
- 能接受 RGB / Depth / K / Mask；
- `register()` 一次；
- 后续 `track_one()`；
- 显示实时 3D box；
- 打印实时 4×4 pose。

先把输入写成抽象接口：

```python
get_live_frame()
```

不要一开始耦合 Unity。

---

## Stage 3

建立一个测试数据源。

可先用：

```text
demo_data/mustard0
```

但按照“实时流”方式逐帧读。

目的：

验证新的 `run_live.py` 与官方结果一致。

也就是说：

```text
run_demo.py
```

和：

```text
run_live.py + RecordedFrameProvider
```

输出应该基本一致。

这是非常重要的 regression baseline。

---

## Stage 4

接入 iPhone / Unity 网络数据。

Unity 发送：

```text
RGB
Depth
Camera Intrinsics
YOLO bbox / mask
timestamp
```

PC 接收。

---

## Stage 5

首帧：

```text
YOLO Detection
↓
Mask
↓
FoundationPose register()
```

之后：

```text
RGB + Depth + K
↓
FoundationPose track_one()
```

如果 tracking lost：

```text
重新检测
↓
重新 mask
↓
register()
```

---

## Stage 6

将 4×4 Pose 发送回 Unity。

---

## Stage 7

解决坐标系转换：

```text
FoundationPose Camera Coordinate
↓
ARKit Camera Coordinate
↓
Unity World Coordinate
```

这一块不要凭感觉做，需要单独验证：

- handedness；
- forward axis；
- up axis；
- matrix multiplication order；
- object-to-camera / camera-to-object；
- row-major / column-major；
- meters；
- Unity quaternion conversion。

---

# 25. 网络数据格式建议

后续 iPhone / Unity → PC，可以先使用 TCP。

Header 例如：

```json
{
  "width": 1280,
  "height": 720,
  "fx": 921.3,
  "fy": 921.7,
  "cx": 640.1,
  "cy": 359.8,
  "bbox": [400, 180, 820, 650],
  "timestamp": 12345678
}
```

RGB：

```text
JPEG
```

避免发送原始：

```text
1280 × 720 × 3
≈ 2.76 MB / frame
```

Depth：

Unity/iPhone 内：

```text
meters float
↓ ×1000
millimeters
↓
uint16
↓
PNG
```

PC：

```python
depth = depth_png.astype(np.float32) / 1000.0
```

这样与 FoundationPose 官方 demo 格式一致。

---

# 26. FPS 建议

不要求 FoundationPose 本身跑 Unity 的 60 FPS。

推荐：

```text
iPhone Camera
30 FPS
    ↓
PC receiver
    ↓
只保留 latest frame
    ↓
FoundationPose
5–15 FPS
    ↓
Pose
    ↓
Unity Interpolation
60 FPS rendering
```

避免 frame queue 无限积压。

应该采用：

```text
latest-frame-wins
```

而不是：

```text
processing every captured frame
```

---

# 27. RTX 3070 注意事项

GPU：

```text
RTX 3070
8 GB VRAM
```

官方 mustard0 已成功。

后续自定义场景如果出现 CUDA OOM：

优先尝试：

1. 降低 RGB / Depth 输入分辨率；
2. 同步缩放 K；
3. 简化 Mesh；
4. 减少 debug；
5. 关闭其他 GPU 程序；
6. 减少 refine iteration。

不要第一反应去升级 CUDA / PyTorch。

---

# 28. FoundationPose API 重点

## Initialization

```python
FoundationPose(
    model_pts,
    model_normals,
    mesh,
    scorer,
    refiner,
    glctx
)
```

---

## Initial Registration

```python
pose = est.register(
    K=K,
    rgb=rgb,
    depth=depth,
    ob_mask=mask,
    iteration=5
)
```

输出：

```text
4×4 numpy pose matrix
```

---

## Tracking

```python
pose = est.track_one(
    rgb=rgb,
    depth=depth,
    K=K,
    iteration=2
)
```

前提：

```text
register() 已成功
```

内部使用上一次：

```text
pose_last
```

作为 tracking initialization。

---

# 29. Pose Matrix 语义

官方 debug 输出目录：

```text
debug/ob_in_cam
```

因此 FoundationPose 输出应按：

```text
Object in Camera
```

理解。

即：

```text
Object Coordinate
      ↓
Pose Matrix
      ↓
Camera Coordinate
```

后续 Unity integration 时必须认真确认：

```text
T_camera_object
```

还是反矩阵：

```text
T_object_camera
```

的命名约定。

不要直接把矩阵 16 个数字塞进 Unity Transform。

---

# 30. Debug / 可视化

FoundationPose：

```bash
python run_demo.py --debug 1
```

会显示：

```text
3D bounding box
XYZ axis
```

`debug >= 2` 还可以输出：

```text
debug/track_vis/
```

以及更多中间结果。

当开发 `run_live.py` 时，建议始终保留：

```python
draw_posed_3d_box()
draw_xyz_axis()
```

作为 Pose 是否正确的视觉检查。

---

# 31. 不要做的事情

Codex 后续请避免以下操作：

## 不要

```text
重新创建 FoundationPose CUDA 环境
```

当前已经运行成功。

---

## 不要

```text
升级 Docker 中 CUDA
升级 PyTorch
升级 Kaolin
```

除非出现明确不可解决的问题。

当前 RTX 3070 已运行官方 demo。

---

## 不要

在 WSL host 执行：

```bash
bash build_all.sh
```

---

## 不要

第一步就把整个 FoundationPose 塞进 Unity。

FoundationPose 保持：

```text
Python + GPU service
```

Unity 保持：

```text
AR front end / sensor / renderer
```

---

## 不要

用普通 webcam 的伪 depth 直接评价 FoundationPose 平移性能。

本项目真正目标输入是：

```text
iPhone LiDAR Metric Depth
```

---

## 不要

一开始同时修改：

```text
FoundationPose
Unity
Networking
YOLO
Coordinate Conversion
```

必须分阶段建立 baseline。

---

# 32. Codex 下一步明确任务

请从当前 repo 开始，不重建环境。

工作目录：

```text
/home/brandon/projects/FoundationPose-main
```

### Task 1

阅读：

```text
run_demo.py
estimater.py
datareader.py
Utils.py
```

理解官方数据流。

---

### Task 2

创建：

```text
run_live.py
```

但不要修改：

```text
run_demo.py
```

---

### Task 3

先创建一个统一 Frame Provider Interface，例如：

```python
class FramePacket:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    mask: Optional[np.ndarray]
    timestamp: float
```

以及：

```python
class FrameProvider:
    def get_frame(self) -> FramePacket:
        ...
```

---

### Task 4

第一版实现：

```text
RecordedFoundationPoseProvider
```

仍读取：

```text
demo_data/mustard0
```

但通过新的实时 frame API 喂给 `run_live.py`。

目的：

**先证明新的 live architecture 没有改变 FoundationPose 结果。**

---

### Task 5

实现状态机：

```text
UNREGISTERED
↓
register()
↓
TRACKING
↓
track_one()
↓
TRACKING

tracking failed
↓
UNREGISTERED
```

第一版 tracking failure detection 可以先留简单接口。

---

### Task 6

保留 GUI：

```text
FoundationPose Live
```

显示：

- RGB；
- 3D Bounding Box；
- XYZ Axis；
- 当前状态；
- FPS；
- Frame ID。

---

### Task 7

每帧打印 / 可选保存：

```text
4×4 pose
timestamp
processing time
FPS
```

---

# 33. 后续 iPhone Integration 接口预留

建议未来实现：

```python
class NetworkFrameProvider(FrameProvider):
    ...
```

负责接收：

```text
RGB JPEG
Depth uint16 PNG
K
bbox / mask
timestamp
```

这样 FoundationPose 主逻辑完全不用知道数据来自：

```text
mustard0
iPhone
RealSense
recorded sequence
```

---

# 34. 当前项目最重要原则

请始终保持三层解耦：

```text
Sensor / Input
      ↓
Pose Estimation
      ↓
AR Rendering
```

即：

```text
iPhone / Unity
      ↓
Frame Packet
      ↓
FoundationPose
      ↓
Pose Packet
      ↓
Unity
```

不要让 FoundationPose 代码直接依赖 Unity。

不要让 Unity 代码直接依赖 FoundationPose 内部 Python implementation。

---

# 35. 当前完成状态总结

```text
[✓] RTX 3070 Docker GPU passthrough
[✓] FoundationPose official Docker image
[✓] weights
[✓] mustard0 demo data
[✓] mycpp build
[✓] Kaolin environment
[✓] run_demo.py --debug 0
[✓] 4×4 pose output
[✓] WSL Internet
[✓] permanent APT proxy
[✓] WSLg / X11
[✓] OpenCV GUI
[✓] run_demo.py visualization

[ ] run_live.py
[ ] realtime FrameProvider architecture
[ ] recorded-data live regression test
[ ] iPhone RGB streaming
[ ] iPhone LiDAR Depth streaming
[ ] Intrinsics transport
[ ] automatic object mask
[ ] real-time FoundationPose tracking
[ ] pose network return
[ ] FoundationPose → ARKit coordinate conversion
[ ] Unity virtual object replacement
```

---

# 36. 建议 Codex 开始时先做的检查

进入 WSL：

```bash
cd ~/projects/FoundationPose-main/docker
bash run_container.sh
```

Docker 中：

```bash
cd /home/brandon/projects/FoundationPose-main
```

验证：

```bash
python run_demo.py --debug 1
```

如果官方 mustard demo 正常显示：

**不要再调整环境。**

立即开始：

```text
run_live.py
FrameProvider
RecordedFoundationPoseProvider
```

---

# 37. 用户偏好 / 开发要求

本项目当前阶段的重点是：

```text
做出可以工作的系统 / Demo
```

而不是为了理论完整性重构 FoundationPose。

开发方式：

- 每一步尽量可独立验证；
- 不一次改太多模块；
- 保留官方 demo 作为 baseline；
- 每次改动都能通过视觉或 numerical output 验证；
- 优先让 pipeline 跑通，再逐步提高精度和实时性；
- 代码注释需明确矩阵坐标系与单位；
- 不要隐藏异常或 silently fallback；
- 网络输入必须校验 RGB / depth / K 尺寸一致性。

---

# 38. 最终目标

最终目标不是单纯运行 FoundationPose，而是：

```text
iPhone 拿着真实物体
        ↓
实时 RGB + LiDAR
        ↓
PC FoundationPose
        ↓
稳定输出真实物体 6DoF Pose
        ↓
Unity 得到姿态
        ↓
虚拟模型与真实物体准确重合
        ↓
真实物体被虚拟模型视觉替换
```

FoundationPose 是整个系统中的：

```text
6DoF Pose Estimation Backend
```

而不是完整 AR Application。

---

## Codex 接手时的第一条建议

**不要再做环境配置。**

从：

```text
run_live.py
```

开始。

第一目标：

> 使用现有 `demo_data/mustard0`，通过新的 `FrameProvider` 实时循环架构复现 `run_demo.py` 的姿态结果和可视化。

这个目标通过之后，再开始接 iPhone。
