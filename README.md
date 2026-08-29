# FoundationPose（本仓库）

NVIDIA [FoundationPose](https://nvlabs.github.io/FoundationPose/)（CVPR 2024）的本地工程副本，用来做 **RGB-D 6DoF 位姿估计与跟踪**，并接到 **PC 实时推理 + iPhone/Unity TCP 推流**。

上游论文、BOP 排行榜、训练数据说明见同目录的 [`readme.md`](readme.md)。本文件只写：**这个仓库是什么、怎么在本机（尤其是 RTX 5070 Ti + Docker）跑起来。**

许可证为 NVIDIA Source Code License，**仅限非商业研究/评估**。

---

## 这个仓库实际做什么

给定：

- RGB
- 对齐的深度（米）
- 相机内参 `K`
- **第一帧物体 mask**
- 物体 mesh（CAD，或 Polycam/BundleSDF 重建）

输出：

- `ob_in_cam`：物体在 **OpenCV 相机坐标系** 下的 4×4 矩阵，平移单位 **米**（x 右、y 下、z 前）

流程：

```text
第一帧  REGISTER（旋转采样 + 深度质心 + refiner + scorer）
随后    TRACK（上一帧 pose 再 refine）
可选    经 TCP 把 Unity/iPhone 的 RGB-D 送到本机 GPU
```

当前状态：

| 能力 | 状态 |
|---|---|
| 离线 demo / recorded 跟踪 | 可用 |
| PC 收 TCP 帧并估计 pose | 可用 |
| Unity/iPhone 发帧 | 有脚本（需拷进 Unity 工程） |
| Unity 接收 pose / 坐标转换 / AR 物体替换 | **未实现** |
| 相对真值的 Translation/Rotation Error | 本仓库 mustard0 **没有** `annotated_poses/` |

---

## 目录速查

| 路径 | 作用 |
|---|---|
| `run_demo.py` | 官方 demo：mustard0 上 register→track |
| `run_live.py` | 实时入口：`--provider recorded` 或 `network` |
| `send_recorded_frames.py` | 把 recorded 序列以 FPFRAME 发给 PC |
| `measure_tcp_latency.py` | 本机 TCP 时延测量（会读 `FPRESULT`） |
| `check_env.py` | 检查 CUDA、扩展、权重、demo |
| `docker/` | 3070 原镜像 + **5070 Ti（CUDA 12.8）** |
| `weights/` | scorer / refiner 权重（需下载） |
| `demo_data/mustard0/` | 官方 demo 序列 |
| `unity_sender/` | Unity 发送端 C# 参考脚本 |
| `docs/POLYCAM_MODEL_FREE.md` | Polycam → model-free 重建 |

---

## 运行前准备

### 硬件与软件

- NVIDIA GPU。本仓库日常环境是 **RTX 5070 Ti（Blackwell, sm_120）**，必须用 `foundationpose:5070ti` 镜像，**不要**用原版 CUDA 11 的 `foundationpose` 镜像。
- Linux 或 **WSL2** + Docker + NVIDIA Container Toolkit（`docker run --gpus all` 可用）。
- 磁盘：权重 + mustard0 demo 大约数 GB。

### 1. 下载权重

从 [Google Drive（官方 weights）](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) 下载后放到 `weights/`：

```text
weights/2024-01-11-20-02-45/model_best.pth   # scorer
weights/2024-01-11-20-02-45/config.yml
weights/2023-10-28-18-33-37/model_best.pth   # refiner
weights/2023-10-28-18-33-37/config.yml
```

### 2. 下载 demo 数据

从 [Google Drive（demo_data）](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) 解压到 `demo_data/`。至少需要：

```text
demo_data/mustard0/mesh/textured_simple.obj
demo_data/mustard0/rgb/
demo_data/mustard0/depth/
demo_data/mustard0/masks/     # 至少第一帧
demo_data/mustard0/cam_K.txt
```

### 3. 构建 5070 Ti 镜像（只需一次）

在**宿主机**项目根目录：

```bash
cd /home/brandon/projects/FoundationPose-main
cd docker
bash build_5070ti.sh
```

镜像名：`foundationpose:5070ti`。第一次构建会很久。

---

## 推荐路径：5070 Ti Docker

后面所有 Python 命令都在**容器内**执行。宿主机没有配齐 CUDA 栈时，不要直接 `python run_demo.py`。

### 启动容器

```bash
cd /home/brandon/projects/FoundationPose-main
bash docker/run_container_5070ti.sh
```

注意：

- 这个脚本会 `docker rm -f foundationpose-5070ti` 再新建容器，**会关掉已有的同名容器**。
- 需要交互终端（`-it`）。映射了 **5000** 端口，给 Unity/手机 TCP 用。
- 项目目录以 bind mount 挂进容器，改代码立刻生效。

进入后提示符类似 `root@...:/home/brandon/projects/FoundationPose-main#`。

容器已在跑时，另开一个终端进入（不要再跑 `run_container_5070ti.sh`）：

```bash
docker exec -it foundationpose-5070ti bash
cd /home/brandon/projects/FoundationPose-main
```

### 第一次：编译本地扩展

只在该镜像里做一次：

```bash
bash docker/build_extensions_5070ti.sh
```

会编译 `mycpp/build_5070ti_docker` 和 BundleSDF 的 `mycuda`。5070 Ti 上 native `cluster_poses` 可能 segfault，demo 脚本默认：

```text
FOUNDATIONPOSE_USE_PY_CLUSTER=1
```

### 检查环境

```bash
export FOUNDATIONPOSE_MYCPP_BUILD_DIR=/home/brandon/projects/FoundationPose-main/mycpp/build_5070ti_docker
python check_env.py --demo-data
```

期望看到：`CUDA available: True`、两套 weights `ok`、`run_demo demo_data: ok`。有 `ERROR` 再往下跑没有意义。

---

## 怎么跑

以下均在 **容器内、项目根目录**。无 GUI 时用 `--debug 0` 或 `DEBUG=0`。

### A. 官方 demo（最快验证 GPU）

有窗口：

```bash
bash docker/run_demo_5070ti.sh
```

无窗口 / Docker 无 DISPLAY：

```bash
DEBUG=0 bash docker/run_demo_5070ti.sh
```

保存可视化（overlay 写到目录，不一定弹窗）：

```bash
DEBUG=2 bash docker/run_demo_5070ti.sh
```

等价手写：

```bash
export FOUNDATIONPOSE_MYCPP_BUILD_DIR=/home/brandon/projects/FoundationPose-main/mycpp/build_5070ti_docker
export FOUNDATIONPOSE_USE_PY_CLUSTER=1
python run_demo.py --debug 0 --debug_dir debug_5070ti_docker
```

第一帧 REGISTER 会明显慢（约 1–3 秒），后面 TRACK。第一次还可能触发 CUDA/nvdiffrast 编译，更慢。结果在 `--debug_dir`。

### B. Recorded live（本仓库主路径，推荐）

用磁盘上的 mustard0 当「相机」，走 `run_live.py` 状态机（和 TCP 同一套代码）：

```bash
mkdir -p results_today/live_logged_full
python -u run_live.py \
  --provider recorded \
  --debug 0 \
  --test_scene_dir demo_data/mustard0 \
  --mesh_file demo_data/mustard0/mesh/textured_simple.obj \
  --debug_dir results_today/live_logged_full \
  > results_today/live_logged_full/run.log 2>&1
```

成功标志：

- 日志末尾：`frame provider reached end of stream`
- `results_today/live_logged_full/ob_in_cam/` 里 **737** 个 `*.txt`（4×4 pose）
- log 里第一帧 `operation: REGISTER`，其余 `TRACK`

常用参数：

| 参数 | 含义 | 建议 |
|---|---|---|
| `--debug 0` | 不弹 OpenCV 窗 | Docker/无头必开 |
| `--debug 1` | 弹窗，按 `q` 退出 | 需要 X11 |
| `--save_track_vis` | 把 overlay 存到 `debug_dir/track_vis/` | 体积大 |
| `--est_refine_iter` | REGISTER refine 次数 | 默认 5 |
| `--track_refine_iter` | TRACK refine 次数 | 默认 2 |

看 REGISTER / TRACK 耗时：

```bash
grep -a -n "operation: REGISTER" -A3 results_today/live_logged_full/run.log | head
grep -a -c "operation: TRACK" results_today/live_logged_full/run.log
```

### C. 本机 TCP（PC 听端口 + 发 recorded 帧）

两个进程，**先启动 server**。

终端 1（容器内）：

```bash
python -u run_live.py \
  --provider network \
  --debug 0 \
  --host 0.0.0.0 \
  --port 5000 \
  --mesh_file demo_data/mustard0/mesh/textured_simple.obj \
  --debug_dir results_today/live_tcp \
  > results_today/live_tcp/run.log 2>&1
```

等到 log 出现：

```text
waiting for frame sender on 0.0.0.0:5000
```

终端 2（再 `docker exec` 进同一个容器）：

**只发帧、不等回包：**

```bash
python send_recorded_frames.py \
  --host 127.0.0.1 \
  --port 5000 \
  --test_scene_dir demo_data/mustard0 \
  --fps 0
```

`--fps 0` 表示发完一帧立刻发下一帧。`--fps 30` 按 30 Hz 节流。

**测发送→收到 `FPRESULT` 的时延（推荐）：**

Ping-pong（一帧一结果，不丢帧）：

```bash
python measure_tcp_latency.py \
  --host 127.0.0.1 --port 5000 \
  --mode pingpong --fps 0 --rgb_codec png \
  --csv_path results_today/tcp_pingpong/latency.csv
```

此时 PC 侧建议关掉「只留最新帧」，否则和 ping-pong 语义不一致：

```bash
python -u run_live.py --provider network --debug 0 \
  --host 0.0.0.0 --port 5000 --no_latest_frame_only \
  --debug_dir results_today/tcp_pingpong
```

模拟直播（30 Hz JPEG，默认 `latest_frame_only` 会丢中间帧）：

```bash
# 终端 1
python -u run_live.py --provider network --debug 0 \
  --host 0.0.0.0 --port 5006 --latest_frame_only \
  --debug_dir results_today/tcp_live30

# 终端 2（等 waiting for frame sender 后再跑）
python measure_tcp_latency.py \
  --host 127.0.0.1 --port 5006 \
  --mode paced --fps 30 --rgb_codec jpeg --jpeg_quality 95 \
  --csv_path results_today/tcp_live30/latency.csv
```

默认 `--send_results` 为开。发送端如果不读回包，TCP 窗口可能被 `FPRESULT` 撑满导致卡住；`measure_tcp_latency.py` 会读回包。`send_recorded_frames.py` 不读回包，长时间快发可能堵死。

### D. iPhone / Unity 推流

1. PC 仍用上面的 `run_live.py --provider network --host 0.0.0.0 --port 5000`。
2. 防火墙放行 **5000**。Docker 脚本已 `-p 5000:5000`。
3. 把 `unity_sender/Assets/Scripts/FoundationPose/` 拷进 Unity 工程，接线见 [`unity_sender/README.md`](unity_sender/README.md)。
4. `FoundationPoseTcpSender.host` 填 **Windows 局域网 IP**，不要填 `127.0.0.1`（手机连不到 WSL 的 localhost）。
5. 建议 Inspector：`rgbCodec = JPEG`，`targetFps` 先用默认 **5**。
6. **第一帧必须带 mask**（YOLO bbox 矩形或手标），否则 PC 无法 REGISTER。
7. Unity **目前不接收** `FPRESULT`，手机上看不到估计 pose；PC 可用 `--debug 1` 看 overlay，或看 `ob_in_cam/*.txt`。

协议检查（不跑估计器）：

```bash
python inspect_fpframe_stream.py --host 0.0.0.0 --port 5000 --num_frames 3 --expect_synthetic
```

### E. Polycam model-free（可选、很慢）

见 [`docs/POLYCAM_MODEL_FREE.md`](docs/POLYCAM_MODEL_FREE.md)。在容器内：

```bash
python run_polycam_model_free.py \
  --input "Polycam file/2026_8_16" \
  --output demo_data/polycam_model_free \
  --num_views 24
```

加 `--run` 会跑 BundleSDF/NeRF，耗时长、吃 GPU。

### F. 官方数据集（可选）

需要自己下载 LINEMOD / YCB-Video，路径改成你的盘：

```bash
python run_linemod.py --linemod_dir /path/to/LINEMOD --use_reconstructed_mesh 0
python run_ycb_video.py --ycbv_dir /path/to/YCB_Video --use_reconstructed_mesh 0
```

---

## 输出在哪

| 产物 | 位置 |
|---|---|
| 每帧 pose | `<debug_dir>/ob_in_cam/<frame_id>.txt` |
| overlay | `--debug>=1` 或 `--save_track_vis` 时的 `track_vis/` |
| 运行日志 | 你重定向的 `run.log`（务必 `python -u` + 重定向，终端滚动会丢 REGISTER） |
| TCP 时延 CSV | `measure_tcp_latency.py --csv_path` |

Pose 文件是 4×4 文本，可用 `numpy.loadtxt`。

---

## 常见问题

**`docker: command not found`**  
你已经在容器里了。不要在容器内再 `docker exec`。从 **WSL/Windows 宿主机** 执行 `docker`。

**容器一启动就没了 / `run_container_5070ti.sh` 把正在跑的实验杀掉了**  
脚本每次都会删同名容器。容器还在时只用 `docker exec -it foundationpose-5070ti bash`。

**`CUDA available: False` 或 Blackwell 报错**  
确认用的是 `foundationpose:5070ti`，并且 `docker run --gpus all`。原版 `wenbowen123/foundationpose` 不支持 5070 Ti。

**第一帧报 `first registration frame requires a mask`**  
recorded：`demo_data/mustard0/masks/` 缺第一帧。网络：Unity 第一包没带 mask。

**弹窗失败 / 卡住**  
`--debug 0`。不要在无 DISPLAY 的 Docker 里用默认 `--debug 1`。

**`latest_frame_only` 丢帧**  
`network` 默认只处理最新帧。要逐帧不丢：`--no_latest_frame_only`，并让发送端不要快过处理速度（或 ping-pong）。

**WSL 里 `docker.sock` 连不上**  
Docker Desktop 打开，WSL 集成打开。本 README 里的命令都假设 Docker 已可用。

**宿主机 conda**  
若不用 Docker，可看 `readme.md` 的 conda 章节，或 `run_5070ti.sh`（依赖 `.conda-envs/foundationpose5070`）。5070 Ti 优先 Docker。

---

## 和论文数字相关的说明

本地实测摘要在 [`CODEBASE_RESEARCH_AUDIT.md`](CODEBASE_RESEARCH_AUDIT.md) **§21**。不要把交接文档里的「20–35 FPS」或 3070 数字当成本机 Results。

- **processing FPS** 和 **墙钟 FPS** 必须分开报。
- 无 `annotated_poses/` 就不能报 Translation/Rotation Error。
- 手机 WiFi 时延随网络变化，本仓库没有强制真机测量。

---

## 引用

若使用本方法，请引用 FoundationPose CVPR 2024（以及 model-free 时的 BundleSDF CVPR 2023）。BibTeX 见 [`readme.md`](readme.md)。
