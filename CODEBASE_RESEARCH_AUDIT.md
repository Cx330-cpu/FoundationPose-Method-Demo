# CODEBASE_RESEARCH_AUDIT

**审计范围：** `/home/brandon/projects/FoundationPose-main`  
**审计日期：** 2026-08-21  
**方法：** 上午只读扫描代码与文档；下午在 RTX 5070 Ti Docker 上补测 mustard0 recorded / localhost TCP，并把可填指标写入 **§21**。未跑 iPhone/Unity 真机（网络环境强绑定，改用预估）。  
**证据优先级（有矛盾时）：** 本机落盘 log/CSV > 实际可执行代码 > 当前配置/协议实现 > 本仓库局部文档 > 交接文档 > 上游 NVIDIA `readme.md` 中的历史声明。预估值单独标注，不得与测得值混写。

---

## 1. Repository Identity

| 字段 | 结论 | 状态 | 证据 |
|---|---|---|---|
| repo name | `FoundationPose-main`（上游正式名 FoundationPose） | implemented | `readme.md` L1；工作区路径 |
| repo purpose | 统一 novel-object **6DoF pose estimation + tracking**；支持 **model-based（CAD/mesh）** 与 **model-free（少量参考视图 + Neural Object Field / BundleSDF）**。本 fork 另加：PC live 推理、FPFRAME TCP、Unity/iPhone RGB-D sender、Polycam→BundleSDF 转换、RTX 5070 Ti Docker。 | implemented（核心）+ partially implemented（AR 闭环） | `readme.md` L1–L7；`run_demo.py` L50–63；`run_live.py` L471–621；`docs/POLYCAM_MODEL_FREE.md` L1–5 |
| owner / author | 上游：Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield（NVIDIA / CVPR 2024）。本工作区另有本地工程交接（PC live / Unity / Polycam）。无 git remote。 | implemented（版权信息） | `readme.md` L5, L37–L45, L250–L255；`LICENSE` L1–L2, L53–L57 |
| route / method name | **FoundationPose 6DoF RGB-D pose estimation & tracking**（model-based 为主路径；model-free 为可选重建路径；live AR streaming 为系统集成路径） | implemented / partially implemented | `estimater.py` `FoundationPose` L51–L75, `register` L195–L276, `track_one` L286–L304 |
| route objective | 给定 RGB-D、相机内参、首帧 object mask、以及 CAD mesh（或重建 mesh），估计 **object-in-camera 4×4 pose**，随后逐帧 tracking；可选把帧从 iPhone/Unity 经 TCP 送到 PC GPU。 | implemented（PC 估计） / partially implemented（Unity AR overlay / 回传） | `run_live.py` L518–L567；`unity_sender/README.md` L5–L14, L88–L91 |
| implementation status | **核心算法：implemented。** Live 网络输入：implemented。Unity sender 脚本：implemented（非完整 Unity 工程）。PC→Unity pose 接收/坐标转换/AR object replacement：**partially implemented / planned**。论文级 accuracy evaluation：**partially implemented**（有函数，无跑通结果）。 | 见右 | 见下“矛盾” |
| 与 AR/VR / pose / tracking / replacement / perception 的关系 | **Pose estimation + tracking + perception pipeline：implemented。** AR 采集（AR Foundation RGB + LiDAR depth）：Unity 脚本 implemented，真机编译 **UNKNOWN**。Object replacement / AR overlay：**PC OpenCV overlay implemented；Unity 侧 virtual object overlay N/A。** VR headset runtime：**N/A。** | mixed | `run_live.py` L230–L237, L571–L581；`FoundationPoseFrameStreamer.cs` L12–L27, L132–L150；`unity_sender/README.md` L88–L91 |

**支撑证据（身份）：**

- 上游论文与任务：`readme.md` L1–L7, L19–L32（明确写 AR applications demo）。
- 本仓库 live 目标：`FoundationPose_Handoff_2026-08-14.md` L12；`FOUNDATIONPOSE_TO_UNITY_HANDOFF.md` L10–L42。
- License：NVIDIA Source Code License，**non-commercial / research or evaluation only**（`LICENSE` L53–L57）。投稿引用必须 cite FoundationPose CVPR 2024 与（若用 model-free）BundleSDF CVPR 2023（`readme.md` L37–L56）。

**文档 vs 代码矛盾（本节）：**

1. `FoundationPose_Handoff_2026-08-14.md` L47–L54 写“本阶段暂不做 pose 回传 / tracking-loss recovery”。**更信任代码：** `run_live.py` L206–L207, L540–L563 已实现 `FPRESULT` 发送与 `mask_request`；`mask_recovery` 默认关闭但已实现（L528–L539, L602）。
2. `FOUNDATIONPOSE_TO_UNITY_HANDOFF.md` L464–L478 仍写 “PC -> Unity Pose Return 尚未实现”。**PC 发送已实现，Unity 接收未实现。** 更信任：`run_live.py` L555–L563 vs `unity_sender/README.md` L30, L88–L91。
3. 交接文档假设 `demo_data/mustard0` 与 `weights/` 存在。**上午审计时缺失；同日已补齐**，并完成 `run_live.py` recorded 全序列（737 帧）。更信任当前文件系统与 `results_today/`。

---

## 2. High-level Pipeline

本仓库实际有 **三条可运行路径**，不是单一线性 pipeline。

### Route A — Official model-based demo / recorded tracking（implemented）

```text
RGB-D scene dir + mesh OBJ + cam_K.txt + first-frame mask
  -> YcbineoatReader
  -> FoundationPose.register (frame 0)
  -> FoundationPose.track_one (later frames)
  -> save ob_in_cam/*.txt + optional OpenCV overlay
```

| Stage | 关键文件 / 函数 | Status | Input | Output |
|---|---|---|---|---|
| Input IO | `datareader.py` `YcbineoatReader` L57–L127 | implemented | `rgb/*.png`, `depth/*.png` (mm→m), `cam_K.txt`, `masks/` | RGB uint8, depth meters, K 3×3, mask bool |
| Mesh load | `run_demo.py` L29–L41 | implemented | `.obj` | trimesh + `FoundationPose` |
| Depth preprocess | `estimater.py` `register` L209–L210, `track_one` L292–L294 | implemented | depth | eroded + bilateral-filtered depth |
| Pose hypotheses | `estimater.py` `make_rotation_grid` L139–L160, `generate_random_pose_hypo` L163–L170, `guess_translation` L173–L192 | implemented | mask+depth+K | rotation grid + translation guess |
| Refine | `PoseRefinePredictor.predict` `learning/training/predict_pose_refine.py` L149–L237 | implemented | rendered vs observed crops | refined `ob_in_cam` (centered mesh) |
| Score / select | `ScorePredictor.predict` `learning/training/predict_score.py` L160–L226 | implemented | pose hypotheses | best pose |
| Tracking | `estimater.py` `track_one` L286–L304 | implemented | previous pose + RGB-D | updated 4×4 |
| Visualize / log | `run_demo.py` L65–L78 | implemented | pose | `debug/ob_in_cam/*.txt`, optional `track_vis/*.png` |

证据：`run_demo.py` L46–L78。

### Route B — Live PC inference + FPFRAME TCP（implemented on PC；Unity sender implemented as scripts）

```text
Unity/iPhone or send_recorded_frames.py
  -> FPFRAME v1 TCP
  -> NetworkFrameProvider / RecordedFoundationPoseProvider
  -> REGISTER / TRACK / optional RE-REGISTER
  -> print + save pose txt + optional FPRESULT JSON back on same TCP socket
  -> PC OpenCV overlay (not Unity render)
```

| Stage | 关键文件 / 函数 | Status | Input | Output |
|---|---|---|---|---|
| Network decode | `network_frame_protocol.py` `receive_frame` L166–L211 | implemented | TCP bytes | rgb, depth (m), K, mask, header |
| Frame provider | `run_live.py` `NetworkFrameProvider` L119–L212, `RecordedFoundationPoseProvider` L49–L116 | implemented | socket or disk | `FramePacket` |
| State machine | `run_live.py` `run_live` L501–L548 | implemented | packet | REGISTER / TRACK / RE-REGISTER |
| Lost-track heuristic | `tracking_lost_reason` L435–L455 | implemented | pose deltas, projected bbox | reason string or None |
| Mask request | `request_mask` L198–L204 | implemented | reason | `FPCONTROL` JSON |
| Result send | `make_pose_result` L382–L432, `send_result` L206–L207 | implemented (PC) / **not received in Unity** | pose | `FPRESULT` JSON |
| Overlay | `draw_live_overlay` L230–L237 | implemented | pose | OpenCV window / PNG |

证据：`run_live.py` L471–L586；`unity_sender/README.md` L5–L14。

### Route C — Polycam model-free reconstruction（converter implemented；NeRF training optional / not evidenced as completed here）

```text
Polycam LiDAR Raw (RGB, depth, confidence, camera JSON)
  -> YOLO-seg or projected mask
  -> OpenGL/ARKit -> OpenCV cam_in_ob
  -> BundleSDF / Neural Object Field
  -> model/model.obj
  -> (intended) feed back into Route A/B as mesh
```

| Stage | 关键文件 / 函数 | Status | Input | Output |
|---|---|---|---|---|
| Convert | `tools/polycam_to_foundationpose.py` `convert` ~L439–L677 | implemented | `Polycam file/2026_8_16` | `ob_0000001/{rgb,depth_enhanced,mask,cam_in_ob,K.txt}` |
| Quality / view select | `tools/polycam_quality_pipeline.py` | implemented | same | diagnostics JSON/PLY/overlays |
| NeRF reconstruct | `bundlesdf/run_nerf_custom.py` L47–L101；`bundlesdf/run_nerf.py` `run_neural_object_field` L18–L46 | implemented in code | ref views | `model/model.obj` |
| Use reconstructed mesh | `datareader.py` `get_reconstructed_mesh` L508–L510；`run_ycb_video.py` L99–L102 | implemented | `model/model.obj` | mesh for FoundationPose |
| Completed reconstruction in this checkout | — | **UNKNOWN / missing** | — | 无 `demo_data/polycam_*` 输出目录 |

证据：`docs/POLYCAM_MODEL_FREE.md` L21–L62；`run_polycam_model_free.py` L43–L51。

**N/A stages（本 repo 没有）：** SLAM、AprilTag/ArUco、OpenCV PnP 作为主估计器、Gaussian Splatting、Unity AR object replacement、headset compositor。

---

## 3. Inputs and Outputs

### Inputs

| Item | Status | Format / unit | Example location | Code evidence |
|---|---|---|---|---|
| RGB image / video | implemented | PNG/JPEG；uint8 HxWx3 RGB | `demo_data/mustard0/rgb/` **缺失于本 checkout**；Polycam `.../images/*.jpg`, `.../corrected_images/*.jpg` **存在** | `datareader.py` L107–L110；`network_frame_protocol.py` L67–L82, L179–L180 |
| Depth map / LiDAR / RGB-D | implemented | uint16 PNG **millimetres** on disk/wire；runtime **meters** | Polycam `keyframes/depth/*.png`；live: AR raw environment depth | `datareader.py` L122–L126；`network_frame_protocol.py` L84–L85, L181–L184；`FoundationPoseFrameStreamer.cs` L327–L351 |
| Mask / segmentation | implemented / partial | uint8 PNG；bool at runtime。首帧必须。Unity 默认为 **YOLO bbox 矩形 mask**，不是 instance seg | demo `masks/` 缺失；Polycam converter 可写 `mask/` | `run_live.py` L91–L97, L518–L520；`YoloBBoxToMaskAdapter.cs` L102–L114；`tools/polycam_to_foundationpose.py` L518–L566 |
| Camera intrinsics | implemented | 3×3 K；fx,fy,cx,cy | `cam_K.txt` / `K.txt` / FPFRAME header / Polycam JSON `fx,fy,cx,cy` | `datareader.py` L63；`network_frame_protocol.py` L105–L108, L199–L203；Polycam JSON 样本 `.../cameras/25540516047.json` L5–L22 |
| Camera extrinsics | implemented **only on model-free / Polycam / BundleSDF path** | 4×4 `cam_in_ob` | converter 写入 `cam_in_ob/*.txt` | `tools/polycam_to_foundationpose.py` L49–L59, L511, L598；`bundlesdf/run_nerf.py` L23 |
| CAD model / mesh | implemented（model-based 必需） | OBJ | 默认 `demo_data/mustard0/mesh/textured_simple.obj` **本 checkout 缺失** | `run_demo.py` L18, L29；`run_live.py` L475, L592 |
| Object size / scale | derived, not a separate input | mesh vertices；BOP mesh `*1e-3` m | BOP `models_info.json` diameter/1e3 | `estimater.py` L78–L88；`datareader.py` L291–L295, L494–L496 |
| Calibration files | partial | `cam_K.txt` / `scene_camera.json`；无独立 stereo calib 流程 | demo / BOP | `datareader.py` L63, L166–L170 |
| Network stream | implemented | TCP FPFRAME v1 | port 5000 default | `run_live.py` L119–L135, L600–L601 |
| Unity scene data | N/A as Unity project | 仅 sender 脚本包，无 `ProjectSettings/` | `unity_sender/` | `FoundationPose_Handoff_2026-08-14.md` L87–L88 |
| Config files | implemented | argparse + OmegaConf `weights/*/config.yml` + BundleSDF yml | `bundlesdf/config_ycbv.yml`；weights configs **缺失** | `predict_score.py` L124–L126；`bundlesdf/config_ycbv.yml` L1–L16 |
| Dataset files | partial | YCB-V / LINEMOD / mustard demo / Polycam | Polycam **存在**（约 99 camera JSON）；YCB/LINEMOD/mustard **本 checkout 不存在** | `run_ycb_video.py` L137；`Polycam file/2026_8_16/` |

Live FPFRAME **不传 mesh、不传 camera extrinsics / SLAM pose**（`FOUNDATIONPOSE_TO_UNITY_HANDOFF.md` L42–L43 与 `network_frame_protocol.py` header 字段 L97–L115 一致）。

### Outputs

| Item | Status | Format / unit | Example | Evidence |
|---|---|---|---|---|
| 3D position | implemented（作为 4×4 的 t） | meters（BOP/demo depth `/1e3`） | `pose[:3,3]` | `datareader.py` L122–L123, L314；`run_live.py` L466–L467 |
| 6DoF pose | implemented | 4×4 homogeneous `ob_in_cam` | `debug*/ob_in_cam/{frame_id}.txt` | `run_demo.py` L65–L66；`run_live.py` L567 |
| Rotation matrix | implemented | 3×3 in 4×4 | 同上 | `estimater.py` L269–L276 |
| Quaternion | implemented **only for result smoothing** | **wxyz** | 未保存为独立文件 | `run_live.py` `rotation_matrix_to_quat` L254–L271, `smooth_pose` L305–L314 |
| Euler angles | not an output API | pytorch3d imported in `Utils.py` L11 | N/A as saved result | `Utils.py` L11 |
| 4×4 transform | implemented | text / JSON list | txt + `FPRESULT.ob_in_cam` | `run_live.py` L429, L557–L558 |
| object-to-camera pose | implemented | `ob_in_cam` | 变量名与路径 | `run_live.py` L429；`FOUNDATIONPOSE_TO_UNITY_HANDOFF.md` L431–L441 |
| camera-to-object pose | implemented on model-free path | `cam_in_ob` | Polycam converter | `docs/POLYCAM_MODEL_FREE.md` L89–L94 |
| Unity transform | **not implemented** | N/A | N/A | `unity_sender/README.md` L30, L88–L91 |
| AR overlay result | PC overlay implemented；Unity AR overlay **N/A** | PNG / GUI | `track_vis/` if enabled | `run_live.py` L230–L237, L571–L581 |
| logs | stdout + logging | text | `print_pose_report` | `run_live.py` L458–L468 |
| CSV / JSON result files | partial | JSON on wire (`FPRESULT`)；YCB/LM `*_res.yml`；无 CSV benchmark | 本 checkout **没有已保存结果文件** | `run_ycb_video.py` L129–L130；`run_live.py` L419–L432 |
| visualization images / videos | implemented if `debug>=2` / `--save_track_vis` | PNG | 本 checkout 无 `debug/` 产物 | `run_demo.py` L76–L78 |

---

## 4. Key Files and Code Evidence

Grep `.` 计数是非空行；下表 **Lines** 在已通读文件上用实际末行号，其余标 `~`。

| File | Lines | Role | Evidence Summary |
|---|---:|---|---|
| `readme.md` | 255 | 上游身份、安装、demo/YCB/LINEMOD 命令、BOP 声明 | L1–L7, L69–L194, L250 |
| `run_demo.py` | 80 | 官方 model-based demo 入口 | L15–L78 register then track |
| `run_live.py` | 622 | Live 入口：recorded/network、状态机、FPS、FPRESULT | L471–L621 |
| `estimater.py` | 305 | FoundationPose 核心 register/track | L51–L304 |
| `datareader.py` | ~614 | mustard/BOP/YCB/LINEMOD IO 与 GT pose 读取 | L57–L127, L307–L349, L508–L510 |
| `Utils.py` | ~1023 | 渲染、depth filter、ADD/ADD-S 函数、OpenGL↔OpenCV | L70–L73, L135, L234–L268, L401 |
| `network_frame_protocol.py` | 211 | FPFRAME v1 编解码 | L9–L11, L66–L211 |
| `send_recorded_frames.py` | 63 | 本地 TCP sender | L17–L62 |
| `inspect_fpframe_stream.py` | ~351 | 协议诊断接收器 | L305–L320 |
| `check_network_packets.py` | 111 | recorded round-trip exactness | L15–L95 |
| `learning/training/predict_score.py` | 227 | Scorer 加载与推理 | L117–L154, L160 |
| `learning/training/predict_pose_refine.py` | ~297 | Refiner 加载与迭代 | L93–L141, L149–L237 |
| `learning/models/score_network.py` | ~91 | ScoreNet CNN | L27 |
| `learning/models/refine_network.py` | ~94 | RefineNet CNN | L26 |
| `run_ycb_video.py` | 150 | YCB-V 逐 keyframe **register-only**，写 yaml | L43–L130, L114–L115 跳过非 keyframe |
| `run_linemod.py` | 150 | LINEMOD register-only，写 yaml | L90–L133 |
| `run_polycam_model_free.py` | 56 | Polycam 一键 convert + 可选 NeRF | L43–L51 |
| `tools/polycam_to_foundationpose.py` | ~729 | Polycam→FP 目录 + YOLO-seg | L20–L22, L143–L197, L511 |
| `tools/polycam_quality_pipeline.py` | ~1038 | 选视图 / pose alignment 质量 | L400+, L754 |
| `bundlesdf/run_nerf.py` | ~116 | Neural Object Field | L18–L46 |
| `bundlesdf/run_nerf_custom.py` | ~105 | 自定义物体 NeRF | L27–L101 |
| `check_env.py` | 123 | 环境/权重路径检查（不推理） | L32–L98 |
| `docker/dockerfile` | 61 | 原 CUDA 11.3 / Python 3.8 / torch 2.0.0 | L1, L33, L41 |
| `docker/dockerfile.5070ti` | 73 | CUDA 12.8 / torch nightly / sm_120 | L1, L9, L35–L38 |
| `unity_sender/README.md` | 91 | Unity 脚本说明与范围限制 | L16–L91 |
| `.../FPFrameProtocol.cs` | ~267 | C# FPFRAME builder | L65–L135 |
| `.../FoundationPoseTcpSender.cs` | ~470 | TCP client；latest-frame-wins drop | L20–L48, L171–L194 |
| `.../FoundationPoseFrameStreamer.cs` | ~546 | AR Foundation 采集 | L12–L27, L180–L264 |
| `.../YoloBBoxToMaskAdapter.cs` | 177 | bbox→矩形 mask | L13–L114 |
| `.../FoundationPoseSyntheticSender.cs` | 156 | 无 AR 的协议测试 | L5–L80 |
| `LICENSE` | 95 | NVIDIA 非商业研究许可 | L53–L57 |

无 `main` 训练脚本被本审计跑通：`learning/training/` 仅有 predictor + `training_config.py`，**没有** 可直接提交的 train entry 被确认用于本项目实验。

---

## 5. Dependencies and Versions

| Item | Value | Status | Evidence |
|---|---|---|---|
| Python | conda pin **3.11**；原 Docker **3.8**；5070ti 镜像 `python3`（Ubuntu 22.04 通常 3.10） | **contradiction / UNKNOWN at runtime** | `environment.yml` L10；`docker/dockerfile` L33；`docker/dockerfile.5070ti` L25–L29 |
| CUDA | 原图 **11.3** + torch cu118；5070ti **12.8** / `TORCH_CUDA_ARCH_LIST=12.0`；conda 文档示例 cu124 | multiple supported paths | `docker/dockerfile` L1, L41；`dockerfile.5070ti` L1, L9；`readme.md` L136 |
| PyTorch | 原 Docker `2.0.0+cu118`；5070ti **nightly cu128**；conda **未 pin** | UNKNOWN in local env | `docker/dockerfile` L41；`dockerfile.5070ti` L35–L38；`requirements.txt` L1–L2 |
| OpenCV | `opencv-python` **未 pin**；原 Docker 另装 `opencv-contrib-python` | UNKNOWN exact version | `requirements.txt` L23；`docker/dockerfile` L55 |
| Unity | **UNKNOWN**。本仓库不是 Unity 工程，无 `ProjectSettings/ProjectVersion.txt` | unknown | `FoundationPose_Handoff_2026-08-14.md` L87–L88 |
| FoundationPose version / commit | 本工作区 **不是 git repo**；commit **UNKNOWN**。算法对应 CVPR 2024 官方实现 + 本地 live/Polycam/5070ti 扩展 | unknown commit | `readme.md` L1–L3；user_info: not a git repo |
| Segmentation | Polycam：`ultralytics==8.0.120` + `yolov8x-seg.pt`；Unity：bbox 矩形，不是 YOLO-seg | implemented / partial | `dockerfile.5070ti` L50；`run_polycam_model_free.py` L16–L18；`YoloBBoxToMaskAdapter.cs` L102–L114 |
| Third-party models | FoundationPose scorer `2024-01-11-20-02-45`；refiner `2023-10-28-18-33-37`；YOLO `yolov8x-seg.pt`；nvdiffrast；pytorch3d；kaolin（model-free） | weights **missing in checkout** | `predict_score.py` L120–L124；`predict_pose_refine.py` L97–L100；`check_env.py` L82–L98 |
| System libraries | Eigen, Boost, pybind11, cmake, ninja；Docker：EGL/GL, boost, eigen | documented | `environment.yml` L12–L17；`dockerfile.5070ti` L21–L25 |
| `pyproject.toml` / `package.json` | **N/A** | — | 未找到 |
| `requirements.txt` | 存在，未锁 torch/opencv 精确版本 | implemented | `requirements.txt` L11–L43 |

---

## 6. Hardware and Runtime Assumptions

| Assumption | Status | Evidence |
|---|---|---|
| GPU | **required**。tensors `.cuda()`，无 CPU-only 推理路径 | `estimater.py` L96–L97, L107；`predict_score.py` L148, L156 |
| CUDA | required（nvdiffrast `RasterizeCudaContext`） | `run_demo.py` L40；`estimater.py` L204 |
| Specific VRAM | **UNKNOWN**（代码未检查显存） | 无 `memory_allocated` / `nvidia-smi` 调用 |
| CPU-only fallback | **not implemented** | `.cuda()` 硬编码 |
| LiDAR | Unity live：**AR Occlusion raw environment depth**（iPhone LiDAR 路径） | `FoundationPoseFrameStreamer.cs` L146–L150, L327–L351 |
| RGB-D camera | demo/YCB 为 RGB-D 数据集；live 为 AR RGB + depth | `datareader.py` L107–L126 |
| iPhone / iPad | **intended** via AR Foundation scripts；真机是否已编译 **UNKNOWN** | `unity_sender/README.md` L3–L4, L55–L59 |
| Webcam | **N/A**（无 OpenCV VideoCapture 主路径） | 未找到 webcam capture 入口 |
| Depth camera | required（depth 无效则 register 退化） | `estimater.py` L220–L225 |
| Unity runtime | sender 需要；本仓库不能独立运行 Unity | `unity_sender/README.md` L50–L59 |
| Local network | TCP `0.0.0.0:5000`；Docker `-p 5000:5000` | `run_live.py` L600–L601；`docker/run_container_5070ti.sh` L16 |
| Client-server | **yes**：Unity/iPhone client → PC server | `NetworkFrameProvider` L129–L135 |
| RTX 5070 Ti / Blackwell | 本地适配路径存在 | `readme.md` L88–L108；`env_5070ti.sh` L1–L12；`dockerfile.5070ti` L9, L32–L33 |

---

## 7. Algorithms, Models, and Third-party Sources

| Algorithm / model | In code? | README/docs only? | Notes |
|---|---|---|---|
| FoundationPose register (icosphere rotation grid + depth centroid + refine + score) | **actually used** | also in README | `estimater.py` L139–L276 |
| FoundationPose tracking (refiner from last pose) | **actually used** | yes | `estimater.py` L286–L304 |
| nvdiffrast differentiable/raster render | **actually used** | yes | `Utils.py` L18, L135；predictors |
| Warp CUDA kernels (erode/bilateral depth) | **actually used** if `warp` imports | — | `Utils.py` L55–L59, L307–L389 |
| ScoreNet / RefineNet pretrained | **actually used** if weights present | README download | `predict_score.py` L120–L154；weights **missing here** |
| BundleSDF / Neural Object Field / hash grid (torch-ngp style) | **code present** | README model-free | `bundlesdf/run_nerf.py` L18–L40；`nerf_runner.py` |
| YOLOv8-seg (`ultralytics`) | **actually used** in Polycam converter | docs | `tools/polycam_to_foundationpose.py` L143–L197 |
| YOLO bbox rectangle mask (Unity) | **actually used** | README 明确非最终质量 | `YoloBBoxToMaskAdapter.cs` L102–L114；`unity_sender/README.md` L41–L44 |
| SAM / SAM2 | **literature/planned only** | handoff L53, L553 | Python 中无 SAM 调用 |
| OpenCV PnP | **not used as pose solver** | N/A | 无 `solvePnP` |
| ICP | **not used** as pose estimator。Polycam 有 point-alignment **quality**（cKDTree distances），不是 ICP 6DoF solver | quality pipeline | `polycam_quality_pipeline.py` L400+ |
| SLAM | **not implemented** | N/A | — |
| AprilTag / ArUco | **not implemented** | N/A | — |
| NeRF | **used for mesh reconstruction**, not live pose | README | `run_neural_object_field` |
| Gaussian Splatting | **not present** | N/A | — |
| Depth estimation (learned monocular) | **not used**；depth 来自传感器/数据集 | N/A | — |
| Kaolin | model-free Docker 依赖 | README L158 | `dockerfile` L45–L47 |
| LLM texture / Stable Diffusion aug | README 说明 **不能 release** | `readme.md` L240–L241 | literature-only for this checkout |

**License / citation 风险：**

- NVIDIA Source Code License **非商业**（`LICENSE` L53–L57）→ IEEE VR 研究使用需确认合规；商业 demo 有风险。
- 必须引用 FoundationPose CVPR 2024；model-free 再引 BundleSDF CVPR 2023（`readme.md` L37–L56）。
- Ultralytics YOLO、Polycam 数据、Kaolin、nvdiffrast、pytorch3d 均需在论文 Related Work / Acknowledgements 列出。
- YOLO 权重 `yolov8x-seg.pt` 不在仓库内，运行会触发下载（本次审计未执行）。

---

## 8. Data Formats and Dataset Structure

### 本 checkout 实际存在的数据

| Path | 含义 | 单位 / convention | Sample? |
|---|---|---|---|
| `Polycam file/2026_8_16/keyframes/images/*.jpg` | RGB | pixels | **yes**（采集数据，非论文实验结果） |
| `.../corrected_images/*.jpg` | 校正 RGB | pixels | yes |
| `.../depth/*.png` | LiDAR depth | 按 converter 按 **uint16 mm** 处理 | yes |
| `.../confidence/*.png` | depth confidence | 离散等级 | yes |
| `.../cameras/*.json` + `corrected_cameras/*.json` | fx,fy,cx,cy,width,height, `t_00`..`t_23`, `timestamp` | timestamp 为整数（样本 `25540516047`）；pose `t_ij` 被当作 OpenGL/ARKit cam-to-world | yes，约 **99** 个 camera JSON |
| `Polycam file/2026_8_16/thumbnail.jpg` | 缩略图 | — | yes |

样本相机字段：`Polycam file/2026_8_16/keyframes/cameras/25540516047.json` L5–L22（`fx=744.20013`, `width=1024`, `height=768`, `timestamp=25540516047`）。

**Pose convention（Polycam→FP）：**  
`cam_in_ob = T_polycam @ glcam_in_cvcam`，`glcam_in_cvcam = diag(1,-1,-1,1)`。证据：`tools/polycam_to_foundationpose.py` L20–L22, L511；`docs/POLYCAM_MODEL_FREE.md` L89–L94；`Utils.py` L70–L73。

**Timestamp convention：**

- Polycam JSON `timestamp`：整数，文件名同 stem。
- FPFRAME / recorded：`frame_id` 若可解析为 int，则 `timestamp = int(frame_id)*1e-9` 秒（`network_frame_protocol.py` L59–L63；`run_live.py` L69–L73）。
- Unity AR：`XRCpuImage.timestamp` 秒（`FoundationPoseFrameStreamer.cs` L182–L184）；synthetic 用 `Time.realtimeSinceStartupAsDouble`（`FoundationPoseSyntheticSender.cs` L47）—— **与 ARKit 时钟不同域**（`FOUNDATIONPOSE_TO_UNITY_HANDOFF.md` L385–L398）。

### 代码期望但本 checkout 缺失

| Path | 用途 |
|---|---|
| `demo_data/mustard0/{rgb,depth,masks,mesh,cam_K.txt}` | demo / live recorded regression |
| `weights/2024-01-11-20-02-45/model_best.pth` + `config.yml` | scorer |
| `weights/2023-10-28-18-33-37/model_best.pth` + `config.yml` | refiner |
| `demo_data/polycam_model_free/` 等转换输出 | model-free 训练目录 |
| YCB-Video / LINEMOD 根目录 | 官方 benchmark |
| `debug/`, `debug_live/`, `*_res.yml` | 运行产物 |

**不要把 Polycam 原始扫描当作 FoundationPose 实验结果。**

---

## 9. Ground Truth / Reference Pose Mechanism

| Mechanism | Status | Evidence |
|---|---|---|
| GT pose files (BOP `scene_gt.json`) | **reader implemented**；数据集 **不在本 checkout** | `datareader.py` L172–L178, L307–L349：`cam_R_m2c` + `cam_t_m2c/1e3` → `ob_in_cam` |
| Demo `annotated_poses/` | reader 尝试加载；失败返回 None | `datareader.py` L77, L98–L104 |
| GT 用于 **metric 计算** | **未接线**。`est.gt_pose = reader.get_gt_pose(...)` 被赋值，但 `compute_add_err_to_gt_pose` **恒返回 -1** | `run_ycb_video.py` L67–L68；`estimater.py` L247–L248, L279–L283 |
| `add_err` / `adds_err` / `compute_auc_sklearn` | **函数存在，全仓库无调用点** | `Utils.py` L234–L268；Grep 仅定义处 |
| SLAM pose | **N/A** | — |
| AprilTag / ArUco | **N/A** | — |
| Manual measurement | **N/A** | — |
| Synthetic / simulator GT | 仅 FPFRAME **协议** synthetic RGB/depth，**不是 pose GT** | `FoundationPoseSyntheticSender.cs` L50–L57；`inspect_fpframe_stream.py` L11–L48 |
| Camera calibration | K from file/stream；无标定实验脚本 | — |
| Object frame | mesh 原点；register 输出会乘 `get_tf_to_centered_mesh()` | `estimater.py` L115–L118, L269–L270 |
| camera-to-world / world-to-camera | model-free `cam_in_ob`；live 路径 **没有 world** | `bundlesdf/run_nerf.py` L23 |
| Unity coordinate transform | **not implemented** | `unity_sender/README.md` L88–L91 |
| Timestamp alignment (eval) | RGB/depth sync 有阈值 50 ms；**无 pose↔GT 时间对齐脚本** | `FoundationPoseFrameStreamer.cs` L21, L184–L188 |

**Translation Error 能否计算？**

- **YCB-V / LINEMOD：** **computable after running** `run_*` 得到预测 yaml，并自己写脚本对齐 `scene_gt.json`。当前 repo **没有** 该评估脚本，也 **没有** 预测文件。
- **Live iPhone / mustard demo / Polycam AR：** **不能** 作为论文 accuracy。缺外部 GT（marker / mocap / 精密测量 / 仿真 GT）。
- `compute_add_err_to_gt_pose` 是 stub，**不能**当作已计算 ADD。

**Rotation Error：** 同上。代码有测地线角用于 **tracking-loss heuristic**（`run_live.py` L248–L251, L450），比较的是 **相邻帧预测**，不是 GT。

**当前评价上限：**  
- 官方数据集：最多能做 **reference-based evaluation**（需补齐数据 + 评估脚本）。  
- Live AR：目前只有 **qualitative**（overlay）+ **self-consistency / jitter heuristics**。  
- Polycam：有 **多视图点云对齐误差（mm）** 作为重建质量 proxy，**不是** 6DoF tracking vs GT。

---

## 10. Evaluation Scripts, Metrics, and Logs

| Artifact | Status | Location |
|---|---|---|
| Evaluation scripts computing ADD/ADD-S/AUC | **missing as runnable pipeline** | 仅 `Utils.py` L234–L268 |
| Benchmark runners | **pose dump only** | `run_ycb_video.py` L129–L130 → `debug/ycbv_res.yml`；`run_linemod.py` L132–L133 |
| Notebooks | **N/A**（0 个 ipynb） | — |
| Result CSV/JSON/TXT in repo | **missing** | 无 `debug/` 产物 |
| Protocol exactness checker | implemented（非 pose metric） | `check_network_packets.py` L88–L95 |
| Visualization | on-demand | `debug*/track_vis/`（未生成） |
| Saved predictions | code writes `ob_in_cam/*.txt` | 运行后才有 |
| Saved GT | BOP `scene_gt.json` 需外部数据集 | — |

| Metric | already computed / from logs / needs instrumentation / missing | 现有位置 | 计算方式 | 单位 | 可信度 |
|---|---|---|---|---|---|
| Translation Error | **missing**（函数未调用） | `Utils.py` 未接入 | ‖t_pred−t_gt‖ 或 ADD | m 或 cm | 无结果 |
| Rotation Error | **missing** vs GT；相邻帧 delta **已有启发式** | `run_live.py` L448–L454 | geodesic | deg | 不能当 accuracy |
| Latency | **partially computed** per-frame processing | `run_live.py` L516–L551, L464 | `perf_counter` 整段 register/track | s / 可转 ms | 中：含 preprocess+infer，无 stage 分解，无 warmup 标记 |
| FPS | **computed as 1/processing_time** | L550–L551, L465 | 非墙钟吞吐 | FPS | 中：`latest_frame_only` 会丢帧，该 FPS 不是端到端显示帧率 |
| Success Rate | **missing** | lost-reason 只触发 mask_request | 需定义阈值 | % | 无 |
| Pose Jitter | **needs minor instrumentation** | 已有 `trans_delta`/`rot_delta` 未落盘 | 相邻帧 | cm / deg | 可补 log |
| Robustness | **missing** | 无 occlusion/distance 实验 harness | condition-based | — | 无 |
| Registration / tracking time | **needs instrumentation** | 二者包在同一 `start_time` | 可用 `operation` 字段拆分已有 processing_time | ms | 可从 stdout 后处理，但无归档 |
| Network latency | **partial** `pc_queue_latency_ms` | L561 | receive→process start | ms | **不是** RTT / 采集到显示 |
| E2E latency | **missing** | 无 Unity receive timestamp 回传 | — | ms | 无 |
| Dropped frames | Unity **in-memory counters**；PC latest-only **未计数落盘** | `FoundationPoseTcpSender.cs` L48, L185–L187；`run_live.py` L617 | count | 未归档 |
| GPU / VRAM | **missing** | — | — | — | 无 |

交接文档中的 “约 20–35 FPS / 0.028–0.050 s / RTX 3070”（`FOUNDATIONPOSE_TO_UNITY_HANDOFF.md` L412–L424）是 **文档声称**，本仓库 **无对应 log/CSV**，论文中 **不得当作已验证 Results**，标 **UNKNOWN**。

---

## 11. Timing, FPS, and Latency Instrumentation

| Question | Answer | Evidence |
|---|---|---|
| `time.time` / `perf_counter` | **yes** | `run_live.py` L113, L155, L516–L517, L550, L560 |
| CUDA Event | **no** | Grep 无 `cuda.Event` |
| Unity profiler | **no** | 仅 `DateTime.UtcNow` 测 TCP `stream.Write`（`FoundationPoseTcpSender.cs` L295–L299） |
| Stage split (preprocess / infer / register / track / post / net / render) | **no**。整段 register 或 track 一次计时 | `run_live.py` L516–L550 |
| Per-frame latency | **yes, processing only** | L550, L464 |
| End-to-end latency | **no** | Unity 不收 FPRESULT；无 capture_ts→display_ts |
| FPS | **yes, 1/processing_time** | L551 |
| Dropped frames logged to file | **no** | Unity 有 `droppedTrackingFrameCount` 计数器；PC `latest_frame_only` 默认真（L617）会静默丢包 |
| Batch size recorded | scorer/refiner 内部 bs=512/1024，**不写入结果** | `predict_score.py` L69；`predict_pose_refine.py` L167 |
| GPU warmup | **not recorded**。`set_seed(0)` 且 `cudnn.benchmark=False` | `Utils.py` L224–L231 |
| `Utils.enable_timer` | 设为 0，未使用 | `Utils.py` L60 |

**风险：** 把 `processing_fps` 写成 “系统 FPS” 会高估端到端 AR 帧率（采集 `targetFps=5`，`FoundationPoseFrameStreamer.cs` L24）。

---

## 12. Pose Representation, Coordinate Systems, and Units

| Topic | Finding | Evidence | Paper risk |
|---|---|---|---|
| Translation unit | **meters** at estimator；depth PNG **mm** | `/1e3`：`datareader.py` L123；`network_frame_protocol.py` L84–L85, L184；BOP `cam_t_m2c/1e3` L314 | 与 cm 混用会差 100× |
| Rotation | 主输出 **3×3 in 4×4**；refiner 内部 **axis-angle** 增量 | `predict_pose_refine.py` L122–L123, L220–L222 | 评估应在矩阵上算 geodesic |
| Quaternion order | smoothing 用 **wxyz** | `run_live.py` L254–L275 | 若当 xyzw 交给 Unity 会错 |
| Euler | **unknown / unused** as I/O | — | 不要用 Euler 报 error |
| Pose direction | 估计输出 **object-in-camera (`ob_in_cam`)**；model-free 参考 **`cam_in_ob`** | `run_live.py` L429；`docs/POLYCAM_MODEL_FREE.md` L89–L94 | 比较 routes 必须统一 direction |
| Handedness | OpenCV/FoundationPose：**right-handed, x-right y-down z-forward**（标准 OpenCV cam）。Unity：**left-handed**。转换 **未实现** | `glcam_in_cvcam` `Utils.py` L70–L73 | AR overlay 错轴会被误判为 pose error |
| Axis flip / scale | Polycam：Y/Z flip via `glcam_in_cvcam`；RGB 下采样时 K 按分辨率缩放 | `polycam_to_foundationpose.py` L480–L511；`FoundationPoseFrameStreamer.cs` L191, L226 | 内参与图像分辨率不一致会偏平移 |
| Depth scale | mm→m 往返 | encode `*1000` / decode `/1e3` | PNG 量化约 1 mm |
| Centered mesh vs original | 返回值 `pose @ get_tf_to_centered_mesh()`；可视化再 `@ inv(to_origin)` | `estimater.py` L269–L270, L304；`run_live.py` L231, L383 | 用错 frame 会把 bbox 中心当物体原点 |
| RGB row flip in Unity PNG | `FPFrameProtocol.EncodeRgb` `FlipRows` L137–L139 | 与 OpenCV decode 对齐；orientation 有诊断工具 | `inspect_fpframe_stream.py` L52–L68 |

---

## 13. Network / Unity / AR Pipeline

**不是 N/A。** 这是本 fork 相对上游最重要的系统贡献。

| Topic | Status | Evidence |
|---|---|---|
| Client-server | implemented | Unity/TCP client → PC `listen(1)` `run_live.py` L129–L135 |
| Transport | **TCP only** | `socket.AF_INET, SOCK_STREAM` |
| Frame message | FPFRAME v1：4-byte BE header_len + JSON + RGB + depth + optional mask | `network_frame_protocol.py` L116–L118 |
| Pose serialization | JSON `FPRESULT`：`ob_in_cam` 4×4 list + 2D bbox/axis | `run_live.py` L185–L207, L419–L432 |
| Timestamp on frames | **yes** in header | `network_frame_protocol.py` L102 |
| Timestamp on pose return | copies `packet.timestamp` + `pc_received_time` / `pc_result_time` | `run_live.py` L423, L559–L561 |
| Unity receiver for pose | **not implemented** | `unity_sender/README.md` L30 |
| Coordinate conversion | **not implemented** | README L88–L91 |
| AR object replacement | **N/A** | 无虚拟物体挂载代码 |
| Overlay | PC OpenCV box+axis | `run_live.py` L230–L237 |
| Network latency measurement | queue latency only；TCP write ms on Unity | `run_live.py` L561；`FoundationPoseTcpSender.cs` L295–L299 |
| Drop / reconnect / timeout | Unity reconnect loop + send timeout；registration 不可丢，tracking latest-wins | `FoundationPoseTcpSender.cs` L26–L29, L145–L194, L207–L255 |
| E2E measurement feasible today? | **No** without Unity pose receive + synced clocks | 缺闭环 |

**额外矛盾：** `run_live.py` 默认 `--send_results True`（L615），但 Unity 脚本不读 socket 回包。PC 发送可能在对方不读时堆积（取决于 TCP window）；Unity 侧无法验证 pose。

---

## 14. Error Handling, Failure Modes, and Known Limitations

**TODO/FIXME：** 业务代码几乎没有。唯一 `TODO` 在 `bundlesdf/mycuda/torch_ngp_grid_encoder/grid.py` L47。

| Failure mode | Evidence | 论文可信度影响 |
|---|---|---|
| 无 mask / 空 mask → register 失败或零平移 | `run_live.py` L520；`estimater.py` L173–L177, L221–L225 | 首帧分割质量决定 REGISTER |
| Unity 矩形 bbox mask ≠ 真分割 | `unity_sender/README.md` L41–L44 | live accuracy 不能代表 FoundationPose 上限 |
| Depth 无效 / 格式不支持则丢帧 | `FoundationPoseFrameStreamer.cs` L353–L355 | 环境 LiDAR 质量未知 |
| RGB/depth 时间差 > 50 ms 丢帧 | L21, L184–L188 | 运动场景掉帧 |
| `latest_frame_only` 默认丢中间帧 | `run_live.py` L136–L138, L617 | 延迟↓、轨迹不完整 |
| Tracking jump → mask_request，但 REGISTER 不会自动成功 | L540–L546 | 无 success 标注 |
| `compute_add_err_to_gt_pose` 是 stub | `estimater.py` L279–L283 | 日志里 `add_errs min:-1` **毫无评价意义** |
| 无 CPU fallback | `.cuda()` | 复现必须 GPU |
| 权重 / demo_data 缺失 | Glob 0 | 本工作区 **不能复现推理** |
| NVIDIA 非商业许可 | `LICENSE` L53–L57 | 投稿/开源分发需声明 |
| README：无 diffusion 增广权重，预期性能略降 | `readme.md` L240–L241 | 与论文表格不可直接等同官方 BOP 第一 |
| Hard-coded 权重 run names 与默认 mesh 路径 | `predict_score.py` L120；`run_live.py` L592 | 换物体必须换 mesh |
| Clock domain mismatch | handoff L385–L398 | 错误丢弃 mask |
| Handoff 过时 | pose return / recovery 描述落后于 `run_live.py` | 写 Methods 必须以代码为准 |

**脆弱点：** 单 client `listen(1)`；无 pose ACK；mesh 与 Unity 检测物体必须人工一致（`FOUNDATIONPOSE_TO_UNITY_HANDOFF.md` L130–L145）。

---

## 15. Existing Experiment Configurations and Reproducible Commands

**未执行这些命令**（会下载/占用 GPU/写 debug）。仅记录。

| Purpose | Command | Config / Input | Expected Output | Evidence |
|---|---|---|---|---|
| Conda env | `conda env create -f environment.yml` | `environment.yml` | env `foundationpose` | `readme.md` L128–L131 |
| Pip deps | `pip install -r requirements.txt` | `requirements.txt` | packages | `readme.md` L152–L154 |
| Build C++ | `bash build_all_conda.sh` | 需 `CONDA_PREFIX` | `mycpp/build` | `build_all_conda.sh` L8–L21 |
| Env check | `python check_env.py --demo-data` | weights + mustard | stdout warnings | `check_env.py` L12–L18 |
| Official demo | `python run_demo.py` | mustard0 + weights | `debug/ob_in_cam`, GUI | `readme.md` L163；`run_demo.py` L18–L23 |
| Live recorded | `python run_live.py --provider recorded` | 同上 | `debug_live/ob_in_cam` | `run_live.py` L494–L597 |
| Live network | `python run_live.py --provider network --host 0.0.0.0 --port 5000` | mesh + TCP frames | poses + optional FPRESULT | `run_live.py` L496–L497；`unity_sender/README.md` L66–L67 |
| Local TCP sender | `python send_recorded_frames.py --host 127.0.0.1 --port 5000` | mustard0 | FPFRAME stream | `send_recorded_frames.py` L51–L62 |
| Packet exactness | `python check_network_packets.py` | mustard0 | RGB/Depth/K/Mask exact flags | `check_network_packets.py` L98–L110 |
| FPFRAME inspect | `python inspect_fpframe_stream.py --expect_synthetic` | Unity synthetic | decode stats | `unity_sender/README.md` L74–L76 |
| YCB-V | `python run_ycb_video.py --ycbv_dir ...` | 外部大数据集 | `debug/ycbv_res.yml` | `readme.md` L181；`run_ycb_video.py` L137–L141 |
| LINEMOD | `python run_linemod.py --linemod_dir ...` | 外部数据集 | `debug/linemod_res.yml` | `readme.md` L181 |
| Model-free NeRF | `python bundlesdf/run_nerf.py --dataset ycbv` | ref views | `model/model.obj` | `readme.md` L186–L188 |
| Polycam convert | `python run_polycam_model_free.py --input "Polycam file/2026_8_16" --output demo_data/polycam_model_free --num_views 24` | 本地 Polycam | converted dir | `docs/POLYCAM_MODEL_FREE.md` L23–L28 |
| Polycam + NeRF | 同上加 `--run`（**长、GPU、写目录**） | YOLO 权重可能下载 | `model/model.obj` | `run_polycam_model_free.py` L48–L51 |
| 5070ti Docker demo | `bash docker/run_demo_5070ti.sh` | 镜像 + demo_data | `debug_5070ti_docker` | `docker/run_demo_5070ti.sh` L1–L10 |
| 5070ti local wrapper | `bash run_5070ti.sh run_demo.py` | `.conda-envs/foundationpose5070` | `debug_5070ti` | `run_5070ti.sh` L4–L24 |

**当前阻塞复现（上午审计）：** 当时无 `weights/`、无 `demo_data/`。**同日已补齐**，5070 Ti Docker recorded 全序列已跑通。Polycam `--run` 仍未做。

---

## 16. Existing Result Files

| 类型 | 本仓库 | 论文可用性 |
|---|---|---|
| CSV / benchmark tables | **有** localhost TCP latency CSV | `results_today/tcp_pingpong/latency.csv`；`results_today/tcp_live30/latency.csv` |
| `ycbv_res.yml` / `linemod_res.yml` | **无** | 未跑官方 BOP |
| `ob_in_cam/*.txt` | **有** | `results_today/live_logged_full/`（737）；`live_wallclock/`、`live_recorded/` 同序列；**不要混用** `live_recorded2/` |
| logs | **有** | `results_today/live_logged_full/run.log`；TCP `run.log` |
| GPU | **有** | `results_today/gpu/gpu.csv` |
| 摘要 | **有** | `results_today/live_logged_full/metrics.txt`；`tcp_*/metrics.txt` |
| 交接文档中的 FPS 数字 | 仅 markdown | **不可**作 Results；用 §21 实测 |
| Polycam RGB/depth/JSON | **有**，属 **输入采集** | 可写 Methods 数据来源，**不是** pose accuracy 结果 |
| screenshots / videos | 上游 README 链到 GitHub assets；本地 `track_vis/` 视 debug 设置 | 定性可用，非精度 |
| 转换后的 `diagnostics/validation_report.json` | **无**（尚未跑 convert 输出） | N/A |

**结论：PC recorded + localhost TCP 已有可引用数字（§21）。Translation/Rotation Error、手机 WiFi、Unity 显示 e2e 仍无实测。**

---

## 17. Paper-ready Metrics Feasibility Table

填表数字与口径见 **§21**。下表只标可行性，避免和实测值打架。

| Metric | Status | Existing Evidence | How to Compute | Unit | Missing Pieces |
|---|---|---|---|---|---|
| Translation Error | **N/A today** | 无 `annotated_poses/` | ‖t_pred−t_gt‖ 或 ADD | cm | GT |
| Rotation Error | **N/A today** | 同上 | geodesic / ADD-S | degree | GT |
| Latency | **measured** | `live_logged_full/run.log` TRACK | skip 5 warmup median | ms | stage split 仍无 |
| FPS | **measured** | 处理 FPS + 墙钟 737/44.37 s | 必须分开报 | FPS | — |
| Success Rate | **measured (heuristic)** | 相邻帧阈值，非 ADD | Δt≤0.25 m 且 ΔR≤55° | % | 真值 success |
| Pose Jitter | **measured** | `live_logged_full/ob_in_cam` | 相邻帧 median | cm / deg | — |
| Robustness | **N/A** | 单序列 | 条件分桶 | — | 多条件采集 |
| Registration Time | **measured** | REGISTER `processing_time_sec` | 第一帧 | s | 多次冷启动均值 |
| Tracking Time | **measured** | 同 Latency | skip warmup | ms | — |
| Network Latency | **localhost measured; phone estimated** | `tcp_live30/latency.csv` | 排队 + 传输；手机填预估 | ms | 真机 WiFi |
| End-to-end Latency | **localhost measured; phone estimated** | send→FPRESULT | 手机+显示填预估 | ms | Unity 收包/显示 |
| Dropped Frames | **measured localhost 30 Hz** | PC `received/emitted/dropped` | 119/737 | % | 真机 5 Hz 未测 |
| GPU / VRAM | **measured** | `gpu/gpu.csv` | nvidia-smi 1 Hz | MiB / % | CUDA Event 级 util |

---

## 18. Missing Paper-level Experiments

按 IEEE VR 2027 **broader-scope multi-route comparison** 优先级：

### P0（没有则 Results 很难成立）

1. **Accuracy vs 可验证 GT**（至少一条）：YCB-V/LINEMOD 复现 **或** 自建 marker/mocap/精密测量的 live GT。当前 live 路径无法报 Translation/Rotation Error。
2. **把 per-frame pose + timing 落到 CSV**（否则 Methods 能写、Results 无数字）。
3. **明确这条 route 的比较协议**：model-based CAD vs 其他 routes；live AR 只比 latency/qualitative 必须写清楚，避免和精度表混为一谈。
4. **补齐可复现资产记录**：weights 版本/哈希、mesh、K、深度单位、`ob_in_cam` 定义。

### P1（强烈建议）

5. Latency/FPS：区分 REGISTER vs TRACK、排除 warmup、墙钟 FPS vs processing FPS、PC vs 采集 5 Hz。
6. Failure case 收集：空 mask、LiDAR hole、快速运动、bbox mask vs YOLO-seg。
7. Robustness：距离 / 视角 / 遮挡 / 运动（即使 GT 弱，也要条件标签 + 定性+jitter）。
8. Unity 闭环：接收 `FPRESULT`、坐标转换、一次 e2e latency（capture_ts→display_ts）。
9. Model-free Polycam：跑完 convert+NeRF，报告重建质量（已有 alignment mm）再接入 tracking。

### P2（有时间再补）

10. Ablation：`est_refine_iter` / `track_refine_iter`、mask 类型（bbox vs YOLO-seg vs GT mask）、`latest_frame_only`、smoothing α。
11. Occlusion / viewpoint / motion 专门序列。
12. User-facing AR quality（alignment 观感、替换稳定性）—— 现无 overlay 替换实现。
13. Cross-route 同一序列、同一 GT、同一时钟。
14. GPU/VRAM 占用表。

---

## 19. Minimal Additional Instrumentation

**不要改架构。** 建议只在现有 `run_live.py` 循环和（若做数据集）yaml dump 后加 logger。

**挂钩位置：**

- `run_live.py` `run_live()` 在 L550 之后、L567 附近：每帧写一行。
- 可选：`estimater.py` `register`/`track_one` 入口出口各打一个 `perf_counter`（preprocess vs network forward 仍可更细，但非最小）。
- Unity：若做 e2e，在未来 pose receiver 记录 `result.timestamp` 与 `Time.realtimeSinceStartupAsDouble`（需先解决 clock domain）。

**建议 CSV schema（每行一帧）：**

```text
frame_id, index, timestamp, pc_received_time, pc_result_time,
state, operation, processing_time_sec, processing_fps, pc_queue_latency_ms,
tx, ty, tz, r00,r01,r02,r10,r11,r12,r20,r21,r22,
smoothing_alpha, lost_reason, mask_present,
image_width, image_height,
cuda_max_mem_mb  # optional
```

**字段来源：** 几乎都已在 `FramePacket` / `print_pose_report` / `make_pose_result` / `tracking_lost_reason` 中。最小改动是 **fopen append CSV**。

| 需求 | 最小做法 |
|---|---|
| per-frame pose | 已有 `np.savetxt`；另写一行 CSV 便于分析 |
| latency / FPS | 已有 processing_time；按 `operation` 分列 |
| success / failure | 把 `tracking_lost_reason(...)` 返回值写入，即使未触发 request |
| GT alignment | 数据集模式才加 `gt_tx..`；live 填空 |
| network latency | 已有 queue latency；e2e 需 Unity 回传 display_ts |
| GPU / VRAM | 每 N 帧 `torch.cuda.max_memory_allocated()/1024**2` |
| dropped frames | PC：统计 `get_frame` 间隔 vs `packet.index`；Unity：周期性 log `DroppedTrackingFrameCount` |

---

## 20. Summary for IEEE VR 2027 Paper

### Methods 可写（有代码证据）

- 本 route 是 **FoundationPose RGB-D 6DoF**：首帧 **register**（旋转采样 + 深度质心 + refiner + scorer），随后 **track**（上一帧 pose 的 refiner）。
- 输入：**RGB、对齐 depth（米）、K、首帧 mask、物体 mesh**。
- 输出：**object-in-camera 4×4**，平移米、OpenCV 相机系。
- 系统层：iPhone/Unity **AR Foundation** 采 RGB+LiDAR depth，经 **FPFRAME v1 / TCP** 到 PC GPU；PC 可用 OpenCV 画 bbox/axis。
- 可选支路：Polycam 多视图 + YOLO-seg + BundleSDF 重建 mesh（model-free）。
- 坐标：Polycam/OpenGL `diag(1,-1,-1,1)` 转到 OpenCV；Unity 左手系转换 **尚未做**。

### Results 目前能支持的证据

- **能报** TRACK latency / 双口径 FPS / REGISTER / jitter / heuristic success / GPU / localhost TCP 时延与丢帧（§21，有 log/CSV）。
- **不能报** Translation/Rotation error、官方 BOP、多条件 Robustness。
- **手机 WiFi / Unity 显示 e2e：** 未测；§21 给预估，正文必须写 estimated。
- 交接文档 20–35 FPS **不可**直接当 Results；用 5070 Ti 实测。
- Polycam 文件夹证明有 **iPhone LiDAR 采集**，不是 tracking benchmark 结果。

### Discussion — strengths（基于实现，非夸大性能）

- 统一 register+track，CAD 即可 novel object（与论文定位一致）。
- 已把学术估计器接到 **真实 RGB-D 网络流水线**（AR 相关）。
- 对丢失有 **可解释启发式**（出画、深度空洞、位姿跳变）。
- Model-free 工具链完整写在代码里，适合作为同一 route 的 CAD-unavailable 变体。
- 深度/内参/协议单位在代码里相对明确（mm vs m），有利于和其他 routes 对齐。

### Discussion — limitations（必须写）

- Live 精度 **无 GT**；Unity mask 是 **矩形 bbox**。
- 无端到端 AR overlay/object replacement；Unity 不接收 `FPRESULT`。
- `latest_frame_only` 与默认 5 Hz 采集使 “FPS” 语义含糊；必须同时报 processing 与 wall-clock。
- 评价函数 ADD 未接入；heuristic 100% **不是** 精度。
- NVIDIA 非商业许可与 “无 diffusion 权重、性能略降”。
- 与 Unity 左手系未转换，不能把 PC overlay 等同 headset 对齐质量。
- 手机网络数字为预估，随 WiFi 变化大。

### 与其他 routes 横比时最重要的指标

1. **6DoF accuracy（cm / deg 或 ADD）** — 有 CAD 的 model-based 核心竞争力；现在 **测不了 live**。  
2. **TRACK latency vs REGISTER latency** — 实时 AR 对比关键。  
3. **输入假设成本**：CAD vs 多视图重建 vs marker。  
4. **Mask 质量依赖**（bbox vs seg vs GT）。  
5. **端到端延迟与丢帧** — 系统 route 对比；精度 route 不够。  
6. **失败模式**（LiDAR hole、运动、遮挡）— VR 审稿关心。

### 当前最大风险

把 **PC 延迟/jitter** 写成 **live iPhone 精度或真机 e2e**。无 GT；手机网络未测。第二风险：handoff 的 20–35 FPS 与 BOP 第一被误贴到本 fork 的 5070 Ti / iPhone 设置。

### 下一步最值得做的 3 件事

1. **精度：** 选定 GT（BOP 或 marker live），才能填 Translation/Rotation Error。  
2. **论文口径：** latency/FPS 用 §21 实测；网络/真机 e2e 标明 estimated，或明确 Results 只覆盖 PC。  
3. **闭环（可选）：** Unity 收 `FPRESULT` + 坐标转换；不做则 AR 质量只写 qualitative。

---

## 21. IEEE VR 指标表（2026-08-21 填完）

投稿用这一张。硬件：**RTX 5070 Ti 16 GB**，Docker CUDA 12.8，`demo_data/mustard0`，mesh `textured_simple.obj`，seed=0，`est_refine_iter=5`，`track_refine_iter=2`。  
**测得** = 有 `results_today/` 证据。**预估** = 未跑 iPhone/Unity（网络强绑定）。不要把预估写成 measured。

### 填表值（论文单元格）

| Metric | 填写 | 单位 | 口径 | 证据 |
|---|---|---|---|---|
| Translation Error | **N/A** | cm | 无 GT（`annotated_poses/` 不存在） | — |
| Rotation Error | **N/A** | deg | 同上 | — |
| Latency | **28.6** | ms | TRACK processing median，skip 5 warmup，n=731 | `live_logged_full/run.log` |
| FPS | **35.0 processing / 16.6 wall-clock** | FPS | 处理 = 1/28.6 ms；墙钟 = 737/44.37 s（含 REGISTER 与存盘） | 同上 + wall_s |
| Success Rate | **100%**（730/730） | % | 启发式 Δt≤0.25 m 且 ΔR≤55°。**不是 ADD / 不是 vs GT** | `live_logged_full/ob_in_cam` |
| Pose Jitter | **0.026 cm / 0.31°** | cm / deg | 相邻帧 median，skip 前 6 帧，n=730；max jump 1.78 cm / 3.64° | 同上 |
| Robustness | **N/A** | — | 只有一条 mustard0 序列 | — |
| Registration Time | **1.83** | s | 第一帧 REGISTER processing | `live_logged_full/run.log` |
| Tracking Time | **28.6** | ms | 与 Latency 同一口径 | 同上 |
| Network Latency | **~25（预估，WiFi LAN）** | ms | iPhone→PC 单向，JPEG，局域网。Localhost 测得：排队 4.1 + 传输解码 5.9 | 预估；测得见 `tcp_live30/latency.csv` |
| End-to-end Latency | **~100（预估，5 Hz JPEG LAN）** | ms | 采集后发送→PC 算完→JSON 回到手机；**不含 Unity 把 pose 画上屏幕**（未实现）。Localhost 测得 44.5 | 预估；测得见 `tcp_live30` TRACK skip5 |
| Dropped Frames | **16.1%**（30 Hz localhost）；**~0%（预估，手机 5 Hz）** | % | 30 Hz JPEG + `latest_frame_only`：119/737。Unity 默认 `targetFps=5`，处理 29 ms < 200 ms，预估几乎不丢 | `tcp_live30/run.log` |
| GPU / VRAM | 峰值 **7879 MiB（48%）**；TRACK **~3.5–3.6 GB** | MiB | REGISTER 瞬时 97% util；TRACK 的 1 Hz nvidia-smi 只有 10–19%（短 kernel 被低估） | `gpu/gpu.csv` |

建议正文句子：

- Latency / Tracking Time：**28.6 ms** (median TRACK, RTX 5070 Ti, mustard0, n=731).
- FPS：**35.0** processing vs **16.6** wall-clock; do not report only 35.
- Registration：**1.83 s** (n=1 this run; an earlier complete log saw 2.63 s when colder).
- Jitter: **0.026 cm / 0.31°** median per frame.
- Success: 100% self-consistency under a loose lost-track threshold; **not accuracy**.
- Network / e2e: **estimated** ~25 ms WiFi one-way and ~100 ms phone loopback at 5 Hz JPEG; localhost lower bound 4.1+5.9 ms network pieces and 44.5 ms send→FPRESULT.

### 预估怎么来的（Network / E2E）

Localhost JPEG 30 Hz、已处理帧（TRACK skip5 median）：

| 段 | 测得 |
|---|---|
| 编码+TCP+解码 | 5.9 ms |
| PC 排队 | 4.1 ms |
| TRACK 推理 | 27.6 ms |
| FPRESULT 回包 | 0.2 ms |
| send→FPRESULT | **44.5 ms** |

手机未测，按局域网 JPEG、Unity 默认 **5 Hz** 加项：

| 段 | 预估 |
|---|---|
| iPhone JPEG+depth 编码 | 10–20 ms |
| WiFi 单向（相对 localhost 多出来的） | 15–35 ms |
| PC 排队（5 Hz，几乎不积压） | ~1 ms |
| TRACK 推理 | 28.6 ms（测得） |
| 小 JSON 回程 | 2–8 ms |
| **合计（发送→回到手机）** | **约 80–120 ms，填表用 ~100 ms** |
| 若再加一帧 Unity 显示（未实现） | 再加 ~8–16 ms → ~90–140 ms |

30 Hz 真机：排队和丢帧都会变差。Localhost 已丢 **16.1%**；WiFi 预估 **15–30%**。不要用 30 Hz 的 16% 去描述默认 5 Hz 采集。

Ping-pong PNG localhost（737/737，0 丢帧）TRACK e2e median **41.8 ms**，排队 **0.06 ms**，传输 **14.6 ms**。这是无积压下限，**不是** 直播网络时延。

### 不要写进 Results 的

- 交接文档 20–35 FPS / 3070 数字  
- Translation/Rotation Error 的编造值  
- 把 heuristic 100% 写成 ADD 或 cm-level accuracy  
- 把 ~25 / ~100 ms 写成 measured on iPhone  

### 原始文件

- Recorded 全序列：`results_today/live_logged_full/`  
- GPU：`results_today/gpu/gpu.csv`  
- TCP ping-pong：`results_today/tcp_pingpong/`  
- TCP 30 Hz JPEG：`results_today/tcp_live30/`  
- 不要混用 `results_today/live_recorded2/`（另一段 400 帧）

