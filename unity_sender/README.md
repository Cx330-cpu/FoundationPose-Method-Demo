# FoundationPose Unity/iPhone Sender

This folder contains Unity-side scripts for sending AR Foundation frames to the
existing PC `NetworkFrameProvider` over FPFRAME v1 TCP.

The PC pipeline remains unchanged:

```text
run_live.py --provider network
  -> NetworkFrameProvider
  -> FramePacket
  -> register() / track_one()
  -> FoundationPose Live GUI
```

## Scripts

- `FPFrameProtocol.cs`
  - Builds FPFRAME v1 messages.
  - Message layout: 4-byte big-endian JSON header length, JSON header, RGB blob,
    depth blob, optional mask blob.
  - RGB is PNG by default. JPEG is available for later smoke/performance tests.
  - Depth is uint16 PNG in millimeters.
  - Mask is optional uint8 PNG.

- `FoundationPoseTcpSender.cs`
  - TCP client for the PC server.
  - The first registration packet is non-droppable.
  - Tracking frames use latest-frame-wins.
  - There is no pose receive path in this phase.

- `FoundationPoseFrameStreamer.cs`
  - Reads RGB from `ARCameraManager.TryAcquireLatestCpuImage`.
  - Reads raw depth from `AROcclusionManager.TryAcquireRawEnvironmentDepthCpuImage`.
  - Compares `XRCpuImage.timestamp` for RGB/depth sync.
  - Uses raw depth native resolution as the final output resolution.
  - Downsamples RGB to depth resolution and scales K every frame.
  - Drops frames when RGB/depth aspect ratios do not match.
  - Sends the first frame with a mask, then sends tracking frames without masks.

- `YoloBBoxToMaskAdapter.cs`
  - Lets the existing CoreML/Vision YOLO flow provide a bbox.
  - Converts the bbox into a rectangle binary mask for first-frame registration.
  - This mask is only for sender/register validation, not final pose quality.

- `FoundationPoseSyntheticSender.cs`
  - Sends synthetic RGB/depth/K/mask frames without AR Foundation.
  - Use this first for Phase A protocol/TCP validation.

## Minimal Unity Wiring

1. Copy `Assets/Scripts/FoundationPose` into the Unity project.
2. Add `FoundationPoseTcpSender` to a GameObject and set the PC host/port.
3. For Phase A, add `FoundationPoseSyntheticSender` and link the sender.
4. For iPhone AR streaming, add `FoundationPoseFrameStreamer` and link:
   - `ARCameraManager`
   - `AROcclusionManager`
   - `FoundationPoseTcpSender`
   - `YoloBBoxToMaskAdapter`
5. Hook the existing YOLO code into `YoloBBoxToMaskAdapter.SetBoundingBox(...)`.

## PC Commands

Start the PC side before Unity connects:

```bash
python run_live.py --provider network --host 0.0.0.0 --port 5000 --debug 1 --debug_dir debug_live_iphone
```

For early protocol inspection, use the existing Python FPFRAME receiver helpers
from `network_frame_protocol.py`. This repository includes a diagnostic receiver
that does not change the FoundationPose live path:

```bash
python3 inspect_fpframe_stream.py --host 0.0.0.0 --port 5000 --num_frames 3 --expect_synthetic
```

## Validation Order

1. Synthetic FPFRAME packet reaches PC and decodes.
   - Confirm `RGB exact`, `Depth exact`, `K exact`, and `Mask exact` with
     `inspect_fpframe_stream.py --expect_synthetic`.
2. Real RGB plus K has the expected orientation and resolution.
3. Raw depth aligns with RGB and has reasonable meter values after decode.
4. YOLO bbox mask overlays the target in final RGB coordinates.
5. FoundationPose receives first-frame mask, enters REGISTER, then TRACKING.

## Current Scope

No pose return, no Unity coordinate conversion, no tracking-loss recovery, and
no smoothed depth path are implemented in this phase.
