# Polycam Model-Free Reconstruction

This tool converts an extracted Polycam LiDAR Raw Data folder into the
FoundationPose model-free reference-view format, then optionally runs the
BundleSDF/Neural Object Field reconstruction.

## Input

Expected Polycam folders:

```text
keyframes/corrected_images
keyframes/corrected_cameras
keyframes/depth
keyframes/confidence
```

The converter pairs frames by timestamp filename intersection, not directory
order.

## One Command

```bash
python run_polycam_model_free.py \
  --input "Polycam file/2026_8_16" \
  --output demo_data/polycam_model_free \
  --num_views 24 \
  --run
```

For a cup, prefer YOLO segmentation masks:

```bash
python run_polycam_model_free.py \
  --input "Polycam file/2026_8_16" \
  --output demo_data/polycam_cup_yolo \
  --num_views 24 \
  --mask_mode yolo \
  --yolo_model yolov8x-seg.pt \
  --yolo_class cup \
  --run
```

YOLO mode requires instance segmentation masks. If a frame has no segmentation
mask for the requested class, that frame is logged as `YOLO-MASK-SKIP` and is
rejected; the converter does not fall back to a bounding-box mask.

On the 5070Ti Docker environment, run the same command inside the container.

## Output

```text
demo_data/polycam_model_free/ob_0000001/
  K.txt
  rgb/
  depth_enhanced/
  mask/
  cam_in_ob/
  diagnostics/
  manifest.json
  model/model.obj
```

## Diagnostics

Check these before trusting the model:

```text
diagnostics/pose_alignment.ply
diagnostics/object_cloud.ply
diagnostics/mask_overlay/
diagnostics/yolo_overlay/
diagnostics/selected_views_contactsheet.png
diagnostics/validation_report.json
```

If reconstruction quality is poor, inspect pose alignment and mask overlays
first. Bad masks or wrong camera convention matter more than simply increasing
training iterations.

## Conventions

Depth is kept as `uint16` millimetres in `depth_enhanced`; FoundationPose reads
it with `/ 1e3`.

RGB is downsampled to the LiDAR depth resolution. Intrinsics are scaled by the
actual image-to-depth size ratio.

Polycam `t_ij` is treated as OpenGL/ARKit camera-to-world. FoundationPose
expects OpenCV camera-to-object/world in `cam_in_ob`, so the converter writes:

```text
cam_in_ob = T_polycam @ glcam_in_cvcam
glcam_in_cvcam = diag(1, -1, -1, 1)
```
