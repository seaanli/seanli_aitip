
"""
demo.py
-------
Full fencing tip tracking pipeline:
  1. Download YouTube video
  2. Extract frames
  3. Run YOLO inference
  4. Smooth & interpolate
  5. Render annotated output video
"""

import os, glob, math, cv2, numpy as np
from ultralytics import YOLO
from smoother import fill_missing_tips_farthest, fill_tips_midpoint

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_URL         = "https://www.youtube.com/watch?v=j8BWUpctZX4"
TIME_RANGE        = "00:07:40-00:08:28"
MODEL_PATH        = "best.pt"
DOWNLOAD_DIR      = "downloads"
FRAMES_DIR        = "output/frames"
OUTPUT_VIDEO      = "output/predict_tips_demo.mp4"
FRAME_INTERVAL    = 1
FRAME_SIZE        = (1920, 1080)
FRAME_SLICE       = (0, None)
BLADE_CLASS       = 2
BLADE_GUARD_CLASS = 0
CONF_THRES        = 0.3
CENTER_NMS_RADIUS = 25
EXPECT_COUNT      = 2
FPS               = 30

# ── STEP 1: Download ──────────────────────────────────────────────────────────
def download_youtube_video(url, output_dir, output_filename="video.mp4", time_range=None):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    section = f'--download-sections "*{time_range}"' if time_range else ""
    command = f"""
    yt-dlp \
      -S "res:1080,codec:h264:vp9:av1,fps,filesize" \
      -f "bv*+ba/b" \
      {section} \
      --force-keyframes-at-cuts \
      --recode-video mp4 \
      --postprocessor-args "ffmpeg:-c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart" \
      -o "{output_path}" "{url}"
    """
    os.system(command)
    print(f"[download] Saved to: {output_path}")
    return output_path

# ── STEP 2: Extract frames ────────────────────────────────────────────────────
def extract_frames(video_path, output_dir, target_size=(640, 480), frame_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[frames] {total} frames @ {fps:.1f} fps ({total/fps:.1f}s)")
    frame_idx, saved_idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % frame_interval == 0:
            resized = cv2.resize(frame, target_size)
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved_idx:05d}.jpg"), resized)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    print(f"[frames] Extracted {saved_idx} frames -> {output_dir}")

# ── Detection helpers ─────────────────────────────────────────────────────────
def center_radius_nms(recs, radius_px=25, max_keep=2):
    recs = sorted(recs, key=lambda r: r["conf"], reverse=True)
    kept, r2 = [], radius_px ** 2
    for r in recs:
        if all((r["cx"]-q["cx"])**2 + (r["cy"]-q["cy"])**2 > r2 for q in kept):
            kept.append(r)
        if len(kept) >= max_keep: break
    return kept

def farthest_corner_from_point(b, gx, gy):
    corners = [(b["x1"],b["y1"]),(b["x1"],b["y2"]),(b["x2"],b["y1"]),(b["x2"],b["y2"])]
    return max(corners, key=lambda c: (c[0]-gx)**2+(c[1]-gy)**2)

def pair_guards_to_blades(guards, blades, expect=2):
    used, pairs = set(), []
    for g in sorted(guards, key=lambda g: g["conf"], reverse=True):
        gx, gy = g["cx"], g["cy"]
        best, best_d2, best_i = None, 1e18, None
        for i, b in enumerate(blades):
            if i in used: continue
            d2 = (gx-(b["x1"]+b["x2"])/2)**2+(gy-(b["y1"]+b["y2"])/2)**2
            if d2 < best_d2: best, best_d2, best_i = b, d2, i
        if best is not None:
            used.add(best_i); pairs.append((g, best))
        if len(pairs) >= expect: break
    return pairs

# ── STEP 3: YOLO inference ────────────────────────────────────────────────────
def run_inference(model_path, frames_dir, frame_slice=(0, None)):
    model = YOLO(model_path)
    img_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.jpg")) +
        glob.glob(os.path.join(frames_dir, "*.jpeg")) +
        glob.glob(os.path.join(frames_dir, "*.png"))
    )
    start, end = frame_slice
    img_files = img_files[start:end]
    if not img_files: raise RuntimeError(f"No images found in {frames_dir}")
    first = cv2.imread(img_files[0])
    H, W = first.shape[:2]
    tips_history = []
    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is None: tips_history.append([]); continue
        results = model(img, verbose=False)[0]
        guards, blades = [], []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONF_THRES: continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1+x2)/2, (y1+y2)/2
            rec = {"cls":cls,"conf":conf,"cx":cx,"cy":cy,"x1":x1,"y1":y1,"x2":x2,"y2":y2}
            if cls == BLADE_GUARD_CLASS: guards.append(rec)
            elif cls == BLADE_CLASS:     blades.append(rec)
        guards = center_radius_nms(guards, CENTER_NMS_RADIUS, EXPECT_COUNT)
        blades = center_radius_nms(blades, CENTER_NMS_RADIUS, EXPECT_COUNT)
        tips = []
        if guards and blades:
            for g, b in pair_guards_to_blades(guards, blades, EXPECT_COUNT):
                tx, ty = farthest_corner_from_point(b, g["cx"], g["cy"])
                tips.append((int(tx), int(ty)))
        tips_history.append(tips)
    print(f"[inference] Processed {len(img_files)} frames")
    return img_files, tips_history, W, H

# ── STEP 4: Smooth & interpolate ──────────────────────────────────────────────
def smooth_and_interpolate(tips_history):
    filled_tips, is_missing = fill_missing_tips_farthest(tips_history, expect_count=EXPECT_COUNT)
    temp = [(filled_tips[0][i], filled_tips[1][i]) for i in range(len(filled_tips[0]))]
    tip_est = fill_tips_midpoint(temp, is_missing)
    tracks_smooth      = [(filled_tips[0][i].tolist(), filled_tips[1][i].tolist()) for i in range(len(filled_tips[0]))]
    tracks_interpolate = [(tip_est[i][0].tolist(), tip_est[i][1].tolist()) for i in range(len(tip_est))]
    print("[smooth] Done")
    return tracks_smooth, tracks_interpolate

# ── STEP 5: Render video ──────────────────────────────────────────────────────
def render_video(img_files, tips_history, tracks_smooth, tracks_interpolate, output_path, W, H):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened(): raise RuntimeError("Failed to open VideoWriter")
    for t, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None: continue
        vis = img.copy()
        raw = tips_history[t] if not isinstance(tips_history[t], tuple) else tips_history[t][1]
        for x, y in raw:
            cv2.drawMarker(vis,(int(x),int(y)),(0,255,0),cv2.MARKER_TILTED_CROSS,14,2,cv2.LINE_AA)
        for x, y in tracks_smooth[t]:
            if x is None or math.isnan(float(x)): continue
            cv2.circle(vis,(int(round(float(x))),int(round(float(y)))),8,(0,0,255),-1,cv2.LINE_AA)
        for x, y in tracks_interpolate[t]:
            if x is None or math.isnan(float(x)): continue
            cv2.drawMarker(vis,(int(round(float(x))),int(round(float(y)))),(0,255,255),cv2.MARKER_DIAMOND,14,2,cv2.LINE_AA)
        writer.write(vis)
    writer.release()
    print(f"[render] Saved -> {output_path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    video_path = download_youtube_video(VIDEO_URL, DOWNLOAD_DIR, time_range=TIME_RANGE)
    extract_frames(video_path, FRAMES_DIR, target_size=FRAME_SIZE, frame_interval=FRAME_INTERVAL)
    img_files, tips_history, W, H = run_inference(MODEL_PATH, FRAMES_DIR, FRAME_SLICE)
    tracks_smooth, tracks_interpolate = smooth_and_interpolate(tips_history)
    render_video(img_files, tips_history, tracks_smooth, tracks_interpolate, OUTPUT_VIDEO, W, H)
    print("Done! Output saved to:", OUTPUT_VIDEO)
