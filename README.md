# fencingcomputervision
Trained computer vision model to track tip across an épée fencing bout. 

What It Does:

Downloads a YouTube fencing match video (configurable time range)
Extracts frames from the video
Runs a trained YOLO model to detect blade guards and blades
Infers the tip position of each blade from the detected bounding boxes
Applies smoothing and interpolation to clean up missing/noisy detections
Outputs an annotated video showing all three layers of tracking:

🟢 Green cross — raw YOLO detection
🔴 Red circle — smoothed tip (forward-carry)
🟡 Yellow diamond — interpolated tip (midpoint)

How It Works:

Tip Detection
The model detects two classes per frame: blade guards and blades. The tip of each blade is inferred by finding the corner of the blade bounding box that is farthest from the blade guard center — since the tip is always at the opposite end from the guard.
Smoothing — fill_missing_tips_farthest
When only one tip is detected in a frame (instead of two), this function:

Assigns the detected point to whichever previous tip it is closest to
Fills the missing tip using the last known position of the other tip (forward-carry)

When zero tips are detected, both tips are forward-carried from the previous frame, then back-filled from the next known frame if needed.
Interpolation — fill_tips_midpoint
For frames still marked as missing after smoothing, this function:

Finds the nearest known frame before and nearest known frame after the gap
Fills the missing frame with the midpoint between those two positions
Falls back to forward-carry or backward-carry if only one side is available


Link to example video used in original notebook: 
https://drive.google.com/file/d/1Z-d8HxU1qTRJ9SpXEckVYyNJi_9i_if5/view?usp=sharing


