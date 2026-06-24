#!/usr/bin/env python3
"""
============================================================
 RTSP FRAME COLLECTOR  --  Smart Parking Dataset
============================================================
Pulls still frames from your Tapo camera (or, later, the Pi
relay) over RTSP and saves them as JPG images for training.

HOW TO USE
  1. Install the one requirement (once):
         pip install opencv-python
  2. Edit the TWO lines in the CONFIG section below:
         RTSP_URL     -> your camera account + IP
         SAVE_FOLDER  -> where the JPGs should go
  3. (Optional) change FRAMES_PER_MINUTE.
  4. Run it:
         python collect_frames.py
  5. Stop it anytime with  Ctrl + C  -- it stops cleanly and
     tells you how many frames it saved.

WHAT IT DOES FOR YOU
  - Reconnects automatically if the stream drops.
  - Always saves a FRESH frame (it drains the buffer so you
    never get stale, delayed images).
  - Timestamps every filename, so separate runs never
    overwrite each other.
============================================================
"""

import os

# Force RTSP over TCP -> far more reliable over Wi-Fi.
# (Must be set before the camera stream is opened.)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
from datetime import datetime


# ============================================================
#  CONFIG  --  edit these
# ============================================================

# Your camera's RTSP link: rtsp://USER:PASS@CAMERA_IP:554/stream1
#   USER:PASS  = the "Camera Account" you made in the Tapo app
#   CAMERA_IP  = the camera's IP on your network (e.g. 192.168.1.50)
#   stream1    = the high-quality main stream (use this for training)
RTSP_URL = "rtsp://johelenmedz:october2028@192.168.254.115:554/stream1"

# Folder where the JPGs are saved. The 'r' before the quotes is
# required on Windows so the backslashes aren't misread.
SAVE_FOLDER = r"E:\Workspace\TestDataset"

# How many images to save each minute.
#   12  = one every 5 seconds   (good default)
#   6   = one every 10 seconds
#   2   = one every 30 seconds  (use for all-day, varied collection)
FRAMES_PER_MINUTE = 1

# JPG quality, 1-100 (higher = clearer image, bigger file).
JPG_QUALITY = 85

# ============================================================


def connect(url):
    """Open the RTSP stream. Returns a capture object, or None on failure."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def main():
    # --- sanity checks before we start ---
    if "USERNAME" in RTSP_URL or "CAMERA_IP" in RTSP_URL:
        print("[!] RTSP_URL still has placeholders.")
        print("    Edit it with your real camera account and IP, then run again.")
        return

    try:
        os.makedirs(SAVE_FOLDER, exist_ok=True)
    except OSError as e:
        print(f"[!] Can't create/use folder: {SAVE_FOLDER}")
        print(f"    Reason: {e}")
        print("    Make sure the drive is plugged in and the path is valid.")
        return

    interval = 60.0 / FRAMES_PER_MINUTE
    print("============================================================")
    print(f"  Saving ~{FRAMES_PER_MINUTE} frames/min (1 every {interval:.1f}s)")
    print(f"  Into: {SAVE_FOLDER}")
    print("  Press Ctrl + C to stop.")
    print("============================================================\n")

    cap = None
    saved = 0
    fails = 0
    last_save = 0.0

    try:
        while True:
            # (Re)connect if we don't have a working stream.
            if cap is None:
                print("[i] Connecting to camera ...")
                cap = connect(RTSP_URL)
                if cap is None:
                    print("[!] Could not connect. Retrying in 3s ...")
                    time.sleep(3)
                    continue
                print("[i] Connected.\n")

            # Read every frame to keep the stream live and current.
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                print(f"[!] Lost the stream (drop #{fails}). Reconnecting ...")
                cap.release()
                cap = None
                time.sleep(2)
                continue
            fails = 0

            # Only SAVE one frame per interval (the rest just drain the buffer).
            now = time.time()
            if now - last_save >= interval:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms
                path = os.path.join(SAVE_FOLDER, f"frame_{stamp}.jpg")
                if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]):
                    saved += 1
                    print(f"[{saved:>4}] saved  {os.path.basename(path)}")
                else:
                    print("[!] Failed to write a frame (check disk space / path).")
                last_save = now

    except KeyboardInterrupt:
        print(f"\n[i] Stopped by user. Total frames saved: {saved}")
        print(f"[i] Location: {SAVE_FOLDER}")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
