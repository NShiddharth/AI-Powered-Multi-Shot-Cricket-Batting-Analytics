import cv2
import os
import numpy as np

class VideoPreprocessor:
    """
    VideoPreprocessor module for loading, processing, normalizing,
    and extracting frames from input cricket videos.
    """
    def __init__(self, video_path, target_resolution=None, normalize_brightness=False, crop_roi=None, save_frames_dir="frames"):
        self.video_path = video_path
        self.target_resolution = target_resolution
        self.normalize_brightness = normalize_brightness
        self.crop_roi = crop_roi
        self.save_frames_dir = save_frames_dir

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {self.video_path}")

        # Extract video properties
        self.raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Target dimensions after crop/resize
        if self.target_resolution:
            self.width, self.height = self.target_resolution
        elif self.crop_roi:
            _, _, self.width, self.height = self.crop_roi
        else:
            self.width, self.height = self.raw_width, self.raw_height

        if self.save_frames_dir:
            os.makedirs(self.save_frames_dir, exist_ok=True)

    def _apply_brightness_normalization(self, frame):
        """
        Equalizes lightness using CLAHE in the LAB color space.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame, frame_idx):
        """
        Applies optional cropping, resolution resizing, brightness normalization,
        and saves frame image to disk.
        """
        processed = frame.copy()

        # 1. Optional Cropping
        if self.crop_roi:
            x, y, w, h = self.crop_roi
            processed = processed[y:y+h, x:x+w]

        # 2. Resolution Normalization
        if self.target_resolution:
            processed = cv2.resize(processed, self.target_resolution, interpolation=cv2.INTER_AREA)

        # 3. Brightness Normalization
        if self.normalize_brightness:
            processed = self._apply_brightness_normalization(processed)

        # 4. Save Extracted Frame to Disk (frame_000001.jpg)
        if self.save_frames_dir:
            frame_path = os.path.join(self.save_frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, processed)

        return processed

    def read_frames(self):
        """
        Generator yielding (frame_idx, processed_frame).
        """
        frame_idx = 1
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame, frame_idx)
            yield frame_idx, processed_frame
            frame_idx += 1

    def release(self):
        """
        Releases the VideoCapture object.
        """
        if self.cap:
            self.cap.release()
