import cv2
import os
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

def extract_faces_from_video(video_path, output_dir, frames_to_extract=10):
    os.makedirs(output_dir, exist_ok=True)
    # Safe CUDA/device detection for OpenCV builds without cv2.cuda
    cuda_available = hasattr(cv2, 'cuda') and getattr(cv2.cuda, 'getCudaEnabledDeviceCount', lambda: 0)() > 0
    device = 'cuda' if cuda_available else 'cpu'
    mtcnn = MTCNN(keep_all=False, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(frame_count // frames_to_extract, 1)
    frame_num = 0
    saved = 0

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i in tqdm(range(frame_count), desc=f"Extracting {base_name}"):
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            # Convert BGR (OpenCV) -> RGB for facenet-pytorch
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                frame_rgb = frame

            boxes, _ = mtcnn.detect(frame_rgb)
            if boxes is not None:
                boxes = np.atleast_2d(boxes)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    h, w = frame.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    face = frame[y1:y2, x1:x2]
                    if face.size != 0:
                        filename = os.path.join(output_dir, f"{base_name}_{saved}.jpg")
                        cv2.imwrite(filename, face)
                        saved += 1
        frame_num += 1
    cap.release()

def prepare_dataset(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for video in os.listdir(video_folder):
        path = os.path.join(video_folder, video)
        if path.lower().endswith(('.mp4', '.avi', '.mov')):
            output_dir = os.path.join(output_folder, os.path.splitext(video)[0])
            extract_faces_from_video(path, output_dir)

if __name__ == "__main__":
    prepare_dataset("videos", "faces")
