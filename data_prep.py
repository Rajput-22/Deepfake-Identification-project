import cv2
import os
from tqdm import tqdm
from facenet_pytorch import MTCNN

def extract_faces_from_video(video_path, output_dir, frames_to_extract=10):
    os.makedirs(output_dir, exist_ok=True)
    mtcnn = MTCNN(keep_all=False, device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(frame_count // frames_to_extract, 1)
    frame_num = 0
    saved = 0

    for i in tqdm(range(frame_count), desc=f"Extracting {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]
                    if face.size != 0:
                        filename = os.path.join(output_dir, f"{os.path.basename(video_path)}_{saved}.jpg")
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