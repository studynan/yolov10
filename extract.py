import cv2
import os

video_path = r'D:\yolov10\yolo10\yolov10-main\A2\A2模拟地雷第二张图.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 1308

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

output_dir = r'D:\yolov10\yolo10\yolov10-main\extract_picture'
os.makedirs(output_dir, exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read.")
        break
    save_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    success = cv2.imwrite(save_path, frame)
    if success:
        print(f"Saved frame to: {save_path}")
    else:
        print(f"Failed to save frame: {frame_count}")
    frame_count += 1

cap.release()
