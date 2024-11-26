import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLOv10
import os
import time  # 导入 time 模块

# YOLOv10 模型加载路径
model = YOLOv10('/home/uav-qkyang/cave_explore_competition_uav-simulator/src/yolo/src/1113v10n.pt')

# 初始化跟踪器字典
trackers = {}
saved_objects = {}  # 保存物体 id和类别
category_counters = {}  # 用于每个类别的唯一ID生成计数器
save_dir = '/media/uav-qkyang/KINGSTON'  # 设置截图保存路径

# 设置物体离开视野后的最大等待时间，单位：秒
max_unseen_time = 3 # 如果物体3秒未被跟踪，认为它离开了视野

# 主函数
def main():
    # 配置 RealSense D435i 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流
    pipeline.start(config)

    try:
        while True:  # 无限循环，直到按下 'q' 键
            # 获取图像帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 将帧转换为 NumPy 数组
            frame = np.asanyarray(color_frame.get_data())

            # 使用 YOLOv10 检测物体
            results = model(frame)
            result = results[0]
            frame = result.plot()

            a = 1  # 默认识别到物体

            current_time = time.time()  # 当前时间戳

            if len(result.boxes) == 0:  # 如果没有检测到物体
                a = 0  # 未识别到物体
                # 检查所有跟踪器，删除已经离开视野的物体
                for track_id, tracker in list(trackers.items()):
                    success, bbox = tracker.update(frame)
                    if not success:
                        # 如果跟踪失败，认为物体离开了视野
                        print(f"物体 {track_id} 离开视野")
                        saved_objects[track_id]["last_seen"] = current_time  # 记录离开时间
                        saved_objects[track_id]["screenshot_taken"] = False  # 标记为未截图
            else:
                # 获取所有检测框
                detections = []
                for detection in result.boxes:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])  # 获取检测框坐标
                    w = x2 - x1
                    h = y2 - y1
                    class_id = int(detection.cls[0])  # 获取物体类别ID
                    confidence = detection.conf[0]  # 获取物体的置信度
                    detections.append([x1, y1, x2, y2, class_id, confidence])  # 检测框坐标和类别ID

                # 为每个检测框分配唯一的物体 ID，ID 只基于类别
                for detection in detections:
                    x1, y1, x2, y2, class_id, confidence = detection

                    # 生成唯一的物体 ID：使用类别来生成 ID
                    track_id = f"ID_{class_id}"

                    # 检查该物体是否已经存在
                    if track_id not in saved_objects:
                        # 如果是新物体，生成 ID 并保存
                        saved_objects[track_id] = {
                            "class_id": class_id,
                            "confidence": confidence,
                            "count": 0,  # 物体的保存计数器
                            "screenshot_taken": False,  # 标记物体是否已截图
                            "last_seen": current_time,  # 记录物体上次出现的时间
                        }
                    # 更新物体信息
                    saved_objects[track_id]["count"] += 1
                    saved_objects[track_id]["confidence"] = confidence

                    # 更新跟踪器
                    if track_id not in trackers:
                        tracker = cv2.TrackerCSRT_create()  # 使用 CSRT 跟踪器
                        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))  # 初始化跟踪器
                        trackers[track_id] = tracker  # 保存到字典中

                    # 更新跟踪器
                    tracker = trackers[track_id]
                    success, bbox = tracker.update(frame)  # 更新跟踪器的位置
                    if success:
                        x1, y1, w, h = [int(v) for v in bbox]
                        x2, y2 = x1 + w, y1 + h

                        # 绘制物体的边界框
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 只有当置信度达到0.6及以上时，才截图保存
                        if confidence >= 0.6 and not saved_objects[track_id]["screenshot_taken"]:
                            # 使用 ID 和时间戳作为文件名
                            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))  # 格式化时间
                            screenshot_path = os.path.join(save_dir, f"{track_id}_{timestamp}.jpg")
                            cv2.imwrite(screenshot_path, frame)  # 保存截图
                            saved_objects[track_id]["screenshot_taken"] = True  # 标记为已截图
                        # 更新物体的最后出现时间
                        saved_objects[track_id]["last_seen"] = current_time

                    # 检查物体是否长时间未出现，若是，则重新保存
                    if current_time - saved_objects[track_id]["last_seen"] > max_unseen_time:
                        # 如果物体离开超过 3 秒，重新保存
                        print(f"物体 {track_id} 重新出现，重新保存")
                        # 使用 ID 和时间戳作为文件名
                        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))  # 格式化时间
                        screenshot_path = os.path.join(save_dir, f"{track_id}_reappeared_{timestamp}.jpg")
                        cv2.imwrite(screenshot_path, frame)  # 保存截图
                        saved_objects[track_id]["screenshot_taken"] = False  # 重新标记为未截图

            # 显示图像（可选）
            cv2.imshow('Frame', frame)

            # 按键 q 退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理和释放资源
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
