import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLOv10
import os
import time
import rospy
from std_msgs.msg import String  # 导入 ROS 字符串消息类型

model = YOLOv10('/home/uav-qkyang/cave_explore_competition_uav-simulator/src/yolo/src/1106v10n.pt')


# 主函数
def main():
    # 初始化 ROS 节点
    rospy.init_node('yolo_detector', anonymous=True)
    
    pub = rospy.Publisher('object_detection_info', String, queue_size=10)  # 创建发布者

    # 配置 RealSense D435i 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流
    pipeline.start(config)

    # 获取屏幕中心坐标
    screen_center_x = 320  # 假设640的宽度一半
    screen_center_y = 240  # 假设480的高度一半

    b = 0  # 标记图像是否已保存
    try:
        while not rospy.is_shutdown():  # 检查 ROS 节点是否被关闭
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
            delta_x, delta_y = 999, 999  # 默认坐标差
            if len(result.boxes) == 0:  # 如果没有检测到物体
                a = 0  # 未识别到物体
            else:
                # 遍历检测结果
                for detection in result.boxes:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])  # 获取检测框坐标
                    w = x2 - x1
                    h = y2 - y1

                    # 计算物体的中心点
                    object_center_x = x1 + w // 2
                    object_center_y = y1 + h // 2

                    # 计算物体中心与屏幕中心的相对坐标差
                    delta_x = object_center_x - screen_center_x
                    delta_y = object_center_y - screen_center_y

                    # 绘制十字光标在物体中心
                    cv2.line(frame, (object_center_x - 10, object_center_y), (object_center_x + 10, object_center_y),
                             (0, 255, 0), 2)
                    cv2.line(frame, (object_center_x, object_center_y - 10), (object_center_x, object_center_y + 10),
                             (0, 255, 0), 2)

                    # 当物体中心接近屏幕中心时，执行截图
                    if abs(delta_x) < 10 and abs(delta_y) < 10 and b == 0:  # 如果图像未保存
                        # 使用当前时间戳作为文件名
                        timestamp = int(time.time())  # 当前时间戳
                        unique_filename = f"{timestamp}.jpg"  # 生成随机文件名
                        save_path = f'/home/uav-qkyang/cave_explore_competition_uav-simulator/src/yolo/src/pic/picture{unique_filename}'  # 保存路径
                        cv2.imwrite(save_path, frame)
                        print(f"截图已保存到 {save_path}")
                        b = 1  # 标记图像已保存
                        message = f"a={a},x={delta_x},y={delta_y},z={b}"
                        pub.publish(message)  # 发布信息
                        print(f"发布信息: {message}")

                        rospy.sleep(8)  # 等待一秒后再次检测
                        b = 0  # 标记图像未保存
                        message = f"a={a},x={delta_x},y={delta_y},z={b}"
                        pub.publish(message)  # 发布信息
                        print(f"发布信息: {message}")
                        



            # 发布检测信息到 ROS 话题
            message = f"a={a},x={delta_x},y={delta_y},z={b}"
            pub.publish(message)  # 发布信息
            print(f"发布信息: {message}")

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
