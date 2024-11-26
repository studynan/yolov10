import rospy
from std_msgs.msg import String
import serial

# 配置 USB 串口设备
serial_port = "/dev/ttyUSB0"  # 根据实际设备修改
baud_rate = 115200  # 串口波特率

# 初始化串口
ser = serial.Serial(serial_port, baud_rate)

# 初始化 ROS 节点
rospy.init_node('serial_listener', anonymous=True)

# 创建发布者
pub = rospy.Publisher('object_detection_info', String, queue_size=10)

def read_serial():
    while not rospy.is_shutdown():
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            rospy.loginfo(f"接收到数据: {data}")
            pub.publish(data)

if __name__ == "__main__":
    read_serial()

