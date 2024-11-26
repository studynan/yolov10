import rospy
from std_msgs.msg import String
import serial

# 配置 USB 串口设备
serial_port = "/dev/ttyUSB0"  # 根据实际设备修改
baud_rate = 115200  # 串口波特率

# 初始化串口
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    rospy.loginfo(f"串口 {serial_port} 已成功打开，波特率: {baud_rate}")
except serial.SerialException as e:
    rospy.logerr(f"无法打开串口 {serial_port}: {e}")
    exit(1)

# 初始化 ROS 节点
rospy.init_node('serial_publisher', anonymous=True)

# 协议定义
FRAME_HEADER = b'\xAA\xBB'
FRAME_TAIL = b'\xCC\xDD'

def calculate_checksum(data_bytes):
    """计算校验和，返回一个字节值"""
    return sum(data_bytes) & 0xFF  # 取最低字节

def pack_data(message_str):
    """打包数据成带帧头、校验位、帧尾的格式"""
    data_bytes = message_str.encode('utf-8')
    checksum = calculate_checksum(data_bytes)
    packed_frame = FRAME_HEADER + data_bytes + bytes([checksum]) + FRAME_TAIL
    return packed_frame

def callback(msg):
    # 确保串口已打开再发送数据
    if ser.is_open:
        try:
            packed_data = pack_data(msg.data)
            ser.write(packed_data)
            rospy.loginfo(f"发送消息: {msg.data} (打包后: {packed_data})")
        except serial.SerialException as e:
            rospy.logerr(f"串口发送数据失败: {e}")
    else:
        rospy.logwarn("串口未打开，无法发送消息")

def listener():
    rospy.Subscriber('object_detection_info', String, callback)
    rospy.spin()  # 保持节点运行

if __name__ == "__main__":
    try:
        listener()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS 节点被中断")
    finally:
        # 关闭串口连接
        if ser.is_open:
            ser.close()
            rospy.loginfo("串口已关闭")
