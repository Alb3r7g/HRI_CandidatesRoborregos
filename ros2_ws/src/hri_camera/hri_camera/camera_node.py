import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('device_id', 0)
        self.declare_parameter('source', os.environ.get('CAMERA_SOURCE', ''))
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('reconnect_after_s', 2.0)

        source = self.get_parameter('source').value.strip()
        self.source = source or self.get_parameter('device_id').value
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value
        self.reconnect_after_s = self.get_parameter('reconnect_after_s').value
        self.last_frame_time = time.monotonic()

        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        self.capture = self.open_capture()
        if self.capture.isOpened():
            self.get_logger().info(f'Cámara abierta: {self.source}')
        else:
            # No lanzamos excepción: el nodo debe seguir vivo (para que 'ros2 node list'
            # y diagnósticos lo vean) en vez de morir en silencio si la cámara no está
            # conectada al arrancar. timer_callback revisa isOpened() en cada tick.
            self.get_logger().error(f'No se pudo abrir la cámara ({self.source})')

        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    def open_capture(self):
        if isinstance(self.source, str):
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        return cv2.VideoCapture(self.source)

    def reconnect_if_stalled(self):
        if time.monotonic() - self.last_frame_time < self.reconnect_after_s:
            return
        self.get_logger().warn(f'Sin cuadros de {self.source}; reabriendo la cámara')
        self.capture.release()
        self.capture = self.open_capture()
        self.last_frame_time = time.monotonic()

    def timer_callback(self):
        if not self.capture.isOpened():
            self.reconnect_if_stalled()
            return

        ret, frame = self.capture.read()
        if not ret:
            self.get_logger().warn('Frame no recibido de la cámara')
            self.reconnect_if_stalled()
            return
        self.last_frame_time = time.monotonic()

        # bgr8 porque OpenCV entrega los frames en orden de canales BGR, no RGB.
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
