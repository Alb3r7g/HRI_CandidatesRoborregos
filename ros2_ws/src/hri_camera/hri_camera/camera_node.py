import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('device_id', 0)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera_frame')

        device_id = self.get_parameter('device_id').value
        fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value

        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        self.capture = cv2.VideoCapture(device_id)
        if not self.capture.isOpened():
            # No lanzamos excepción: el nodo debe seguir vivo (para que 'ros2 node list'
            # y diagnósticos lo vean) en vez de morir en silencio si la cámara no está
            # conectada al arrancar. timer_callback revisa isOpened() en cada tick.
            self.get_logger().error(f'No se pudo abrir la cámara (device_id={device_id})')

        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    def timer_callback(self):
        if not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if not ret:
            self.get_logger().warn('Frame no recibido de la cámara')
            return

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
