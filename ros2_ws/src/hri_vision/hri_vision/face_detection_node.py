import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from hri_interfaces.msg import TrackedFace, TrackedFaceArray


class FaceDetectionNode(Node):

    def __init__(self):
        super().__init__('face_detection_node')

        self.declare_parameter('model_path', '/opt/weights/yolov8n-face-lindevs.pt')
        self.declare_parameter('tracker_config', 'bytetrack.yaml')

        model_path = self.get_parameter('model_path').value
        self.tracker_config = self.get_parameter('tracker_config').value

        self.bridge = CvBridge()

        self.model = None
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            self.get_logger().error(f'No se pudo cargar el modelo YOLO desde {model_path}: {exc}')

        self.tracked_faces_pub = self.create_publisher(TrackedFaceArray, '/vision/tracked_faces', 10)
        self.debug_image_pub = self.create_publisher(Image, '/vision/debug_image', 10)

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

    def image_callback(self, msg):
        if self.model is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'No se pudo convertir la imagen recibida: {exc}')
            return

        results = self.model.track(
            frame, tracker=self.tracker_config, persist=True, verbose=False)

        boxes = results[0].boxes
        tracked_array = TrackedFaceArray()
        tracked_array.header = msg.header

        # persist=True mantiene los IDs del tracker entre llamadas, pero boxes.id
        # es None cuando no hay detecciones (o aún no se ha asignado ningún track).
        if boxes is not None and boxes.id is not None:
            xywh = boxes.xywh.cpu().numpy()
            track_ids = boxes.id.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for (cx, cy, w, h), track_id, conf in zip(xywh, track_ids, confs):
                # bbox_x/bbox_y deben ser la esquina superior izquierda, pero
                # ultralytics entrega xywh con (cx, cy) como centro de la caja.
                x = cx - w / 2.0
                y = cy - h / 2.0

                face = TrackedFace()
                face.track_id = int(track_id)
                face.bbox_x = float(x)
                face.bbox_y = float(y)
                face.bbox_width = float(w)
                face.bbox_height = float(h)
                face.detection_confidence = float(conf)
                tracked_array.faces.append(face)

                pt1 = (int(x), int(y))
                pt2 = (int(x + w), int(y + h))
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(
                    frame, f'ID {int(track_id)}', (pt1[0], max(pt1[1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.tracked_faces_pub.publish(tracked_array)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().error(f'No se pudo publicar la imagen de debug: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
