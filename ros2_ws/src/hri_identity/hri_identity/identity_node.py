import glob
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from insightface.app import FaceAnalysis
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

from hri_interfaces.msg import IdentifiedFace, IdentifiedFaceArray, TrackedFaceArray

ENROLLED_DIR = '/data/enrolled'
UNKNOWN_LABEL = 'desconocido'


class IdentityNode(Node):

    def __init__(self):
        super().__init__('identity_node')

        # 0.5 de similitud coseno es más estricto que el umbral típico reportado
        # para buffalo_l/ArcFace (~0.35-0.45 en el punto FAR≈FRR): prioriza menos
        # falsos positivos a costa de rechazar coincidencias reales con ángulo o
        # iluminación distintos al enrolamiento. Es un parámetro, así que se
        # recalibra con --ros-args -p similarity_threshold:=X sin recompilar.
        self.declare_parameter('similarity_threshold', 0.5)
        self.similarity_threshold = self.get_parameter('similarity_threshold').value

        self.bridge = CvBridge()

        self.face_app = None
        try:
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            # onnxruntime cae en silencio a CPU si CUDA no está disponible o
            # el build de onnxruntime-gpu no matchea la versión de CUDA/cuDNN
            # instalada; sin este log no hay forma de notar la degradación.
            active_providers = self.face_app.models['recognition'].session.get_providers()
            self.get_logger().info(f'InsightFace providers activos: {active_providers}')
        except Exception as exc:
            self.get_logger().error(f'No se pudo cargar InsightFace (buffalo_l): {exc}')

        self.enrolled = self._load_enrolled_embeddings(ENROLLED_DIR)
        self.get_logger().info(f'{len(self.enrolled)} persona(s) enrolada(s) cargadas desde {ENROLLED_DIR}')

        self.identified_pub = self.create_publisher(IdentifiedFaceArray, '/vision/identified_faces', 10)
        self.identity_debug_pub = self.create_publisher(Image, '/vision/identity_debug_image', 10)

        self.image_sub = Subscriber(self, Image, '/camera/image_raw')
        self.tracked_faces_sub = Subscriber(self, TrackedFaceArray, '/vision/tracked_faces')
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.tracked_faces_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synced_callback)

    def _load_enrolled_embeddings(self, directory):
        enrolled = {}
        for path in sorted(glob.glob(os.path.join(directory, '*.npy'))):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                enrolled[name] = np.load(path)
            except Exception as exc:
                self.get_logger().error(f'No se pudo cargar el embedding {path}: {exc}')
        return enrolled

    def synced_callback(self, image_msg, tracked_faces_msg):
        if self.face_app is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'No se pudo convertir la imagen recibida: {exc}')
            return

        height, width = frame.shape[:2]
        identified_array = IdentifiedFaceArray()
        identified_array.header = tracked_faces_msg.header

        # se dibuja sobre una copia; el frame original nunca se anota, para que
        # los recortes de _crop_face sigan siendo pixeles de cámara puros
        debug_frame = frame.copy()

        for tracked_face in tracked_faces_msg.faces:
            identified_face = IdentifiedFace()
            identified_face.face = tracked_face
            identified_face.identity_name = UNKNOWN_LABEL
            identified_face.identity_confidence = 0.0
            identified_face.is_known = False

            # siempre se recorta de la imagen original (bgr8 real), nunca de
            # debug_image, que ya trae cajas y texto dibujados encima
            crop = self._crop_face(frame, tracked_face, width, height)
            if crop is not None:
                name, score = self._match_identity(crop)
                if score is not None:
                    identified_face.identity_confidence = float(score)
                if name is not None:
                    identified_face.identity_name = name
                    identified_face.is_known = True

            identified_array.faces.append(identified_face)
            self._draw_identity(debug_frame, identified_face)

        self.identified_pub.publish(identified_array)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.identity_debug_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().error(f'No se pudo publicar la imagen de debug de identidad: {exc}')

    def _draw_identity(self, debug_frame, identified_face):
        tracked_face = identified_face.face
        x1 = int(tracked_face.bbox_x)
        y1 = int(tracked_face.bbox_y)
        x2 = int(tracked_face.bbox_x + tracked_face.bbox_width)
        y2 = int(tracked_face.bbox_y + tracked_face.bbox_height)

        color = (0, 255, 0) if identified_face.is_known else (0, 0, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        label = f'{identified_face.identity_name} ({tracked_face.track_id})'
        cv2.putText(
            debug_frame, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _crop_face(self, frame, tracked_face, width, height):
        x = tracked_face.bbox_x
        y = tracked_face.bbox_y
        w = tracked_face.bbox_width
        h = tracked_face.bbox_height

        # bbox de YOLO ajustado muy justo a la cara recorta mentón/frente y
        # le quita contexto a InsightFace para el embedding; 30% de margen
        # da mejor calidad de detección/alineación en el recorte.
        padding = 0.3
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = int(max(x - pad_w, 0))
        y1 = int(max(y - pad_h, 0))
        x2 = int(min(x + w + pad_w, width))
        y2 = int(min(y + h + pad_h, height))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _match_identity(self, crop):
        if not self.enrolled:
            return None, None

        try:
            # DEBUG: descomentar para inspeccionar el crop enviado a InsightFace
            # cv2.imwrite('/ros2_ws/debug_crop.jpg', crop)
            faces = self.face_app.get(crop)
        except Exception as exc:
            self.get_logger().warn(f'InsightFace falló al procesar el recorte: {exc}')
            return None, None

        if not faces:
            return None, None

        embedding = faces[0].normed_embedding

        best_name = None
        best_score = -1.0
        for name, enrolled_embedding in self.enrolled.items():
            score = self._cosine_similarity(embedding, enrolled_embedding)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.similarity_threshold:
            return best_name, best_score
        return None, best_score

    @staticmethod
    def _cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(args=None):
    rclpy.init(args=args)
    node = IdentityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
