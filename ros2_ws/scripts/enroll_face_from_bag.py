import sys

import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from insightface.app import FaceAnalysis
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

bag, name, start_s, end_s = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
step = 6

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
bridge = CvBridge()
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions('', ''))

enrolled = {n: np.load(f'/data/enrolled/{n}.npy') for n in ('Albert', name)}
first, index, embeddings, faces_per_frame = None, 0, [], []
while reader.has_next():
    topic, data, stamp = reader.read_next()
    if topic != '/camera/image_raw':
        continue
    first = stamp if first is None else first
    t = (stamp - first) / 1e9
    if t < start_s:
        continue
    if t > end_s:
        break
    index += 1
    if index % step:
        continue
    frame = bridge.imgmsg_to_cv2(deserialize_message(data, Image), 'bgr8')
    faces = app.get(frame)
    faces_per_frame.append(len(faces))
    if not faces:
        continue
    for face in faces:
        if float(np.dot(face.normed_embedding, enrolled['Albert'])) < 0.3:
            embeddings.append(face.normed_embedding)

print(f'cuadros analizados {len(faces_per_frame)}, caras por cuadro {sorted(set(faces_per_frame))}')
seed = enrolled[name]
embeddings = [e for e in embeddings if float(np.dot(e, seed)) >= 0.35]
for _ in range(3):
    if len(embeddings) < 5:
        break
    center = np.mean(embeddings, axis=0)
    center /= np.linalg.norm(center)
    embeddings = [e for e in embeddings if float(np.dot(e, center)) >= 0.5]
print(f'muestras de {name} tras filtrar: {len(embeddings)}')
if len(embeddings) < 5:
    sys.exit(f'Solo {len(embeddings)} caras; no se guarda nada.')
mean = np.mean(embeddings, axis=0)
mean /= np.linalg.norm(mean)
for n, ref in enrolled.items():
    sims = [float(np.dot(e, ref)) for e in embeddings]
    print(f'similitud de las muestras contra {n} actual: min {min(sims):.2f} max {max(sims):.2f}')
print(f'consistencia contra el promedio nuevo: min {min(float(np.dot(e, mean)) for e in embeddings):.2f}')
if len(sys.argv) > 5 and sys.argv[5] == '--save':
    np.save(f'/data/enrolled/{name}.npy', mean)
    print(f'Guardado /data/enrolled/{name}.npy con {len(embeddings)} muestras')
