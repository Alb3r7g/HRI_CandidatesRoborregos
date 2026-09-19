# Contexto del proyecto — Candidates HRI 2026

Este es el reto de HRI/Visión de Candidates Avanzados 2026 (RoBorregos). Antes de generar
código, lee este archivo completo. Si algo que se te pide contradice lo que dice aquí, dilo
explícitamente en vez de improvisar una solución distinta.

## Regla de oro del proyecto (no negociable)

La identidad es un permiso, no una llave de ruteo. El acceso al documento privado de una
persona (RAG) NUNCA se ejecuta si la confianza fusionada no supera el umbral de acceso. Si la
confianza cae a media sesión, el acceso se revoca. Política de decisión de tres estados:
responder / preguntar / negar — nunca un simple sí/no.

## Reglas de Git (del reglamento oficial, no negociables)

- Commits semánticos: una sola línea, prefijo `feat:` / `fix:` / `refactor:` / `docs:` / `chore:`.
- Nunca commits gigantes "todo en uno"; una implementación = un commit lógico.
- `main` se usa ÚNICAMENTE para el entregable final. Todo el trabajo diario va en ramas
  `feat/nombre-descriptivo`.
- No generes mensajes de commit genéricos ("update", "fix stuff", "wip"). Cada mensaje debe
  describir el cambio real.
- NUNCA ejecutes `git push` por tu cuenta, bajo ninguna circunstancia. Puedes preparar el
  staging (`git add`) y proponer el mensaje de commit, o incluso crear el commit localmente,
  pero el candidato hace la última auditoría del diff y decide cuándo y qué se sube al remoto.
  Si necesitas subir algo, pregúntalo explícitamente y espera confirmación.

## Arquitectura de la Semana 1 (ya decidida — no reinventar ni proponer alternativas sin avisar)

Pipeline de tres nodos:

- `camera_node` → publica `/camera/image_raw` (usa `sensor_msgs/Image` estándar, no interfaz propia)
- `face_detection_node` → detecta caras + tracking con IDs estables → publica
  `/vision/tracked_faces` (tipo `hri_interfaces/TrackedFaceArray`)
- `face_identity_node` → compara contra base de embeddings enrolados → publica
  `/vision/identified_faces` (tipo `hri_interfaces/IdentifiedFaceArray`)

## Paquetes del workspace

- `hri_interfaces` (ament_cmake) — SOLO define mensajes (.msg). No agregues lógica aquí.
- `hri_camera` (ament_python) — nodo de cámara.
- `hri_vision` (ament_python) — detección de rostros + tracking.
- `hri_identity` (ament_python) — comparación de identidad (aún sin RAG, eso es Semana 2).

## Interfaces ya definidas en hri_interfaces (campos pueden ajustarse, avisa antes de cambiarlos)

- `TrackedFace`: track_id (uint32), bbox_x/y/width/height (float32), detection_confidence (float32)
- `TrackedFaceArray`: header (std_msgs/Header) + TrackedFace[] faces
- `IdentifiedFace`: face (TrackedFace anidado), identity_name (string), identity_confidence
  (float32), is_known (bool)
- `IdentifiedFaceArray`: header + IdentifiedFace[] faces

## Entorno de desarrollo

- Todo corre dentro de Docker. Imagen base: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` +
  ROS 2 Jazzy instalado vía apt (paquete `ros2-apt-source`, NO el método viejo de gpg key manual).
- GPU NVIDIA disponible (RTX 4060) vía NVIDIA Container Toolkit — usa aceleración GPU cuando
  la librería lo soporte (onnxruntime-gpu, CUDA), pero no asumas que siempre está disponible;
  agrega fallback a CPU cuando sea razonable.
- El código fuente vive en el host (Fedora) y se monta como volumen — nunca asumas que puedes
  escribir archivos fuera de `ros2_ws/src/` dentro del contenedor de forma persistente.

## Preferencias de librerías del candidato (decisiones ya tomadas, no proponer alternativas)

- Transcripción de voz (Semana 2): usar `faster-whisper`. Es la librería que usa el equipo
  RoBorregos, así que se prioriza sobre otras opciones (whisper.cpp, openai-whisper, etc.)
  aunque el PDF mencione varias como igualmente válidas.

## Al generar código

- Comenta las decisiones NO obvias (umbrales, por qué cierto approach) — esto alimenta
  directamente el reporte E2 del reto.
- NO implementes lógica de fusión de señales, umbrales de identidad, ni la política de tres
  estados sin preguntar primero y explicar las opciones — esas decisiones el candidato necesita
  entenderlas a fondo para la entrevista final.
- El boilerplate mecánico de ROS 2 (estructura de paquete, package.xml, CMakeLists.txt,
  esqueleto publisher/subscriber) sí lo puedes generar directo sin preguntar.
- Si una librería sugerida en el reglamento (YOLOv8-face, InsightFace, ByteTrack, etc.) tiene
  una alternativa que consideres mejor, dilo y explica el trade-off — no la cambies en silencio.