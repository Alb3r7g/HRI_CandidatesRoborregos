FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Configuración no interactiva
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

# Configuro locale UTF-8
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Habilito repo universe
RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository universe

# Registro repositorio de ROS 2
RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') \
    && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.noble_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb

# Instalo ROS 2 y herramientas
RUN apt-get update && apt-get install -y \
    ros-jazzy-desktop \
    ros-dev-tools \
    ros-jazzy-foxglove-bridge

# Instalo ultralytics (YOLO) vía pip, fuera del index de apt/rosdep.
# --ignore-installed evita que pip intente desinstalar paquetes que ya trae
# la imagen base vía apt/dpkg (como typing_extensions), que no tienen RECORD de pip.
# Fijo numpy<2: cv_bridge (compilado por apt) está enlazado contra la ABI de
# NumPy 1.x, y un numpy 2.x instalado por pip en dist-packages lo rompe en tiempo
# de ejecución (ImportError al cargar cv_bridge_boost).
RUN apt-get update && apt-get install -y python3-pip wget \
    && pip3 install --break-system-packages --ignore-installed ultralytics "numpy<2" lap \
       insightface onnxruntime-gpu

# insightface declara 'onnxruntime' (CPU-only) como dependencia genérica, y pip
# lo reinstala encima de onnxruntime-gpu — ambos paquetes comparten el mismo
# módulo de import 'onnxruntime', así que el último en escribir archivos gana.
# Si queda el paquete CPU, CUDAExecutionProvider desaparece de
# get_available_providers() en silencio (sin error, solo fallback a CPU).
# Lo quito y reinstalo onnxruntime-gpu limpio para que sus binarios no queden pisados.
RUN pip3 uninstall -y --break-system-packages onnxruntime \
    && pip3 install --break-system-packages --force-reinstall --no-deps onnxruntime-gpu

# Descargo el peso de YOLOv8-face para hri_vision (detección + tracking de rostros)
RUN mkdir -p /opt/weights \
    && wget -O /opt/weights/yolov8n-face-lindevs.pt \
       https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt

# Fuerzo la descarga del modelo buffalo_l de InsightFace durante el build (no en
# el primer arranque del nodo). Sin GPU en tiempo de build, onnxruntime cae a
# CPUExecutionProvider para este prepare() — solo importa que el modelo quede
# cacheado en ~/.insightface/, la inferencia real en runtime sí usa GPU si está.
RUN python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"

# Captura de audio desde el host. El contenedor no recibe /dev/snd: se conecta al
# servidor PulseAudio que expone PipeWire, cuyo socket monta docker-compose.
# libasound2-plugins aporta el plugin ALSA->PulseAudio que necesita PortAudio
# (sounddevice) para ver un dispositivo de entrada; sin el, query_devices() no
# encuentra ninguno porque ALSA no tiene hardware que enumerar.
RUN apt-get update && apt-get install -y \
    libpulse0 \
    pulseaudio-utils \
    libportaudio2 \
    libasound2-plugins \
    && pip3 install --break-system-packages sounddevice

# Enruta el dispositivo ALSA por defecto hacia PulseAudio.
RUN printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf

# Cargo ROS 2 automáticamente
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /ros2_ws