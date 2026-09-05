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
    ros-dev-tools

# Cargo ROS 2 automáticamente
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /ros2_ws