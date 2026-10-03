import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

NODES = [
    ('hri_vision', 'face_detection_node'),
    ('hri_identity', 'identity_node'),
    ('hri_audio', 'vad_node'),
    ('hri_audio', 'transcription_node'),
    ('hri_identity', 'voice_identity_node'),
    ('hri_rag', 'rag_service_node'),
    ('hri_fusion', 'fusion_node'),
]
SENSORS = [
    ('hri_camera', 'camera_node'),
    ('hri_audio', 'audio_capture_node'),
]


def generate_launch_description():
    config = LaunchConfiguration('config')
    use_sensors = LaunchConfiguration('use_sensors')
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    def node(package, executable, condition=None):
        return Node(package=package, executable=executable, name=executable,
                    output='screen', parameters=[config, {'use_sim_time': sim_time}],
                    condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=os.path.join(get_package_share_directory('hri_bringup'),
                                       'config', 'hri.yaml'),
            description='Archivo YAML con los umbrales de todos los nodos.'),
        DeclareLaunchArgument(
            'use_sensors', default_value='true',
            description='Lanzar camara y microfono. false al reproducir un bag.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true al reproducir un bag con ros2 bag play --clock.'),
        DeclareLaunchArgument(
            'foxglove', default_value='true',
            description='Lanzar foxglove_bridge en el puerto 8765 para ver el overlay.'),
        *[node(p, e, IfCondition(use_sensors)) for p, e in SENSORS],
        *[node(p, e) for p, e in NODES],
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
             output='screen', parameters=[{'port': 8765, 'address': '0.0.0.0'}],
             condition=IfCondition(LaunchConfiguration('foxglove'))),
    ])
