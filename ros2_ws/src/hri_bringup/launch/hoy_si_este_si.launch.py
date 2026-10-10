import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
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
OLLAMA_URL = 'http://localhost:11434'
START_OLLAMA = (f'curl -s -m 2 {OLLAMA_URL}/api/version > /dev/null '
                '&& echo "Ollama ya estaba corriendo" || exec ollama serve')
WAIT_OLLAMA = (f'for i in $(seq 60); do curl -s -m 2 {OLLAMA_URL}/api/version > /dev/null '
               '&& break; sleep 1; done; ')


def generate_launch_description():
    config = LaunchConfiguration('config')
    use_sensors = LaunchConfiguration('use_sensors')
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    def node(package, executable, condition=None, extra=None):
        return Node(package=package, executable=executable, name=executable,
                    output='screen',
                    parameters=[config, {'use_sim_time': sim_time, **(extra or {})}],
                    condition=condition)

    ollama = IfCondition(LaunchConfiguration('ollama'))
    model = LaunchConfiguration('llm_model')
    warm_up = [WAIT_OLLAMA, f'curl -s {OLLAMA_URL}/api/generate -d \'{{"model": "', model,
               '", "keep_alive": "30m"}\' > /dev/null && echo "Ollama listo: ', model,
               ' cargado en la GPU"']

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
        DeclareLaunchArgument(
            'camera_source', default_value=EnvironmentVariable('CAMERA_SOURCE', default_value=''),
            description='URL de la tablet (http://IP:8080/video) o vacio para la webcam USB.'),
        DeclareLaunchArgument(
            'ollama', default_value='true',
            description='Arrancar ollama serve si no esta corriendo y precargar el modelo.'),
        DeclareLaunchArgument(
            'llm_model', default_value='qwen2.5:7b',
            description='Modelo de Ollama que se precarga en la GPU.'),
        ExecuteProcess(cmd=['bash', '-c', START_OLLAMA], name='ollama', output='own_log',
                       condition=ollama),
        TimerAction(period=3.0, actions=[
            ExecuteProcess(cmd=['bash', '-c', warm_up], name='ollama_warm_up',
                           output='screen', condition=ollama)]),
        node('hri_camera', 'camera_node', IfCondition(use_sensors),
             {'source': LaunchConfiguration('camera_source')}),
        node('hri_audio', 'audio_capture_node', IfCondition(use_sensors)),
        *[node(p, e) for p, e in NODES],
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
             output='screen', parameters=[{'port': 8765, 'address': '0.0.0.0'}],
             condition=IfCondition(LaunchConfiguration('foxglove'))),
    ])
