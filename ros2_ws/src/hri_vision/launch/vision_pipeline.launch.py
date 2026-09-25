from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # use_camera:=false omite camera_node para alimentar el pipeline desde un
    # ros2 bag que publique /camera/image_raw.
    use_camera = LaunchConfiguration('use_camera')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_camera',
            default_value='true',
            description='Lanzar camera_node. Usar false al reproducir un bag.',
        ),
        Node(
            package='hri_camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            condition=IfCondition(use_camera),
        ),
        Node(
            package='hri_vision',
            executable='face_detection_node',
            name='face_detection_node',
            output='screen',
        ),
        Node(
            package='hri_identity',
            executable='identity_node',
            name='identity_node',
            output='screen',
        ),
    ])
