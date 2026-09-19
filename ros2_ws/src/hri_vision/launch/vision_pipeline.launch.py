from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hri_camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
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
