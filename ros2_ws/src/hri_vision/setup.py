from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hri_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alb3r7g12',
    maintainer_email='constansalazar@gmail.com',
    description='Detección y tracking de rostros: publica hri_interfaces/TrackedFaceArray en /vision/tracked_faces',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_detection_node = hri_vision.face_detection_node:main',
        ],
    },
)
