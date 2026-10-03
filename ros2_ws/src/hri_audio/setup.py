from setuptools import find_packages, setup

package_name = 'hri_audio'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alb3r7g12',
    maintainer_email='constansalazar@gmail.com',
    description='Nodo de captura de audio: publica hri_interfaces/AudioChunk en /audio/raw',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_capture_node = hri_audio.audio_capture_node:main',
            'vad_node = hri_audio.vad_node:main',
            'transcription_node = hri_audio.transcription_node:main',
        ],
    },
)
