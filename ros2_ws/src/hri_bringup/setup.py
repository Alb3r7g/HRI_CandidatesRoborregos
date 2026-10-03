from glob import glob

from setuptools import find_packages, setup

package_name = 'hri_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Albert Constantino Salazar',
    maintainer_email='constansalazar@gmail.com',
    description='Arranque de todo el sistema HRI con un solo ros2 launch.',
    license='MIT',
)
