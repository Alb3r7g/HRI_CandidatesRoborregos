from setuptools import find_packages, setup

package_name = 'hri_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Albert Constantino Salazar',
    maintainer_email='constansalazar@gmail.com',
    description='Fusion de cara, voz y boca con politica de acceso de tres estados.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fusion_node = hri_fusion.fusion_node:main',
        ],
    },
)
