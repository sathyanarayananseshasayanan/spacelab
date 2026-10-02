from setuptools import setup
import os 
from glob import glob 


package_name = 'mpc_hardware'


setup(
    name=package_name,
    version='0.0.0',
    packages=['mpc_controller'],
    # install_requires=[
    #     'setuptools',
    #     'quadprog', 
    #     'cvxopt', 
    #     'numpy'
    # ],
    data_files=[
        ('share/ament_index/resource_index/packages', []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch*')))      
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bibeto',
    maintainer_email='bibeto@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #'mpc_docking_real_sense = mpc_controller.mpc_docking_real_sense:main',
            'mpc_circle = mpc_controller.mpc_circle:main',
            'pwm_mpc_publisher  = mpc_controller.pwm_mpc_publisher:main',
        ],
    },
)
