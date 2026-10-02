from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
import os
from datetime import datetime

from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction

def generate_launch_description():

    return LaunchDescription([
        
        IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        [FindPackageShare('mocap4r2_vicon_driver'), '/launch/mocap4r2_vicon_driver_launch.py']
                    ),
                ),        
        
        TimerAction(period=3.0,actions=[ExecuteProcess(
            cmd=[
                'ros2', 
                'lifecycle', 
                'set', 
                '/mocap4r2_vicon_driver_node',
                'activate'],
            output='screen'
        )]),
                
        Node(
            package='mocap_pose_to_odom',             
            executable='mocap_pose_to_odom', 
            name='mocap_pose_to_odom',
            output='screen',
        )
    
    ])