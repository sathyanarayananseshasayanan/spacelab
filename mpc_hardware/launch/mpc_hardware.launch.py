#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def typed(name, value_type):
    """Convert a string launch argument to a correctly typed ROS parameter."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("mpc_frequency", default_value="10"),
        DeclareLaunchArgument("horizon", default_value="3"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("dock_odom_topic", default_value="/odom1_vison"),
        DeclareLaunchArgument("dock_valid_topic", default_value="/odom1_valid"),
        DeclareLaunchArgument("dock_fresh_topic", default_value="/odom1_fresh"),
        DeclareLaunchArgument("cmd_topic", default_value="/slider_1/thrust_cmd"),
        DeclareLaunchArgument("robot_odom_timeout", default_value="0.30"),
        DeclareLaunchArgument("dock_odom_timeout", default_value="0.30"),
        DeclareLaunchArgument("dock_valid_timeout", default_value="0.30"),
        DeclareLaunchArgument("dock_fresh_timeout", default_value="0.30"),
        DeclareLaunchArgument("require_dock_valid", default_value="true"),
        DeclareLaunchArgument("require_fresh_near_dock", default_value="true"),
        DeclareLaunchArgument("fresh_required_distance", default_value="0.50"),
        DeclareLaunchArgument("max_dock_linear_speed", default_value="0.75"),
        DeclareLaunchArgument("max_dock_yaw_rate", default_value="1.50"),
        DeclareLaunchArgument("allow_final_creep", default_value="true"),
        DeclareLaunchArgument("auto_final_creep", default_value="false"),
        DeclareLaunchArgument("final_creep_entry_distance", default_value="0.30"),
        DeclareLaunchArgument("final_creep_lateral_tolerance", default_value="0.03"),
        DeclareLaunchArgument("final_creep_yaw_tolerance_deg", default_value="3.0"),
        DeclareLaunchArgument("final_creep_entry_speed", default_value="0.08"),
        DeclareLaunchArgument("final_creep_abort_lateral", default_value="0.08"),
        DeclareLaunchArgument("final_creep_abort_yaw_deg", default_value="8.0"),
        DeclareLaunchArgument("final_creep_max_force", default_value="0.20"),
        DeclareLaunchArgument("final_creep_max_travel", default_value="0.20"),
        DeclareLaunchArgument("final_creep_max_duration", default_value="5.0"),
        DeclareLaunchArgument("contact_topic", default_value="/docking_contact"),
        DeclareLaunchArgument("require_contact_sensor", default_value="false"),
        DeclareLaunchArgument("pwm_frequency", default_value="3.0"),
        DeclareLaunchArgument("pwm_resolution", default_value="6"),
        DeclareLaunchArgument("use_integral_xy", default_value="true"),
        DeclareLaunchArgument("integral_gain_xy", default_value="0.5"),
        DeclareLaunchArgument("integral_limit_xy", default_value="0.2"),
        DeclareLaunchArgument("integral_leak_xy", default_value="0.00"),
        DeclareLaunchArgument("integral_deadband_xy", default_value="0.00"),
        DeclareLaunchArgument("use_cbf", default_value="false"),
        DeclareLaunchArgument("cbf_k1", default_value="2.0"),
        DeclareLaunchArgument("cbf_k0", default_value="1.0"),
        DeclareLaunchArgument("cbf_goal_shift", default_value="0.22"),
        DeclareLaunchArgument("orbits", default_value="1"),
        DeclareLaunchArgument("orbit_margin", default_value="0.10"),
        DeclareLaunchArgument("orbit_direction", default_value="1"),
        DeclareLaunchArgument("use_soft_cbf", default_value="false"),
        DeclareLaunchArgument("cbf_slack_weight", default_value="500.0"),
        DeclareLaunchArgument("safe_distance", default_value="0.58"),
        DeclareLaunchArgument("dock_safe_distance", default_value="0.0"),
    ]

    mpc_node = Node(
        package='mpc_hardware',
        executable='mpc_circle',  # your rewritten MPC executable name
        name='mpc_circle',
        output='screen',
        parameters=[{
            'frequency': LaunchConfiguration('mpc_frequency'),
            'horizon': LaunchConfiguration('horizon'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'dock_odom_topic': LaunchConfiguration('dock_odom_topic'),
            'cmd_topic': LaunchConfiguration('cmd_topic'),
            "deadband_xy": 0.00,
            "deadband_vxy": 0.00,
            "deadband_r": 0.00,
            "deadband_yaw": 0.0*3.14159/180.0,
            "terminal_scale": 4.0,
            "use_integral_xy": LaunchConfiguration('use_integral_xy'),
            "integral_gain_xy": LaunchConfiguration('integral_gain_xy'),
            "integral_limit_xy": LaunchConfiguration('integral_limit_xy'),
            "integral_leak_xy": LaunchConfiguration('integral_leak_xy'),
            "integral_deadband_xy": LaunchConfiguration('integral_deadband_xy'),
            "use_cbf": LaunchConfiguration('use_cbf'),
            "cbf_k1": LaunchConfiguration('cbf_k1'),
            "cbf_k0": LaunchConfiguration('cbf_k0'),
            "cbf_goal_shift": LaunchConfiguration('cbf_goal_shift'),
            "use_soft_cbf": LaunchConfiguration('use_soft_cbf'),
            "cbf_slack_weight": LaunchConfiguration('cbf_slack_weight'),
            "safe_distance": LaunchConfiguration('safe_distance'),
            "dock_safe_distance": LaunchConfiguration('dock_safe_distance'),
            }],
        )

    pwm_node = Node(
        package="mpc_hardware",
        executable="pwm_mpc_publisher",
        name="pwm_mpc_publisher",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "frequency": typed("pwm_frequency", float),
            "resolution": typed("pwm_resolution", int),
            "on_threshold_ratio": 0.45,
            "off_threshold_ratio": 0.10,
            "min_switch_time": 0.05,
            "cmd_alpha": 0.25,
            "active_low": False,
        }],
    )

    return LaunchDescription(arguments + [mpc_node, pwm_node])
