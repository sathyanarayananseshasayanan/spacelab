#!/usr/bin/env bash

# Stop the ROS 2 daemon to refresh discovery
#ros2 daemon stop

export ROS_DOMAIN_ID=0
#ros2 daemon start

WS=~/slider_ws
BAG_DIR="$WS/rosbags"
# Modify BAG_NAME to include a timestamp for unique file names
BAG_NAME="pid_vicon_test_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BAG_DIR"
rm -rf "$BAG_DIR/$BAG_NAME"

cd "$WS"
colcon build
# Build if needed: colcon build --packages-select mocap_pose_to_odom pid_hardware
source "$WS/install/setup.bash"

cleanup() {
  echo "Stopping thrust..."
  ros2 topic pub --once -w 0 /eight_thrust_pulse std_msgs/msg/UInt8MultiArray \
  "{data: [0,0,0,0,0,0,0,0]}" 2>/dev/null || true

  echo "Stopping ROS processes..."
  kill ${PID_BAG:-} ${PID_PID:-} ${PID_ODOM:-} 2>/dev/null || true

  # give bag a moment to flush
  wait ${PID_BAG:-} 2>/dev/null || true

  echo "Cleanup complete."
}
trap cleanup INT TERM EXIT

echo "Stopping thrust..."
ros2 topic pub --once -w 0 /eight_thrust_pulse std_msgs/msg/UInt8MultiArray \
  "{data: [0,0,0,0,0,0,0,0]}" 2>/dev/null || true


echo "Starting Vicon driver and mocap pose to odom bridge..."
ros2 launch mocap_pose_to_odom get_odom_vicon.launch.py &
PID_ODOM=$!
sleep 5

echo "Starting PID hardware controller..."
ros2 launch mpc_hardware mpc_hardware.launch.py \
  target_x:=0.0 \
  target_y:=0.0 \
  target_tau:=0.0 \
  dock_odom_topic:=/odom1_vison &
PID_PID=$!
sleep 2

BAG_TOPICS=(
  /rigid_bodies
  /odom
  /odom1_vision
  /docking_station/pose
  /docking_station/debug/reproj_error_px
  /docking_station/debug/fit_error_m
  /docking_station/debug/num_points_used
  /docking_station/debug/distance_m
  /docking_station/debug/method
  /docking_station/debug/pose_age_sec
  /slider_1/thrust_cmd
  /eight_thrust_pulse
  /tf
  /tf_static
)

if [[ "$RECORD_DEBUG_IMAGE" == "1" ]]; then
  BAG_TOPICS+=(/docking_station/debug_image)
fi

echo "Starting rosbag recording: $BAG_DIR/$BAG_NAME"
ros2 bag record \
  -o "$BAG_DIR/$BAG_NAME" \
  "${BAG_TOPICS[@]}" &
PID_BAG=$!

echo "All processes started."
echo "PIDs: Odom/Vicon=$PID_ODOM PID=$PID_PID Bag=$PID_BAG"
echo "Press Ctrl+C to stop everything."

wait
