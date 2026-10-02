#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

MAX_FORCE = 0.7     
MIN_ON_TIME = 0.00

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class ThrustController(Node):
    def __init__(self):
        super().__init__('pwm_mpc_publisher')

        # Paper-like defaults: 10 Hz control loop, 2-step pulse window
        self.declare_parameter('frequency', 10.0)
        self.declare_parameter('resolution', 2)
        self.declare_parameter('max_force', MAX_FORCE)
        self.declare_parameter('min_on_time', MIN_ON_TIME)

        self.frequency = int(self.get_parameter('frequency').value)
        self.resolution = int(self.get_parameter('resolution').value)
        self.max_force = float(self.get_parameter('max_force').value)
        self.min_on_time = float(self.get_parameter('min_on_time').value)

        self.period = 1.0 / self.frequency
        self.tick_dt = self.period / self.resolution

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.sub = self.create_subscription(
            Float32MultiArray,
            '/slider_1/thrust_cmd',
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            UInt8MultiArray,
            '/eight_thrust_pulse',
            1
        )

        # per-thruster pulse patterns for the current control period
        self.signals = [[0] * self.resolution for _ in range(8)]
        self.i = 0

        self.create_timer(self.tick_dt, self.send_signals)

        self.get_logger().info(
            f"Running: frequency={self.frequency} Hz, resolution={self.resolution}, "
            f"tick_dt={self.tick_dt:.3f} s, min_on_time={self.min_on_time:.3f} s"
        )
    def callback(self, msg: Float32MultiArray):

        if len(msg.data) < 8:
            self.get_logger().warn(
                "thrust_cmd has fewer than 8 elements"
            )
            return

        T = np.array(
            msg.data[:8],
            dtype=float
        )

        for k in range(8):

            # Clamp requested force
            thrust = clamp(
                T[k],
                0.0,
                self.max_force
            )

            # =====================================================
            # IMPORTANT FIX:
            # Zero requested thrust must mean completely OFF
            # =====================================================

            if thrust <= 1e-6:

                self.signals[k] = (
                    [0] * self.resolution
                )

                continue

            # =====================================================
            # Convert desired force to PWM duty cycle
            # =====================================================

            duty = (
                thrust
                / self.max_force
            )

            # Number of ON ticks in the 6-tick PWM period
            n_on = int(
                round(
                    duty
                    * self.resolution
                )
            )

            # =====================================================
            # Quantization can legitimately produce zero ticks
            # =====================================================

            if n_on <= 0:

                self.signals[k] = (
                    [0] * self.resolution
                )

                continue

            # Never exceed the available resolution
            n_on = min(
                n_on,
                self.resolution
            )

            # =====================================================
            # Build PWM pattern
            # =====================================================

            self.signals[k] = (
                [1] * n_on
                +
                [0] * (
                    self.resolution
                    - n_on
                )
            )
    def callback(self, msg: Float32MultiArray):
        if len(msg.data) < 8:
            self.get_logger().warn("thrust_cmd has fewer than 8 elements")
            return

        T = np.array(msg.data[:8], dtype=float)

        for k in range(8):
            thrust = clamp(T[k], 0.0, self.max_force)

            # Convert force to on-time inside the current control period
            on_time = (thrust / self.max_force) * self.period

            # Paper logic: if the pulse is shorter than minimum on-time, ignore it
            if on_time < self.min_on_time:
                self.signals[k] = [0] * self.resolution
                continue

            # Convert on-time to number of ON ticks
            n_on = int(round(on_time / self.tick_dt))
            n_on = int(clamp(n_on, 1, self.resolution))

            # Contiguous ON pulse, then OFF
            self.signals[k] = [1] * n_on + [0] * (self.resolution - n_on)

    def send_signals(self):
        req = UInt8MultiArray()

        # publish one tick from each thruster pattern
        req.data = [int(self.signals[k][self.i]) for k in range(8)]
        self.pub.publish(req)

        self.i += 1
        if self.i >= self.resolution:
            self.i = 0

def main(args=None):
    rclpy.init(args=args)
    node = ThrustController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()