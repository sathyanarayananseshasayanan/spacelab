#!/usr/bin/env python3
import math
import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# -------------------------------------------------------------------------
# VEHICLE PARAMETERS (keep yours)
# -------------------------------------------------------------------------
mass = 4.528
Izz  = 0.109214481
MAX_FORCE = 0.7# N per thruster

A = np.array([
    [-1,     0,     1,     0,     1,     0,    -1,     0],
    [ 0,     1,     0,     1,     0,    -1,     0,    -1],
    [-0.14,  0.14,  0.14, -0.14, -0.14,  0.14,  0.14, -0.14],
], dtype=np.float64)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
def deadband_error(e, db):
    """
    Returns zero inside the deadband, and the remaining error outside it.
    Works for CasADi expressions too.
    """
    return ca.sign(e) * ca.fmax(ca.fabs(e) - db, 0.0)


# =========================================================================
# MPC NODE (PLANAR EULER-YAW MODEL)
# State: [X, Y, psi, vx, vy, r]  (world pos/vel, yaw, yaw rate)
# Inputs: u (8 thrusters), u_body = A u = [Fx_b, Fy_b, tau_z]
# =========================================================================
class SingleMPC(Node):
    def __init__(self):
        super().__init__("mpc_circle")

        # -------------------
        # Parameters
        # -------------------
        self.declare_parameter("frequency", 10)
        self.declare_parameter("horizon", 8)
        self.declare_parameter("deadband_xy", 0.03)     # meters
        self.declare_parameter("deadband_yaw", 3.0)     # degrees
        self.declare_parameter("deadband_vxy", 0.02)    # m/s for linear velocity
        self.declare_parameter("deadband_r", 0.05)      # rad/s for angular velocity
        self.declare_parameter("terminal_scale", 4.0)   # terminal cost weight
        self.declare_parameter("use_integral_xy", False)
        self.declare_parameter("integral_gain_xy", 0.35)
        self.declare_parameter("integral_limit_xy", 0.30)
        self.declare_parameter("integral_leak_xy", 0.00)
        self.declare_parameter("integral_deadband_xy", 0.00)
        self.declare_parameter("use_cbf", True)
        self.declare_parameter("cbf_k1", 4.0)
        self.declare_parameter("cbf_k0", 4.0)
        self.declare_parameter("cbf_goal_shift", 0.2)
        self.declare_parameter("use_soft_cbf", False)
        self.declare_parameter("cbf_slack_weight", 500.0)
        self.declare_parameter("safe_distance", 0.3)
        self.declare_parameter("dock_safe_distance", 0.15)

        self.deadband_xy = float(self.get_parameter("deadband_xy").value)
        self.deadband_yaw = np.deg2rad(float(self.get_parameter("deadband_yaw").value))
        self.deadband_vxy = float(self.get_parameter("deadband_vxy").value)
        self.deadband_r = float(self.get_parameter("deadband_r").value)
        self.terminal_scale = float(self.get_parameter("terminal_scale").value)
        self.use_integral_xy = bool(self.get_parameter("use_integral_xy").value)
        self.integral_gain_xy = float(self.get_parameter("integral_gain_xy").value)
        self.integral_limit_xy = float(self.get_parameter("integral_limit_xy").value)
        self.integral_leak_xy = float(self.get_parameter("integral_leak_xy").value)
        self.integral_deadband_xy = float(self.get_parameter("integral_deadband_xy").value)
        self.use_cbf = bool(self.get_parameter("use_cbf").value)
        self.cbf_k1 = float(self.get_parameter("cbf_k1").value)
        self.cbf_k0 = float(self.get_parameter("cbf_k0").value)
        self.cbf_goal_shift = float(self.get_parameter("cbf_goal_shift").value)
        self.use_soft_cbf = bool(self.get_parameter("use_soft_cbf").value)
        self.cbf_slack_weight = float(self.get_parameter("cbf_slack_weight").value)
        self.safe_distance = float(self.get_parameter("safe_distance").value)
        self.dock_safe_distance = float(self.get_parameter("dock_safe_distance").value)
        self.r_step_down = 0.008
        self.r_min = 0.12
        self.prev_abs_errors = None

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/slider_1/thrust_cmd")

        self.frequency = float(self.get_parameter("frequency").value)
        self.dt = 1.0 / max(self.frequency, 1e-6)
        self.N = int(self.get_parameter("horizon").value)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        # Circle reference: radius 0.5 m around the origin.
        # psi_ref is the outward radial direction; the controller's existing
        # 180 deg yaw offset keeps the backward axis pointing toward the center.
        self.circle_center = np.array([0.0, 0.0], dtype=float)
        self.circle_radius = 1.0
        self.circle_omega = 0.1    # rad/s, positive = counterclockwise
        self.circle_phase = 0.0


        # --------------------
        # Reference (keep simple like you had)
        # ref_base/ref_vel are 6D now: [X, Y, psi, vx, vy, r]
        # -------------------
        self.ref_base = np.array([self.circle_radius, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.ref_vel  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        # Your CBF parameter (keep concept; you can drive it however you want)
        self.r2 = 0.0  # you were setting it from a condition; keep as-is
         # this is just for shifting the reference point forward of the CoM; tune as you like 
        self.delta=0.35
        self.r3=0.0
        self.t = 0.0
        self.have_state = False
        self.r = self.safe_distance

        # State: [x, y, psi, vx, vy, r]
        self.state = np.zeros((6, 1), dtype=float)
        self.goal_traj = np.tile(self.ref_base.reshape(6, 1), (1, self.N + 1))
        self.outer_ref = np.tile(self.state, (1, self.N + 1))
        self.prev_u_outer = np.zeros((8, self.N), dtype=float)
        self.prev_u_inner = np.zeros((8, self.N), dtype=float)
        
        # Low-pass filter for position (reduce oscillations from noisy measurements)
        self.x_filt = 0.0
        self.y_filt = 0.0
        self.filter_alpha = 0.4  # 0=full smoothing, 1=no smoothing

        # Warm start memory
        self.prev_u = np.zeros((8, self.N), dtype=float)
        self.integral_xy = np.zeros(2, dtype=float)

        # -------------------
        # QoS (don’t use frequency as depth)
        # -------------------


        self.pub = self.create_publisher(Float32MultiArray, self.cmd_topic, 1)
        self.sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        #self.sub1 = self.create_subscription(Odometry, "/odom1", self.odom_callback1, 10)

        # Setup dual MPC layers
        self.setup_mpc()

        # Timer loop
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"MPC (Euler yaw) running at {self.frequency:.2f} Hz, horizon={self.N}, dt={self.dt:.3f}s"
        )
        self.get_logger().info(
            f"Integral XY: enabled={self.use_integral_xy}, gain={self.integral_gain_xy:.3f}, "
            f"limit={self.integral_limit_xy:.3f}, leak={self.integral_leak_xy:.3f}, "
            f"db={self.integral_deadband_xy:.3f}"
        )
        self.get_logger().info(
            f"CBF: enabled={self.use_cbf}, k1={self.cbf_k1:.3f}, k0={self.cbf_k0:.3f}, "
            f"soft={self.use_soft_cbf}, slack_w={self.cbf_slack_weight:.1f}, shift={self.cbf_goal_shift:.3f}"
        )

    # -------------------
    # Reference trajectory (6 x (N+1))
    # -------------------
    def build_goal_traj(self) -> np.ndarray:
        """
        Build a circular reference trajectory over the horizon.
        The circle is centered at the origin and has fixed radius.
        psi_ref is the outward radial direction, so the controller's
        existing 180 deg yaw offset keeps the backward axis pointing inward.
        """
        goal_traj = np.zeros((6, self.N + 1), dtype=float)
        for k in range(self.N + 1):
            tk = self.t + k * self.dt
            theta = self.circle_phase + self.circle_omega * tk
            x = self.circle_center[0] + self.circle_radius * math.cos(theta)
            y = self.circle_center[1] + self.circle_radius * math.sin(theta)

            # Outward radial direction. With the controller's 180 deg offset,
            # the vehicle's backward axis points toward the center.
            psi = wrap_angle(theta)

            vx = -self.circle_radius * self.circle_omega * math.sin(theta)
            vy = self.circle_radius * self.circle_omega * math.cos(theta)
            r = self.circle_omega

            goal_traj[0, k] = x
            goal_traj[1, k] = y
            goal_traj[2, k] = psi
            goal_traj[3, k] = vx
            goal_traj[4, k] = vy
            goal_traj[5, k] = r
        return goal_traj
    def odom_callback1(self, msg):
        # ego (slider_0) odom callback
        x_raw = msg.pose.pose.position.x
        y_raw = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        vx_body = msg.twist.twist.linear.x
        vy_body = msg.twist.twist.linear.y
        r  = msg.twist.twist.angular.z

        _, _, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        psi = wrap_angle(float(psi))

        
        self.ref_base = np.array([x_raw, y_raw, psi, vx_body, vy_body, r])

    def update_barrier_radius(self, rx: float, ry: float, rpsi: float) -> float:
        abs_rx = abs(rx)
        abs_ry = abs(ry)
        abs_rpsi = abs(rpsi)
        
        far_rx = self.safe_distance - self.cbf_goal_shift+0.25
        far_ry = 0.07
        far_rpsi = np.deg2rad(8.0)

        if abs_rx > far_rx or abs_ry > far_ry or abs_rpsi > far_rpsi:
            target_r = self.safe_distance
        else:
            target_r = self.r
            if self.prev_abs_errors is not None:
                prev_rx, prev_ry, prev_rpsi = self.prev_abs_errors
                improving = (abs_rx <= prev_rx) and (abs_ry <= prev_ry) and (abs_rpsi <= prev_rpsi)
                if improving:
                    target_r = self.r - self.r_step_down

        self.prev_abs_errors = (abs_rx, abs_ry, abs_rpsi)
        return float(np.clip(target_r, self.dock_safe_distance, self.safe_distance))

    # -------------------
    # ODOM callback -> state
    # -------------------
    def odom_callback(self, msg: Odometry):
        x_raw = msg.pose.pose.position.x
        y_raw = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        vx_body = msg.twist.twist.linear.x
        vy_body = msg.twist.twist.linear.y
        r  = msg.twist.twist.angular.z

        _, _, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        psi = wrap_angle(float(psi))
        self.state = np.array([[x_raw], [y_raw], [psi], [vx_body], [vy_body], [r]], dtype=float)
        self.have_state = True
    # -------------------
    # MPC setup
    # -------------------
    def setup_mpc(self):
        self.setup_outer_mpc()
        self.setup_inner_mpc()

    def _dynamics(self, x, u):
        X = x[0]
        Y = x[1]
        psi = x[2]
        vx = x[3]
        vy = x[4]
        r = x[5]

        u_body = A @ u
        Fx_b = u_body[0]
        Fy_b = u_body[1]
        tau = u_body[2]

        c = ca.cos(psi)
        s = ca.sin(psi)
        Fx_w = c * Fx_b - s * Fy_b
        Fy_w = s * Fx_b + c * Fy_b

        d_r = 0.0
        dX = vx
        dY = vy
        dpsi = r
        dvx = Fx_w / mass
        dvy = Fy_w / mass
        dr = (tau - d_r * r) / Izz
        return ca.vertcat(dX, dY, dpsi, dvx, dvy, dr)

    def _rk4_step(self, ff, Xk, Uk, Ts):
        k1 = ff(Xk, Uk)
        k2 = ff(Xk + Ts / 2 * k1, Uk)
        k3 = ff(Xk + Ts / 2 * k2, Uk)
        k4 = ff(Xk + Ts * k3, Uk)
        return Xk + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def setup_outer_mpc(self):
        self.opti_outer = ca.Opti()
        self.Xo = self.opti_outer.variable(6, self.N + 1)
        self.Uo = self.opti_outer.variable(8, self.N)
        self.outer_slack = self.opti_outer.variable(1, self.N)

        self.x0o = self.opti_outer.parameter(6, 1)
        self.goalp = self.opti_outer.parameter(6, self.N + 1)
        self.r2_outer = self.opti_outer.parameter()

        Qo = np.diag([6.0, 6.0, 10.0, 1.0, 1.0, 4.0])
        Ro = np.diag([0.00] * 8)
        du_weight = 0.0

        self.opti_outer.subject_to(self.Xo[:, 0] == self.x0o)
        Jo = 0
        for k in range(self.N):
            goal_k = self.goalp[:, k]
            e_psi = ca.atan2(ca.sin(self.Xo[2, k] - goal_k[2]), ca.cos(self.Xo[2, k] - goal_k[2]))
            e_pos = self.Xo[0:2, k] - goal_k[0:2]+self.cbf_goal_shift*ca.vertcat(ca.cos(goal_k[2]), ca.sin(goal_k[2]))# Debug
            e_vel = self.Xo[3:5, k] - goal_k[3:5]
            e_r = self.Xo[5, k] - goal_k[5]

            dx = ca.vertcat(e_pos[0], e_pos[1], e_psi, e_vel[0], e_vel[1], e_r)
            Jo += ca.mtimes([dx.T, Qo, dx]) + ca.mtimes([self.Uo[:, k].T, Ro, self.Uo[:, k]])
            if k > 0:
                du = self.Uo[:, k] - self.Uo[:, k - 1]
                Jo += du_weight * ca.dot(du, du)

            Xn = self._rk4_step(self._dynamics, self.Xo[:, k], self.Uo[:, k], self.dt)
            self.opti_outer.subject_to(self.Xo[:, k + 1] == Xn)

            for thr in range(8):
                self.opti_outer.subject_to(self.Uo[thr, k] >= 0.0)
                self.opti_outer.subject_to(self.Uo[thr, k] <= MAX_FORCE)

            self.opti_outer.subject_to(self.outer_slack[0, k] >= 0.0)
            e = self.Xo[0:2, k] - goal_k[0:2] 
            vel_rel = self.Xo[3:5, k] - goal_k[3:5]
            B = ca.dot(e, e) - self.r2_outer ** 2
            B_dot = 2 * ca.dot(e, vel_rel)
            u_body = A @ self.Uo[:, k]
            c = ca.cos(self.Xo[2, k])
            s = ca.sin(self.Xo[2, k])
            ax = (c * u_body[0] - s * u_body[1]) / mass
            ay = (s * u_body[0] + c * u_body[1]) / mass
            B_ddot = 2 * ca.dot(vel_rel, vel_rel) + 2 * ca.dot(ca.vertcat(ax, ay), e)
            if self.use_cbf:
                if self.use_soft_cbf:
                    Jo += self.cbf_slack_weight * self.outer_slack[0, k] ** 2
                    self.opti_outer.subject_to(B_ddot + self.cbf_k1 * B_dot + self.cbf_k0 * B + self.outer_slack[0, k] >= 0)
                else:
                    self.opti_outer.subject_to(self.outer_slack[0, k] == 0.0)
                    self.opti_outer.subject_to(B_ddot + self.cbf_k1 * B_dot + self.cbf_k0 * B >= 0)
            else:
                self.opti_outer.subject_to(self.outer_slack[0, k] == 0.0)

        self.opti_outer.minimize(Jo)
        self.opti_outer.solver("ipopt", {"print_time": False}, {"print_level": 0})

    def setup_inner_mpc(self):
        self.opti_inner = ca.Opti()
        self.Xi = self.opti_inner.variable(6, self.N + 1)
        self.Ui = self.opti_inner.variable(8, self.N)

        self.x0i = self.opti_inner.parameter(6, 1)
        self.refi = self.opti_inner.parameter(6, self.N + 1)
        self.inti_p = self.opti_inner.parameter(2, 1)

        Qi = np.diag([3.0, 3.0, 20.0, 4.0, 4.0, 4.0])
        Ri = np.diag([0.00] * 8)
        du_weight = 0.0

        self.opti_inner.subject_to(self.Xi[:, 0] == self.x0i)
        Ji = 0
        for k in range(self.N):
            ref_k = self.refi[:, k]

            e_psi = ca.atan2(ca.sin(self.Xi[2, k] - ref_k[2]), ca.cos(self.Xi[2, k] - ref_k[2]))
            e_pos = self.Xi[0:2, k] - ref_k[0:2] + self.integral_gain_xy * self.inti_p[:, 0]
            e_vel = self.Xi[3:5, k] - ref_k[3:5]
            e_r = self.Xi[5, k] - ref_k[5]

            dx = ca.vertcat(e_pos[0], e_pos[1], e_psi, e_vel[0], e_vel[1], e_r)
            Ji += ca.mtimes([dx.T, Qi, dx]) + ca.mtimes([self.Ui[:, k].T, Ri, self.Ui[:, k]])
            if k > 0:
                du = self.Ui[:, k] - self.Ui[:, k - 1]
                Ji += du_weight * ca.dot(du, du)

            Xn = self._rk4_step(self._dynamics, self.Xi[:, k], self.Ui[:, k], self.dt)
            self.opti_inner.subject_to(self.Xi[:, k + 1] == Xn)

            for thr in range(8):
                self.opti_inner.subject_to(self.Ui[thr, k] >= 0.0)
                self.opti_inner.subject_to(self.Ui[thr, k] <= MAX_FORCE)

        self.opti_inner.minimize(Ji)
        self.opti_inner.solver("ipopt", {"print_time": False}, {"print_level": 0})

    # -------------------
    # Control loop
    # -------------------
    def control_loop(self):
        if not self.have_state:
            return

        ref_traj = self.build_goal_traj()
        psi = float(self.state[2, 0])
        psi_d = float(ref_traj[2, 0])
        rx = float(self.state[0, 0] - ref_traj[0, 0]) +self.cbf_goal_shift * ca.cos(psi_d)
        ry = float(self.state[1, 0] - ref_traj[1, 0]) + self.cbf_goal_shift * ca.sin(psi_d)
        rx_body= rx * math.cos(psi_d) + ry * math.sin(psi_d)
        ry_body= -rx * math.sin(psi_d) + ry * math.cos(psi_d)
        rpsi = wrap_angle(psi - psi_d-180.0*math.pi/180.0)
        self.r = self.update_barrier_radius(rx_body, ry_body, rpsi)

        if self.use_integral_xy:
            e_xy = np.array([
                float(self.state[0, 0] - self.outer_ref[0, 0]),
                float(self.state[1, 0] - self.outer_ref[1, 0]),
            ], dtype=float)

            if self.integral_deadband_xy > 0.0:
                e_xy[np.abs(e_xy) < self.integral_deadband_xy] = 0.0

            leak = max(0.0, self.integral_leak_xy)
            decay = max(0.0, 1.0 - leak * self.dt)
            self.integral_xy = decay * self.integral_xy + self.dt * e_xy

            i_lim = max(1e-6, self.integral_limit_xy)
            self.integral_xy = np.clip(self.integral_xy, -i_lim, i_lim)
        else:
            self.integral_xy[:] = 0.0

        # Outer MPC solve: produces safe reference trajectory
        self.opti_outer.set_value(self.x0o, self.state)
        self.opti_outer.set_value(self.goalp, ref_traj)
        self.opti_outer.set_value(self.r2_outer, self.r)
        self.opti_outer.set_initial(self.Uo, self.prev_u_outer)
        self.opti_outer.set_initial(self.Xo, np.tile(self.state, (1, self.N + 1)))

        try:
            sol_outer = self.opti_outer.solve()
            self.outer_ref = sol_outer.value(self.Xo)
            self.prev_u_outer = sol_outer.value(self.Uo)
        except Exception as e:
            self.get_logger().warn(f"Outer MPC failed: {e}")
            self.outer_ref = ref_traj.copy()

        # Add integral action into inner MPC tracking
        self.opti_inner.set_value(self.x0i, self.state)
        self.opti_inner.set_value(self.refi, self.outer_ref)
        self.opti_inner.set_value(self.inti_p, self.integral_xy.reshape(2, 1))
        self.opti_inner.set_initial(self.Ui, self.prev_u_inner)
        self.opti_inner.set_initial(self.Xi, np.tile(self.state, (1, self.N + 1)))
        
        try:
            sol_inner = self.opti_inner.solve()
            u0 = sol_inner.value(self.Ui[:, 0])
            self.prev_u_inner = sol_inner.value(self.Ui)
        except Exception as e:
            self.get_logger().warn(f"Inner MPC failed: {e}")
            u0 = np.zeros(8, dtype=float)
        e = self.state[0:2, 0] - ref_traj[0:2, 0]
        B = float(ca.dot(e, e) - self.r ** 2)
        # Clamp and publish
        u0 = np.clip(np.array(u0, dtype=float).reshape(-1), 0.0, MAX_FORCE)

        msg = Float32MultiArray()
        msg.data = u0.tolist()
        self.pub.publish(msg)
        e_pos = self.state[0:2, 0] - ref_traj[0:2, 0]+self.cbf_goal_shift*ca.vertcat(ca.cos(ref_traj[2, 0]), ca.sin(ref_traj[2, 0]))-ca.vertcat(0.016, 0.0)
        # Debug
        x, y, psi, vx, vy, r = (self.state[i, 0] for i in range(6))
        self.get_logger().info(
            f"state: rx={float(e_pos[0, 0]):.3f} ry={float(e_pos[1, 0]):.3f} rpsi={rpsi:.3f} | sd={self.r:.3f} | B={B:.3f}"
        )

        # Advance ref time
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = SingleMPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()