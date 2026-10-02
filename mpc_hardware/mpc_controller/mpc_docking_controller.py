#!/usr/bin/env python3
import math
import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion


# -------------------------------------------------------------------------
# VEHICLE PARAMETERS (keep yours)
# -------------------------------------------------------------------------
mass = 4.528
Izz  = 0.109214481
MAX_FORCE = 0.7# N per thruster
CAMERA_YAW_OFFSET = 0.03       # rad; calibrated camera/body yaw offset
LATERAL_ALIGN_TOL = 0.07       # m; switch from LOS to final docking yaw
YAW_ALIGN_TOL = math.radians(8.0)

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
        super().__init__("mpc_docking_controller")

        # -------------------
        # Parameters
        # -------------------
        self.declare_parameter("frequency", 10.0)
        # ``horizon`` is retained as a backward-compatible default. Set
        # outer_horizon and inner_horizon independently to use different MPC
        # prediction lengths (in samples).
        self.declare_parameter("horizon", 8.0)
        self.declare_parameter("outer_horizon", 3)
        self.declare_parameter("inner_horizon", 3)
        self.declare_parameter("deadband_xy", 0.03)     # meters
        self.declare_parameter("deadband_yaw", 3.0)     # degrees
        self.declare_parameter("deadband_vxy", 0.02)    # m/s for linear velocity
        self.declare_parameter("deadband_r", 0.05)      # rad/s for angular velocity
        self.declare_parameter("terminal_scale", 3.0)   # terminal cost weight
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
        self.r_step_down = 0.01
        self.dock_yaw = 0.0
        self.c=1
        self.z=0.0

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("dock_odom_topic", "/odom1_vision")
        self.declare_parameter("cmd_topic", "/slider_1/thrust_cmd")

        self.frequency = float(self.get_parameter("frequency").value)
        self.dt = 1.0 / max(self.frequency, 1e-6)
        legacy_horizon = int(self.get_parameter("horizon").value)
        outer_horizon = int(self.get_parameter("outer_horizon").value)
        inner_horizon = int(self.get_parameter("inner_horizon").value)
        self.N_outer = outer_horizon if outer_horizon > 0 else legacy_horizon
        self.N_inner = inner_horizon if inner_horizon > 0 else legacy_horizon
        if self.N_outer < 1 or self.N_inner < 1:
            raise ValueError("outer_horizon and inner_horizon must each be at least 1")

        self.odom_topic = self.get_parameter("odom_topic").value
        self.dock_odom_topic = self.get_parameter("dock_odom_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value


        # --------------------
        # Reference (keep simple like you had)
        # ref_base/ref_vel are 6D now: [X, Y, psi, vx, vy, r]
        # -------------------
        self.ref_base = np.array([0.0, 0.0, 0.00, 0.0, 0.0, 0.0], dtype=float)
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
        self.goal_traj = np.tile(self.ref_base.reshape(6, 1), (1, self.N_outer + 1))
        self.outer_ref = np.tile(self.state, (1, self.N_outer + 1))
        self.prev_u_outer = np.zeros((8, self.N_outer), dtype=float)
        self.prev_u_inner = np.zeros((8, self.N_inner), dtype=float)
        
        # Low-pass filter for position (reduce oscillations from noisy measurements)
        self.x_filt = 0.0
        self.y_filt = 0.0
        self.filter_alpha = 0.4  # 0=full smoothing, 1=no smoothing
        self.Qo=np.diag([6.0, 10.0, 10.0, 1.0, 1.0, 4.0])

        self.integral_xy = np.zeros(2, dtype=float)

        # -------------------
        # QoS (don’t use frequency as depth)
        # -------------------


        self.pub = self.create_publisher(Float32MultiArray, self.cmd_topic, 1)
        self.sub = self.create_subscription(Odometry,self.odom_topic, self.odom_callback, 10)
        self.sub1 = self.create_subscription(Odometry, self.dock_odom_topic, self.odom_callback1, 10)

        # Setup dual MPC layers
        self.setup_mpc()

        # Timer loop
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"MPC (Euler yaw) running at {self.frequency:.2f} Hz, "
            f"outer_horizon={self.N_outer}, inner_horizon={self.N_inner}, "
            f"dt={self.dt:.3f}s"
        )
        self.get_logger().info(
            f"Integral XY: enabled={self.use_integral_xy}, gain={self.integral_gain_xy:.3f}, "
            f"limit={self.integral_limit_xy:.3f}, leak={self.integral_leak_xy:.3f}, "
            f"db={self.integral_deadband_xy:.3f}"
        )
        
    # -------------------
    # Outer reference trajectory (6 x (N_outer+1))
    # -------------------
    def build_goal_traj(self) -> np.ndarray:
        """
        Build predicted docking-target trajectory over the horizon.
        The target is assumed to move with constant velocity from odom.
        """
        goal = np.array(self.ref_base, dtype=float)
        goal_traj = np.tile(goal.reshape(6, 1), (1, self.N_outer + 1))
        for k in range(self.N_outer + 1):
            tk = k * self.dt
            goal_traj[0, k] = goal[0] + goal[3] * tk   
            goal_traj[1, k] = goal[1] + goal[4] * tk
            goal_traj[2, k] = goal[2]
            goal_traj[3, k] = goal[3]
            goal_traj[4, k] = goal[4]
            goal_traj[5, k] = goal[5]
        return goal_traj

    def build_inner_reference(self, outer_ref: np.ndarray) -> np.ndarray:
        """Resize the outer prediction for the inner MPC horizon.

        Both MPC layers use the same sampling time, so matching prediction
        indices are copied directly. If the inner horizon is longer, the
        outer terminal state is held for the remaining inner prediction.
        """
        inner_ref = np.empty((6, self.N_inner + 1), dtype=float)
        copied_columns = min(outer_ref.shape[1], self.N_inner + 1)
        inner_ref[:, :copied_columns] = outer_ref[:, :copied_columns]
        if copied_columns < self.N_inner + 1:
            inner_ref[:, copied_columns:] = outer_ref[:, [-1]]
        return inner_ref
    def odom_callback1(self, msg):
        # ego (slider_0) odom callback
        x_raw = msg.pose.pose.position.x
        y_raw = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        # /odom1 twist is already expressed in the inertial/world frame.
        vx_world = msg.twist.twist.linear.x
        vy_world = msg.twist.twist.linear.y
        r  = msg.twist.twist.angular.z

        _, _, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        psi = wrap_angle(float(psi))

        
        self.ref_base = np.array([x_raw, y_raw, psi, vx_world, vy_world, r])

    def update_barrier_radius(self, rx_body: float, ry_body: float, rpsi: float) -> float:
        """Hold the CBF circle during alignment, then shrink it for docking."""
        docking_side_ready = rx_body < 0.0
        lateral_ready = abs(ry_body) <= LATERAL_ALIGN_TOL
        yaw_ready = abs(rpsi) <= YAW_ALIGN_TOL

        target_r = self.r
        if docking_side_ready and lateral_ready and yaw_ready:
            target_r -= self.r_step_down

        return float(np.clip(target_r, self.dock_safe_distance, self.safe_distance))

    # -------------------
    # ODOM callback -> state
    # -------------------
    def odom_callback(self, msg: Odometry):
        x_raw = msg.pose.pose.position.x
        y_raw = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        # /odom twist is already expressed in the inertial/world frame.
        vx_world = msg.twist.twist.linear.x
        vy_world = msg.twist.twist.linear.y
        r  = msg.twist.twist.angular.z

        _, _, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        psi = wrap_angle(float(psi))
        self.state = np.array([[x_raw], [y_raw], [psi], [vx_world], [vy_world], [r]], dtype=float)
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
        self.Xo = self.opti_outer.variable(6, self.N_outer + 1)
        self.Uo = self.opti_outer.variable(8, self.N_outer)
        self.outer_slack = self.opti_outer.variable(1, self.N_outer)

        self.x0o = self.opti_outer.parameter(6, 1)
        self.goalp = self.opti_outer.parameter(6, self.N_outer + 1)
        self.r2_outer = self.opti_outer.parameter()

        # [rx_body, ry_body, yaw, vx_world, vy_world, yaw_rate]
        
        Ro = np.diag([0.00] * 8)
        du_weight = 0.0

        self.opti_outer.subject_to(self.Xo[:, 0] == self.x0o)
        Jo = 0
        for k in range(self.N_outer):
            goal_k = self.goalp[:, k]
            
            # Position error from the shifted docking point to tself.he slider,
            # expressed first in world coordinates and then in the docking
            # station frame.
            cg = ca.cos(goal_k[2])
            sg = ca.sin(goal_k[2])
            e_pos_world = (
                self.Xo[0:2, k]
                - goal_k[0:2]
                + self.cbf_goal_shift * ca.vertcat(cg, sg)
                - ca.vertcat(0.021, 0.0))
            rx_body = cg * e_pos_world[0] + sg * e_pos_world[1]
            ry_body = -sg * e_pos_world[0] + cg * e_pos_world[1]
            

            # While laterally displaced, the rear camera tracks the station.
            # Once on the docking axis, use the stable final docking yaw.
            #psi_los = ca.atan2(e_pos_world[1], e_pos_world[0]) + CAMERA_YAW_OFFSET
            psi_los = ca.atan2(self.Xo[1, k]- goal_k[1], self.Xo[0, k] - goal_k[0]) + CAMERA_YAW_OFFSET
            psi_dock = goal_k[2] + math.pi + CAMERA_YAW_OFFSET
            final_axis_ready = ca.logic_and(
                ca.fabs(ry_body) <= 0.1,
                rx_body < 0.0,
            )




            

            psi_d = ca.if_else(final_axis_ready, psi_dock, psi_los)
          

            e_psi = ca.atan2(
                ca.sin(self.Xo[2, k] - psi_d),
                ca.cos(self.Xo[2, k] - psi_d), 
            )
            e_vel = self.Xo[3:5, k] - goal_k[3:5]
            e_r = self.Xo[5, k] - goal_k[5]
            Q2=np.diag([6.0, 6.0, 15.0, 6.0, 6.0, 4.0])
            Q1=np.diag([6.0, 10.0, 15.0, 6.0, 6.0, 4.0])
            Qo=Q2
            ex=ca.if_else(final_axis_ready, rx_body, e_pos_world[0])
            ey=ca.if_else(final_axis_ready, ry_body, e_pos_world[1])
            
            

            dx = ca.vertcat(e_pos_world[0], e_pos_world[1], e_psi, e_vel[0], e_vel[1], e_r)
            Jo += ca.mtimes([dx.T, Qo, dx]) + ca.mtimes([self.Uo[:, k].T, Ro, self.Uo[:, k]])
            if k > 0:
                du = self.Uo[:, k] - self.Uo[:, k - 1]
                Jo += du_weight * ca.dot(du, du)
            
            Xn = self._rk4_step(self._dynamics, self.Xo[:, k], self.Uo[:, k], self.dt)
            e_pos_world = (
                self.Xo[0:2, self.N_outer]
                - self.goalp[0:2, self.N_outer]
                +self.cbf_goal_shift * ca.vertcat(cg, sg)
                - ca.vertcat(0.021, 0.0))
            rx_body = cg * e_pos_world[0] + sg * e_pos_world[1]
            ry_body = -sg * e_pos_world[0] + cg * e_pos_world[1]
            

            # While laterally displaced, the rear camera tracks the station.
            # Once on the docking axis, use the stable final docking yaw.
            #psi_los = ca.atan2(e_pos_world[1], e_pos_world[0]) + CAMERA_YAW_OFFSET
            
            


            psi_los_N = (
                ca.atan2(
                    self.Xo[1, self.N_outer] - self.goalp[1, self.N_outer],
                    self.Xo[0, self.N_outer] - self.goalp[0, self.N_outer],
                )
                + CAMERA_YAW_OFFSET
            )

            psi_dock_N = (
                self.goalp[2, self.N_outer]
                + math.pi
                + CAMERA_YAW_OFFSET
            )
            final_axis_ready = ca.logic_and(
                ca.fabs(ry_body) <= 0.1,
                rx_body < 0.0,
            )

            psi_desired_N = ca.if_else(
                final_axis_ready,
                psi_dock_N,
                psi_los_N,
            )

            e_psi_N = ca.atan2(
                ca.sin(
                    self.Xo[2, self.N_outer]
                    - psi_desired_N
                ),
                ca.cos(
                    self.Xo[2, self.N_outer]
                    - psi_desired_N
                ),
            )

            terminal_error = ca.vertcat(
                rx_body,
                ry_body,
                e_psi_N,
                self.Xo[3, self.N_outer] - self.goalp[3, self.N_outer],
                self.Xo[4, self.N_outer] - self.goalp[4, self.N_outer],
                self.Xo[5, self.N_outer] - self.goalp[5, self.N_outer],
            )

            Jo += self.terminal_scale * ca.mtimes([
                terminal_error.T,
                Qo,
                terminal_error,
            ])

            #self.opti_outer.minimize(Jo)

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
        self.Xi = self.opti_inner.variable(6, self.N_inner + 1)
        self.Ui = self.opti_inner.variable(8, self.N_inner)

        self.x0i = self.opti_inner.parameter(6, 1)
        self.refi = self.opti_inner.parameter(6, self.N_inner + 1)
        self.inti_p = self.opti_inner.parameter(2, 1)

        Qi = np.diag([3.0, 3.0, 10.0, 4.0, 4.0, 4.0])
        Ri = np.diag([0.00] * 8)
        du_weight = 0.0

        self.opti_inner.subject_to(self.Xi[:, 0] == self.x0i)
        Ji = 0
        for k in range(self.N_inner):
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
        goal_psi = float(ref_traj[2, 0])
        cg = math.cos(goal_psi)
        sg = math.sin(goal_psi)
        rx_world = float(self.state[0, 0] - ref_traj[0, 0]) + self.cbf_goal_shift * cg - 0.021
        ry_world = float(self.state[1, 0] - ref_traj[1, 0]) + self.cbf_goal_shift * sg
        rx_body = cg * rx_world + sg * ry_world
        ry_body = -sg * rx_world + cg * ry_world
        vx_body = (
                math.cos(goal_psi) * float(self.state[3, 0])
                + math.sin(goal_psi) * float(self.state[4, 0])
                )
        vy_body = (
                 math.cos(goal_psi) * float(self.state[4, 0])
                 - math.sin(goal_psi) * float(self.state[3, 0])
                )
        Qo=np.diag([6.0, 10.0, 10.0, 1.0, 1.0, 4.0])
        Q1=np.diag([6.0, 6.0, 10.0, 1.0, 1.0, 4.0])
        self.Qo=Qo
        
        psi_los = math.atan2(self.state[1, 0] - ref_traj[1, 0], self.state[0, 0] - ref_traj[0, 0]) + CAMERA_YAW_OFFSET
        psi_dock = goal_psi + math.pi + CAMERA_YAW_OFFSET
        final_axis_ready = abs(ry_body) <= LATERAL_ALIGN_TOL and rx_body < 0.0
        desired_slider_yaw = psi_dock if final_axis_ready else psi_los
        
      
        rpsi = wrap_angle(psi -desired_slider_yaw)
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
        self.opti_outer.set_initial(self.Xo, np.tile(self.state, (1, self.N_outer + 1)))

        try:
            sol_outer = self.opti_outer.solve()
            self.outer_ref = sol_outer.value(self.Xo)
            self.prev_u_outer = sol_outer.value(self.Uo)
        except Exception as e:
            self.get_logger().warn(f"Outer MPC failed: {e}")
            self.outer_ref = ref_traj.copy()

        # Add integral action into inner MPC tracking
        inner_ref = self.build_inner_reference(self.outer_ref)
        self.opti_inner.set_value(self.x0i, self.state)
        self.opti_inner.set_value(self.refi, inner_ref)
        self.opti_inner.set_value(self.inti_p, self.integral_xy.reshape(2, 1))
        self.opti_inner.set_initial(self.Ui, self.prev_u_inner)
        self.opti_inner.set_initial(self.Xi, np.tile(self.state, (1, self.N_inner + 1)))
        
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
        self.get_logger().info(
            f"state: rx_body={rx_body:.3f} ry_body={ry_body:.3f} "
            f"rpsi={rpsi:.3f} mode={'DOCK' if final_axis_ready else 'LOS'} "
            f"| sd={self.r:.3f} | B={B:.3f}| vx_body={vx_body:.3f} | vy_body={vy_body:.3f}"
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
