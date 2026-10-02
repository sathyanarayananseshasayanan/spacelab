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
        super().__init__("mpc_controller")

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

        self.deadband_xy = float(self.get_parameter("deadband_xy").value)
        self.deadband_yaw = np.deg2rad(float(self.get_parameter("deadband_yaw").value))
        self.deadband_vxy = float(self.get_parameter("deadband_vxy").value)
        self.deadband_r = float(self.get_parameter("deadband_r").value)
        self.terminal_scale = float(self.get_parameter("terminal_scale").value)

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/slider_1/thrust_cmd")

        self.frequency = float(self.get_parameter("frequency").value)
        self.dt = 1.0 / max(self.frequency, 1e-6)
        self.N = int(self.get_parameter("horizon").value)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        # -------------------
        # Reference (keep simple like you had)
        # ref_base/ref_vel are 6D now: [X, Y, psi, vx, vy, r]
        # -------------------
        self.ref_base = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.ref_vel  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        # Your CBF parameter (keep concept; you can drive it however you want)
        self.r2_value = 0.3  # you were setting it from a condition; keep as-is
        self.r3 = 0.2  # this is just for shifting the reference point forward of the CoM; tune as you like 

        self.t = 0.0
        self.have_state = False

        # State: [x, y, psi, vx, vy, r]
        self.state = np.zeros((6, 1), dtype=float)
        
        # Low-pass filter for position (reduce oscillations from noisy measurements)
        self.x_filt = 0.0
        self.y_filt = 0.0
        self.filter_alpha = 0.4  # 0=full smoothing, 1=no smoothing

        # Warm start memory
        self.prev_u = np.zeros((8, self.N), dtype=float)

        # -------------------
        # QoS (don’t use frequency as depth)
        # -------------------


        self.pub = self.create_publisher(Float32MultiArray, self.cmd_topic, 1)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)

        # Setup MPC
        self.setup_mpc()

        # Timer loop
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"MPC (Euler yaw) running at {self.frequency:.2f} Hz, horizon={self.N}, dt={self.dt:.3f}s"
        )

    # -------------------
    # Reference trajectory (6 x (N+1))
    # -------------------
    def compute_reference_traj(self) -> np.ndarray:
        refs = []
        for k in range(self.N + 1):
            tk = self.t + k * self.dt
            ref = self.ref_base + self.ref_vel * tk
            # Keep yaw wrapped to avoid huge numbers if you accumulate
            #ref[2] = wrap_angle(float(ref[2]))
            refs.append(ref)
        return np.array(refs).T  # shape (6, N+1)

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

        c = math.cos(psi)
        s = math.sin(psi)
        vx = c * vx_body - s * vy_body
        vy = s * vx_body + c * vy_body
        
        self.state = np.array([[x_raw], [y_raw], [psi], [vx_body], [vy_body], [r]], dtype=float)
        self.have_state = True

    # -------------------
    # MPC setup
    # -------------------
    def setup_mpc(self):
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(6, self.N + 1)  # [X,Y,psi,vx,vy,r]
        self.U = self.opti.variable(8, self.N)      # thrusters

        # Parameters
        self.x0p  = self.opti.parameter(6, 1)
        self.refp = self.opti.parameter(6, self.N + 1)
        self.r2   = self.opti.parameter()  # keep your CBF param

        # Weights (tune as you like)
        # [X, Y, psi, vx, vy, r]
        # Reduced position weights + higher velocity weights for damping
        Q = np.diag([3.0, 3.0, 10.0, 4.0, 4.0, 4.0])
        #Qf = np.diag([2.0, 2.0, 5.0, 1.0, 1.0, 5.0])  # stronger terminal weight
        #R = np.diag([0.01] * 8)
        du_weight = 0.1 # weight on input changes (for smoothness)
        

        # Dynamics f(x,u)
        def f(x, u):
            # x = [X, Y, psi, vx, vy, r]
            X   = x[0]
            Y   = x[1]
            psi = x[2]
            vx  = x[3]
            vy  = x[4]
            r   = x[5]

            # u_body = [Fx_b, Fy_b, tau_z]
            u_body = A @ u
            Fx_b = u_body[0]
            Fy_b = u_body[1]
            tau  = u_body[2]

            c = ca.cos(psi)
            s = ca.sin(psi)

            # body -> world forces
            Fx_w = c * Fx_b - s * Fy_b
            Fy_w = s * Fx_b + c * Fy_b

            d_r = 0.0  # yaw damping (set >0 if you want)

            dX   = vx
            dY   = vy
            dpsi = r
            dvx  = Fx_w / mass
            dvy  = Fy_w / mass
            dr   = (tau - d_r * r) / Izz

            return ca.vertcat(dX, dY, dpsi, dvx, dvy, dr)

        def rk4_step(ff, Xk, Uk, Ts):
            k1 = ff(Xk, Uk)
            k2 = ff(Xk + Ts/2*k1, Uk)
            k3 = ff(Xk + Ts/2*k2, Uk)
            k4 = ff(Xk + Ts*k3, Uk)
            return Xk + Ts/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.x0p)

        # Cost + constraints
        J = 0
        for k in range(self.N):
            # Wrapped yaw error
            psi   = self.X[2, k]
            psi_d = self.refp[2, k]
            e_psi = ca.atan2(ca.sin(psi - psi_d), ca.cos(psi - psi_d))

            dx = ca.vertcat(
                deadband_error(self.X[0, k] - self.refp[0, k], self.deadband_xy),
                deadband_error(self.X[1, k] - self.refp[1, k], self.deadband_xy),
                deadband_error(
                    ca.atan2(ca.sin(self.X[2, k] - self.refp[2, k]),
                    ca.cos(self.X[2, k] - self.refp[2, k])),
                    self.deadband_yaw
                ),
                deadband_error(self.X[3, k] - self.refp[3, k], self.deadband_vxy),
                deadband_error(self.X[4, k] - self.refp[4, k], self.deadband_vxy),
                deadband_error(self.X[5, k] - self.refp[5, k], self.deadband_r),
            )
            

            J += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            dxN = ca.vertcat(
                deadband_error(self.X[0, self.N] - self.refp[0, self.N], self.deadband_xy),
                deadband_error(self.X[1, self.N] - self.refp[1, self.N], self.deadband_xy),
                deadband_error(
                    ca.atan2(ca.sin(self.X[2, self.N] - self.refp[2, self.N]),
                    ca.cos(self.X[2, self.N] - self.refp[2, self.N])),
                    self.deadband_yaw
                ),
                deadband_error(self.X[3, self.N] - self.refp[3, self.N], self.deadband_vxy),
                deadband_error(self.X[4, self.N] - self.refp[4, self.N], self.deadband_vxy),
                deadband_error(self.X[5, self.N] - self.refp[5, self.N], self.deadband_r),
                )

            J += self.terminal_scale * ca.mtimes([dxN.T, Qf, dxN])

            # Smooth input changes to prevent thruster chattering
            if k > 0:
                du = self.U[:, k] - self.U[:, k-1]
                J += du_weight * ca.dot(du, du)

            # Dynamics constraint
            Xn = rk4_step(f, self.X[:, k], self.U[:, k], self.dt)

            # Keep psi bounded a bit (not required, but helps numerics)
            # Wrap via atan2(sin,cos) at next step:
            #psi_n = Xn[2]
            #psi_n = ca.atan2(ca.sin(psi_n), ca.cos(psi_n))
            #Xn = ca.vertcat(Xn[0], Xn[1], psi_n, Xn[3], Xn[4], Xn[5])

            self.opti.subject_to(self.X[:, k + 1] == Xn)

            # Thruster bounds
            for thr in range(8):
                self.opti.subject_to(self.U[thr, k] >= 0.0)
                self.opti.subject_to(self.U[thr, k] <= MAX_FORCE)

            # -------------------------------
            # CBF block (KEEP CONCEPT — still here, still optional)
            # You had it commented out; keep it commented unless you want to enable.
            # If you enable it, confirm your exact definition of B, B_dot, B_ddot.
            # -------------------------------
            # pos      = self.X[0:2, k]
            # pos_goal = self.refp[0:2, k]
            # vel      = self.X[3:5, k]  # world vx, vy
            # #
            # e = pos - pos_goal
            # B = ca.dot(e, e) - 4 * self.r2**2
            # B_dot = 2 * ca.dot(e, vel)
            # #
            # u_body = A @ self.U[:, k]
            # Fx_b, Fy_b = u_body[0], u_body[1]
            # c = ca.cos(self.X[2, k]); s = ca.sin(self.X[2, k])
            # ax = (c*Fx_b - s*Fy_b) / mass
            # ay = (s*Fx_b + c*Fy_b) / mass
            # #
            # B_ddot = 2 * ca.dot(vel, vel) + 2 * ca.dot(ca.vertcat(ax, ay), e)
            # #
            # k1 = 7.0
            # k0 = 1.0
            # self.opti.subject_to(B_ddot + k1*B_dot + k0*B >= 0)

        self.opti.minimize(J)
        self.opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

    # -------------------
    # Control loop
    # -------------------
    def control_loop(self):
        if not self.have_state:
            return

        ref_traj = self.compute_reference_traj()
        self.opti.set_value(self.refp, ref_traj)
        self.opti.set_value(self.x0p, self.state)

        # keep your r2 logic (you were setting it to 0 anyway)
        self.opti.set_value(self.r2, float(self.r2_value))

        # Warm start
        self.opti.set_initial(self.U, self.prev_u)
        self.opti.set_initial(self.X, np.tile(self.state, (1, self.N + 1)))
        
        try:
            sol = self.opti.solve()
            u0 = sol.value(self.U[:, 0])
            self.prev_u = sol.value(self.U)            
        except Exception as e:
            self.get_logger().warn(f"Optimization failed: {e}")
            u0 = np.zeros(8, dtype=float)

        # Clamp and publish
        u0 = np.clip(np.array(u0, dtype=float).reshape(-1), 0.0, MAX_FORCE)

        msg = Float32MultiArray()
        msg.data = u0.tolist()
        self.pub.publish(msg)

        # Debug
        x, y, psi, vx, vy, r = (self.state[i, 0] for i in range(6))
        self.get_logger().info(
            f"state: x={x:.3f} y={y:.3f} psi={psi:.3f} vx={vx:.3f} vy={vy:.3f} r={r:.3f}"
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