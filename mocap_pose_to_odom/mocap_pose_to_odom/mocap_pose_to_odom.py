import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, sosfilt, sosfilt_zi
from mocap4r2_msgs.msg import RigidBodies

from .config import MocapCfg


def safe_quat(q):
    q = np.array(q, dtype=float)
    if not np.all(np.isfinite(q)):
        return None
    n = np.linalg.norm(q)
    if n < 1e-8:
        return None
    return q / n


def slerp_quat(q0, q1, alpha: float):
    """
    Spherical linear interpolation between unit quaternions q0 -> q1.
    alpha in [0,1]
    """
    q0 = safe_quat(q0)
    q1 = safe_quat(q1)
    if q0 is None or q1 is None:
        return None

    # ensure shortest path
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # if nearly identical, use linear interp
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return safe_quat(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = (s0 * q0) + (s1 * q1)
    return safe_quat(q)


class EKFPosVel:
    """
    EKF / KF for 3D position + velocity state:
      x = [px, py, pz, vx, vy, vz]^T

    Linear constant-velocity model with discrete-time Q.
    """

    def __init__(self, dt=0.01,
                 x0=None,
                 P0=None,
                 process_std_pos=1e-3,
                 process_std_vel=1e-2):
        self.state_dim = 6
        self.dt = dt

        if x0 is None:
            self.x = np.zeros((6, 1))
        else:
            self.x = np.asarray(x0).reshape(6, 1)

        if P0 is None:
            self.P = np.eye(6) * 1.0
        else:
            self.P = np.asarray(P0).reshape(6, 6)

        self.base_process_std_pos = process_std_pos
        self.base_process_std_vel = process_std_vel
        self.Q = self._compute_Q(self.dt,
                                 self.base_process_std_pos,
                                 self.base_process_std_vel)

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        self.H_pos = np.block([I3, Z3])   # (3,6)
        self.H_vel = np.block([Z3, I3])   # (3,6)
        self.H_both = np.block([[I3, Z3],
                                [Z3, I3]]) # (6,6)
        self.I = np.eye(6)

    def _F(self, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _compute_Q(self, dt, std_pos, std_vel):
        q_pos = (std_pos ** 2) * dt
        q_vel = (std_vel ** 2) * dt
        Q = np.zeros((6, 6))
        Q[0, 0] = q_pos
        Q[1, 1] = q_pos
        Q[2, 2] = q_pos
        Q[3, 3] = q_vel
        Q[4, 4] = q_vel
        Q[5, 5] = q_vel
        return Q

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
        F = self._F(dt)
        self.Q = self._compute_Q(dt,
                                 self.base_process_std_pos,
                                 self.base_process_std_vel)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, R_cov, meas_type='both'):
        if meas_type == 'pos':
            H = self.H_pos
        elif meas_type == 'vel':
            H = self.H_vel
        elif meas_type == 'both':
            H = self.H_both
        else:
            raise ValueError("meas_type must be 'pos', 'vel' or 'both'")

        z = np.asarray(z).reshape((-1, 1))
        S = H @ self.P @ H.T + np.asarray(R_cov)
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - (H @ self.x)
        self.x = self.x + K @ y
        KH = K @ H
        self.P = (self.I - KH) @ self.P @ (self.I - KH).T + K @ R_cov @ K.T

    def get_state(self):
        return self.x.flatten()

    def set_state(self, x, P=None):
        self.x = np.asarray(x).reshape(6, 1)
        if P is not None:
            self.P = np.asarray(P).reshape(6, 6)


class MocapPoseToOdom(Node):
    def __init__(self):
        super().__init__("mocap_to_odom")

        self.sub = self.create_subscription(
            RigidBodies, "/rigid_bodies", self.rigid_bodies_callback, 10
        )
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.pub1 = self.create_publisher(Odometry, "/odom1_vison", 10)
        self.body_name_to_index = {"slider2.slider2": 0,                # /odom1
                                 "dockingstation.dockingstation": 1,  # /odom
                                  }

        # Low-pass filters for linear and angular velocity
        self.vel_sos = butter(
            MocapCfg.lpf_order,
            MocapCfg.lpf_cutoff_hz,
            btype="low",
            fs=MocapCfg.mocap_rate_hz,
            output="sos",
        )

        self.ang_sos = butter(
            MocapCfg.lpf_order,
            MocapCfg.lpf_cutoff_hz,
            btype="low",
            fs=MocapCfg.mocap_rate_hz,
            output="sos",
        )

        # Orientation smoothing (SLERP “EMA”)
        self.ori_cutoff_hz = float(getattr(MocapCfg, "ori_cutoff_hz", 8.0))

        # Extra smoothing on finite-difference velocity using 1st-order EMA equivalent cutoff
        self.vel_ema_cutoff_hz = float(getattr(MocapCfg, "vel_ema_cutoff_hz", 0.0))

        # Dropout handling
        self.max_dt = float(getattr(MocapCfg, "max_dt", 0.2))  # seconds

        # Whether to use EKF for position/velocity estimation
        self.use_ekf_pos_vel = bool(getattr(MocapCfg, "use_ekf_pos_vel", False))

        # Whether EKF should ingest measured velocity or use position-only update
        self.use_velocity_measurement = bool(getattr(MocapCfg, "use_velocity_measurement", True))

        # EKF configuration - fallback defaults if not in MocapCfg
        proc_std_pos = float(getattr(MocapCfg, "ekf_proc_std_pos", 1e-3))
        proc_std_vel = float(getattr(MocapCfg, "ekf_proc_std_vel", 1e-2))
        self.ekf = EKFPosVel(dt=1.0 / float(getattr(MocapCfg, "mocap_rate_hz", 200.0)),
                             process_std_pos=proc_std_pos,
                             process_std_vel=proc_std_vel)

        # Measurement covariances (user may set MocapCfg.pos_cov, MocapCfg.vel_cov as scalars or 3x3)
        pos_cov = getattr(MocapCfg, "pos_cov", None)
        if pos_cov is None:
            self.R_pos = np.eye(3) * 0.001
        else:
            pc = np.asarray(pos_cov)
            if pc.size == 1:
                self.R_pos = np.eye(3) * float(pc)
            else:
                self.R_pos = pc.reshape(3, 3)

        vel_cov = getattr(MocapCfg, "vel_cov", None)
        if vel_cov is None:
            self.R_vel = np.eye(3) * 0.01
        else:
            vc = np.asarray(vel_cov)
            if vc.size == 1:
                self.R_vel = np.eye(3) * float(vc)
            else:
                self.R_vel = vc.reshape(3, 3)

        # Combined measurement covariance for pos+vel
        self.R_both = np.block([[self.R_pos, np.zeros((3, 3))],
                                [np.zeros((3, 3)), self.R_vel]])

        # If you want to publish the raw mocap pose as well, keep an option:
        self.publish_raw = bool(getattr(MocapCfg, "publish_raw", False))

        # Per-rigid-body processing states (body0->/odom, body1->/odom1)
        self.max_bodies = 2
        self.body_states = [self._init_body_state() for _ in range(self.max_bodies)]
        self.body_publishers = [self.pub, self.pub1]

    def _init_body_state(self):
        state = {
            "pos_buffer": [deque(maxlen=MocapCfg.buffer_size) for _ in range(3)],
            "t_prev": None,
            "p_prev_filt": None,
            "q_prev_filt": None,
            "v_prev_ema": None,
            "vel_zi": np.repeat(sosfilt_zi(self.vel_sos)[:, :, None], 3, axis=2) * 0.0,
            "ang_zi": np.repeat(sosfilt_zi(self.ang_sos)[:, :, None], 3, axis=2) * 0.0,
            "ekf": EKFPosVel(
                dt=1.0 / float(getattr(MocapCfg, "mocap_rate_hz", 200.0)),
                process_std_pos=float(getattr(MocapCfg, "ekf_proc_std_pos", 1e-3)),
                process_std_vel=float(getattr(MocapCfg, "ekf_proc_std_vel", 1e-2)),
            ),
        }
        return state

    def _reset_body_state(self, state):
        state["t_prev"] = None
        state["p_prev_filt"] = None
        state["q_prev_filt"] = None
        state["v_prev_ema"] = None
        state["vel_zi"] *= 0.0
        state["ang_zi"] *= 0.0
        for i in range(3):
            state["pos_buffer"][i].clear()
    def rigid_bodies_callback(self, msg: RigidBodies):
        if not msg.rigidbodies:
            for state in self.body_states:
                self._reset_body_state(state)
            return

        t = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )

        seen_indices = set()

        for rb in msg.rigidbodies:
            body_index = self.body_name_to_index.get(rb.rigid_body_name)

            # Ignore unconfigured rigid bodies
            if body_index is None:
                self.get_logger().debug(
                    f"Ignoring rigid body: {rb.rigid_body_name}"
                )
                continue

            # Protect against duplicate names in one message
            if body_index in seen_indices:
                self.get_logger().warning(
                    f"Duplicate rigid body in frame: {rb.rigid_body_name}"
                )
                continue

            seen_indices.add(body_index)
            self._process_rigid_body(rb, msg, t, body_index)


    def _process_rigid_body(self, rb, msg, t: float, body_index: int):
        state = self.body_states[body_index]

        # Position median smoothing
        p = np.array(
            [rb.pose.position.x, rb.pose.position.y, rb.pose.position.z], dtype=float
        )
        for i in range(3):
            state["pos_buffer"][i].append(p[i])
        p_filt = np.array([np.median(state["pos_buffer"][i]) for i in range(3)], dtype=float)

        # Orientation (SLERP smoothing)
        q_meas = np.array(
            [
                rb.pose.orientation.x,
                rb.pose.orientation.y,
                rb.pose.orientation.z,
                rb.pose.orientation.w,
            ],
            dtype=float,
        )
        q_meas = safe_quat(q_meas)
        if q_meas is None:
            return

        if state["t_prev"] is None:
            # initialize previouss and EKF state with first measurement
            state["t_prev"] = t
            state["p_prev_filt"] = p_filt
            state["q_prev_filt"] = q_meas
            state["v_prev_ema"] = np.zeros(3, dtype=float)

            if self.use_ekf_pos_vel:
                # Initialize EKF state: position from filt pos, velocity zero (large P for velocity)
                x0 = np.hstack([p_filt, np.zeros(3)])
                P0 = np.eye(6)
                P0[0:3, 0:3] *= 0.01  # small pos uncertainty
                P0[3:6, 3:6] *= 1.0   # larger vel uncertainty
                state["ekf"].set_state(x0, P0)
            return

        dt = t - state["t_prev"]
        if dt <= 0.0:
            # ignore bad timestamp
            return

        # Big gap -> reset filters and EKF to avoid spikes
        if dt > self.max_dt:
            state["t_prev"] = t
            state["p_prev_filt"] = p_filt
            state["q_prev_filt"] = q_meas
            state["v_prev_ema"] = np.zeros(3, dtype=float)
            state["vel_zi"] *= 0.0
            state["ang_zi"] *= 0.0
            for i in range(3):
                state["pos_buffer"][i].clear()

            if self.use_ekf_pos_vel:
                # Reset EKF state to current measurement (vel unknown -> zero)
                x0 = np.hstack([p_filt, np.zeros(3)])
                P0 = np.eye(6)
                P0[0:3, 0:3] *= 0.01
                P0[3:6, 3:6] *= 1.0
                state["ekf"].set_state(x0, P0)
            return

        # SLERP smoothing factor from cutoff: alpha = 1 - exp(-2*pi*fc*dt)
        alpha = 1.0 - float(np.exp(-2.0 * np.pi * self.ori_cutoff_hz * dt))
        q_filt = slerp_quat(state["q_prev_filt"], q_meas, alpha)
        if q_filt is None:
            return

        # Linear velocity (from filtered position)
        v = (p_filt - state["p_prev_filt"]) / dt
        v_out, state["vel_zi"] = sosfilt(self.vel_sos, v[None, :], axis=0, zi=state["vel_zi"])
        v = v_out[0]

        # Optional extra EMA smoothing on velocity to reduce oscillations
        if self.vel_ema_cutoff_hz > 0.0:
            alpha_v = 1.0 - float(np.exp(-2.0 * np.pi * self.vel_ema_cutoff_hz * dt))
            if state["v_prev_ema"] is None:
                state["v_prev_ema"] = v.copy()
            state["v_prev_ema"] = state["v_prev_ema"] + alpha_v * (v - state["v_prev_ema"])
            v = state["v_prev_ema"].copy()

        # Angular velocity from relative rotation of filtered quats
        rot_prev = R.from_quat(state["q_prev_filt"])
        rot_curr = R.from_quat(q_filt)
        rel = rot_prev.inv() * rot_curr
        w = rel.as_rotvec() / dt
        #rot_prev = R.from_quat(self.q_prev_filt)
        #rot_curr = R.from_quat(q_filt)
        #rel = rot_curr * rot_prev.inv()
        #w = rel.as_rotvec() / dt  # rad/s
        w_out, state["ang_zi"] = sosfilt(self.ang_sos, w[None, :], axis=0, zi=state["ang_zi"])
        w = w_out[0]

        if self.use_ekf_pos_vel:
            # --- EKF predict & update ---
            # Predict using measured dt
            state["ekf"].predict(dt=dt)

            if self.use_velocity_measurement:
                z = np.hstack([p_filt, v])
                meas_type = 'both'
                R_cov = self.R_both
            else:
                z = p_filt
                meas_type = 'pos'
                R_cov = self.R_pos
            try:
                state["ekf"].update(z, R_cov, meas_type=meas_type)
            except np.linalg.LinAlgError:
                # numerical issue: skip this update
                self.get_logger().warning("EKF update failed due to numerical issue; skipping update")
            # Retrieve EKF state for publishing
            ekf_state = state["ekf"].get_state()
            px, py, pz, vx, vy, vz = ekf_state.tolist()
        else:
            # Publish filtered mocap position and filtered finite-difference velocity directly
            px, py, pz = p_filt.tolist()
            vx, vy, vz = v.tolist()

        # Build Odometry
        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = MocapCfg.frame
        odom.child_frame_id = "base_link" if body_index == 0 else f"base_link{body_index}"

        # Pose from EKF (position) and SLERP orientation
        odom.pose.pose.position.x = float(px)
        odom.pose.pose.position.y = float(py)
        odom.pose.pose.position.z = float(pz)

        odom.pose.pose.orientation.x = float(q_filt[0])
        odom.pose.pose.orientation.y = float(q_filt[1])
        odom.pose.pose.orientation.z = float(q_filt[2])
        odom.pose.pose.orientation.w = float(q_filt[3])

        # Twist: linear from EKF, angular from filtered angular velocity
        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.linear.z = float(vz)

        odom.twist.twist.angular.x = float(w[0])
        odom.twist.twist.angular.y = float(w[1])
        odom.twist.twist.angular.z = float(w[2])

        # Publish
        self.body_publishers[body_index].publish(odom)

        # Update state
        state["t_prev"] = t
        state["p_prev_filt"] = p_filt
        state["q_prev_filt"] = q_filt

    def destroy_node(self):
        # optional: if you want to perform cleanup, override
        super().destroy_node()


def main():
    rclpy.init()
    node = MocapPoseToOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()