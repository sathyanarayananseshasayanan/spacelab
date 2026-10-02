from dataclasses import dataclass

@dataclass
class MocapCfg:
    frame: str = "world"
    buffer_size: int = 5

    mocap_rate_hz: float = 100.0
    lpf_order: int = 2
    lpf_cutoff_hz: float = 5.0

    # orientation smoothing (SLERP “EMA”)
    ori_cutoff_hz: float = 5.0

    # extra smoothing on finite-difference linear velocity
    vel_ema_cutoff_hz: float = 2.0

    # if False, publish filtered position/velocity directly (no EKF on pos/vel)
    use_ekf_pos_vel: bool = False

    # if False, EKF update uses only position (typically smoother velocity estimate)
    use_velocity_measurement: bool = False

    # reset if mocap dropouts create big dt
    max_dt: float = 0.2