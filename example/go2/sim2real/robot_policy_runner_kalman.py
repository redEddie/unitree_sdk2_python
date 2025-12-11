#!/usr/bin/env python3
"""
Unitree Go2 Policy Runner (Kalman velocity)
===========================================

Runs the flat-ground IsaacLab policy on the real Go2, estimating base linear
velocity with a lightweight Kalman filter that fuses IMU acceleration (with
gravity compensation) and SportModeState velocity when available.

Usage:
  python3 robot_policy_runner_kalman.py [network_interface] [policy_path]
    network_interface: e.g., eth0 (default: "default" as in unitree examples)
    policy_path: path to ONNX (default: ./exported/policy.onnx)

Safety:
  - Disables Sport mode (MCF) before control and restores it on exit.
  - Sends zero torques on exit/interrupt.
"""

import os
import sys
import time
from typing import Optional, Tuple
from collections import deque

import numpy as np
import onnxruntime as ort

from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

from utils import unitree_communication as comm
from utils import unitree_joint_config as joint_cfg
from utils import unitree_mode_manager as mode_mgr
from utils import unitree_observation_builder as obs_builder
from utils import unitree_imu_utils as imu_utils


# =============================================================================
# Simple Kalman utilities
# =============================================================================

class MovingAverage:
    """Simple moving average smoother."""

    def __init__(self, window_size: int = 20):
        self.values = deque(maxlen=window_size)

    def reset(self):
        self.values.clear()

    def update(self, value: np.ndarray) -> np.ndarray:
        self.values.append(np.asarray(value, dtype=np.float32))
        return self.average

    @property
    def average(self) -> np.ndarray:
        if not self.values:
            return np.zeros(3, dtype=np.float32)
        return np.mean(np.stack(self.values, axis=0), axis=0)


class SimpleKalman3D:
    """Minimal Kalman filter for body-frame velocity."""

    def __init__(self, process_var: float, meas_var: float, init_var: float):
        self.process_var = process_var
        self.meas_var = meas_var
        self.init_var = init_var

        self.x = np.zeros(3, dtype=np.float32)
        self.P = np.eye(3, dtype=np.float32) * init_var
        self.Q = np.eye(3, dtype=np.float32) * process_var
        self.R = np.eye(3, dtype=np.float32) * meas_var
        self.H = np.eye(3, dtype=np.float32)

    def predict(self, u: np.ndarray, dt: float):
        # x = x + dt * u (u is acceleration in body frame)
        self.x = self.x + dt * u
        self.P = self.P + self.Q

    def update(self, z: np.ndarray):
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(3, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


class VelocityKalmanEstimator:
    """Fuses IMU accel + SportMode velocity to estimate base linear velocity."""

    def __init__(self,
                 process_var: float = 0.2,
                 meas_var: float = 0.05,
                 init_var: float = 0.5,
                 smooth_window: int = 20):
        self.filter = SimpleKalman3D(process_var, meas_var, init_var)
        self.smoother = MovingAverage(window_size=smooth_window)

        self.err_sum = np.zeros(3, dtype=np.float64)
        self.err_sq_sum = np.zeros(3, dtype=np.float64)
        self.err_count = 0

    def reset(self):
        self.filter = SimpleKalman3D(self.filter.process_var,
                                     self.filter.meas_var,
                                     self.filter.init_var)
        self.smoother.reset()
        self.err_sum.fill(0.0)
        self.err_sq_sum.fill(0.0)
        self.err_count = 0

    @staticmethod
    def _gravity_body(quat: np.ndarray) -> np.ndarray:
        R_w_b = imu_utils.quaternion_to_rotation_matrix(quat, direction='world_to_body')
        return R_w_b @ np.array([0.0, 0.0, -9.81], dtype=np.float32)

    def step(self,
             imu_accel: np.ndarray,
             quat: np.ndarray,
             dt: float,
             measurement: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict/update; returns (smoothed_estimate, raw_estimate)."""
        if dt <= 0:
            return self.smoother.average, self.filter.x.copy()

        accel_body = imu_accel + self._gravity_body(quat)
        self.filter.predict(accel_body, dt)

        if measurement is not None:
            prior = self.filter.x.copy()
            self.filter.update(measurement)
            err = prior - measurement
            self.err_sum += err
            self.err_sq_sum += err * err
            self.err_count += 1

        est = self.filter.x.copy()
        smooth = self.smoother.update(est)
        return smooth, est

    def status_string(self) -> str:
        if self.err_count == 0:
            return "Kalman stats: waiting for measurements..."
        mean_err = self.err_sum / self.err_count
        rmse = np.sqrt(self.err_sq_sum / self.err_count)
        return (f"Kalman prior error mean [m/s]: "
                f"{mean_err[0]: .3f}, {mean_err[1]: .3f}, {mean_err[2]: .3f} | "
                f"RMSE [m/s]: {rmse[0]: .3f}, {rmse[1]: .3f}, {rmse[2]: .3f}")


# =============================================================================
# Runner
# =============================================================================

class PolicyConfig:
    """Policy configuration matching IsaacLab training setup."""
    CONTROL_DT = 0.02
    ENABLE_NOISE = False
    ACTION_SCALE = 0.25
    KP = 25.0
    KD = 0.5
    ACTION_CLIP = 23.5


class PolicyRunner:
    """Policy runner using Kalman-estimated base velocity."""

    def __init__(self, policy_path: str, config: PolicyConfig, network_interface: str = "default"):
        self.config = config
        self.network_interface = network_interface
        self.crc = CRC()

        # Load ONNX
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        print(f"\nLoading ONNX policy from: {policy_path}")
        self.ort_session = ort.InferenceSession(policy_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        print("✓ Policy loaded")

        # Observation builder
        obs_config = obs_builder.ObservationConfig(enable_noise=config.ENABLE_NOISE)
        self.obs_builder = obs_builder.ObservationBuilder(obs_config)
        self.obs_builder.set_use_imu_integration(False)  # we provide velocity_override

        # Kalman estimator
        self.vel_estimator = VelocityKalmanEstimator()
        self.last_loop_ts = None

        # Comm
        self.latest_lowstate: Optional[LowState_] = None
        self.latest_sportmode: Optional[SportModeState_] = None
        self._setup_communication()

        # Mode manager
        self.mode_manager = mode_mgr.UnitreeModeManager()

        # Stats
        self.step_count = 0
        self.total_inference_time = 0.0
        self.total_loop_time = 0.0

        # Stand-up posture (Unitree order)
        self.stand_up_joint_pos = np.array([
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
        ], dtype=np.float32)
        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ], dtype=np.float32)

    def _setup_communication(self):
        comm.initialize_channel_factory(self.network_interface)
        self.state_sub = comm.create_lowstate_subscriber(self._lowstate_handler, 10)
        self.sportmode_sub = comm.create_sportmode_subscriber(self._sportmode_handler, 10)
        self.cmd_pub = comm.create_lowcmd_publisher()
        print("✓ Communication channels initialized")

    def _lowstate_handler(self, msg: LowState_):
        self.latest_lowstate = msg

    def _sportmode_handler(self, msg: SportModeState_):
        self.latest_sportmode = msg

    def wait_for_robot_state(self, timeout: float = 5.0):
        lowstate_ref = [self.latest_lowstate]
        sportmode_ref = [self.latest_sportmode]
        comm.wait_for_robot_data(lowstate_ref, sportmode_ref, timeout=timeout, verbose=True)
        self.latest_lowstate = lowstate_ref[0]
        self.latest_sportmode = sportmode_ref[0]
        if self.latest_sportmode is None:
            print("  SportModeState not available; Kalman will rely on IMU only.")

    def prepare_robot(self) -> bool:
        return self.mode_manager.disable_sport_mode()

    def restore_robot(self):
        self.mode_manager.restore_sport_mode()

    def execute_policy(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        obs_batch = obs.reshape(1, -1)
        outputs = self.ort_session.run([self.output_name], {self.input_name: obs_batch})
        actions_raw = outputs[0][0]
        inf_time = (time.perf_counter() - start) * 1000
        return actions_raw.astype(np.float32), inf_time

    def _compute_torques(self, target_pos: np.ndarray, current_pos: np.ndarray,
                        current_vel: np.ndarray) -> np.ndarray:
        pos_err = target_pos - current_pos
        vel_err = -current_vel
        tau = self.config.KP * pos_err + self.config.KD * vel_err
        return np.clip(tau, -self.config.ACTION_CLIP, self.config.ACTION_CLIP)

    def create_torque_command(self, target_positions_unitree: np.ndarray,
                             current_positions_unitree: np.ndarray,
                             current_velocities_unitree: np.ndarray) -> unitree_go_msg_dds__LowCmd_:
        torques = self._compute_torques(target_positions_unitree,
                                       current_positions_unitree,
                                       current_velocities_unitree)
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = float(torques[i])
        for i in range(12, 20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def _create_position_command(self, target_positions_unitree: np.ndarray,
                                 kp: float = 50.0, kd: float = 3.5) -> unitree_go_msg_dds__LowCmd_:
        """Position command helper for stand-up sequence."""
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = float(target_positions_unitree[i])
            cmd.motor_cmd[i].kp = kp
            cmd.motor_cmd[i].kd = kd
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0

        for i in range(12, 20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def _create_zero_command(self) -> unitree_go_msg_dds__LowCmd_:
        zeros = unitree_go_msg_dds__LowCmd_()
        zeros.head[0] = 0xFE
        zeros.head[1] = 0xEF
        zeros.level_flag = 0xFF
        zeros.gpio = 0
        for i in range(20):
            zeros.motor_cmd[i].mode = 0x01
            zeros.motor_cmd[i].q = 0.0
            zeros.motor_cmd[i].kp = 0.0
            zeros.motor_cmd[i].kd = 0.0
            zeros.motor_cmd[i].dq = 0.0
            zeros.motor_cmd[i].tau = 0.0
        zeros.crc = self.crc.Crc(zeros)
        return zeros

    def _stand_sequence(self, rise_time: float = 1.2, hold_time: float = 1.0):
        """Execute a quick stand-up sequence before policy control."""
        print("\nStarting stand-up sequence...")
        start = time.perf_counter()
        while True:
            t = time.perf_counter() - start
            if t >= rise_time + hold_time:
                break

            if t < rise_time:
                phase = np.tanh(t / rise_time)
            else:
                phase = 1.0

            target_q = phase * self.stand_up_joint_pos + (1.0 - phase) * self.stand_down_joint_pos
            cmd = self._create_position_command(target_q)
            self.cmd_pub.Write(cmd)
            time.sleep(self.config.CONTROL_DT)

        # Hold final posture a little longer
        for _ in range(20):
            cmd = self._create_position_command(self.stand_up_joint_pos)
            self.cmd_pub.Write(cmd)
            time.sleep(self.config.CONTROL_DT)
        print("Stand-up complete.")

    def _compute_velocities(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (policy_vel, kalman_smooth, sport_vel)."""
        imu_accel = np.array(self.latest_lowstate.imu_state.accelerometer, dtype=np.float32)
        quat = np.array(self.latest_lowstate.imu_state.quaternion, dtype=np.float32)

        sport_vel = None
        if self.latest_sportmode is not None:
            sport_vel = np.array([
                float(self.latest_sportmode.velocity[0]),
                float(self.latest_sportmode.velocity[1]),
                float(self.latest_sportmode.velocity[2])
            ], dtype=np.float32)

        kalman_smooth, _ = self.vel_estimator.step(imu_accel, quat, dt, sport_vel)

        # Policy velocity: prefer sport if present, otherwise Kalman; switch to Kalman only mode by changing here.
        if sport_vel is not None:
            policy_vel = kalman_smooth  # fuse via KF but still use smooth estimate
        else:
            policy_vel = kalman_smooth

        return policy_vel.astype(np.float32), kalman_smooth.astype(np.float32), sport_vel

    def run(self, duration: Optional[float] = None):
        print("\n" + "="*70)
        print("Starting Policy Execution with Kalman Velocity")
        print("="*70)
        print(f"Control frequency: {1.0/self.config.CONTROL_DT:.1f} Hz")
        print(f"Network interface: {self.network_interface}")
        print("="*70)

        self.wait_for_robot_state()
        self.obs_builder.set_velocity_commands(0.5, 0.0, 0.0)

        if not self.prepare_robot():
            print("❌ Failed to prepare robot (Sport mode disable)")
            return False

        # Stand up before switching to torque-based policy control
        self._stand_sequence()

        print("\nControl loop active. Press Ctrl+C to stop.")
        start_time = time.time()
        last_print = start_time

        try:
            while True:
                loop_start = time.perf_counter()

                if duration and (time.time() - start_time) > duration:
                    break

                if self.latest_lowstate is None:
                    zero_cmd = self._create_zero_command()
                    self.cmd_pub.Write(zero_cmd)
                    time.sleep(0.001)
                    continue

                dt = self.config.CONTROL_DT if self.last_loop_ts is None else (loop_start - self.last_loop_ts)
                self.last_loop_ts = loop_start

                policy_vel, kalman_vel, sport_vel = self._compute_velocities(dt)

                obs = self.obs_builder.build_observation(
                    lowstate=self.latest_lowstate,
                    sportmode=None,             # we override velocity
                    velocity_override=policy_vel,
                    add_noise=self.config.ENABLE_NOISE
                )

                actions_raw_isaaclab, inf_time = self.execute_policy(obs)
                actions_scaled = actions_raw_isaaclab * self.config.ACTION_SCALE
                target_pos_isaac = actions_scaled + joint_cfg.DEFAULT_JOINT_POS_ISAACLAB
                target_pos_unitree = joint_cfg.isaaclab_to_unitree(target_pos_isaac)

                self.obs_builder.update_last_actions(actions_raw_isaaclab)

                current_pos_unitree = np.array([ms.q for ms in self.latest_lowstate.motor_state[:12]], dtype=np.float32)
                current_vel_unitree = np.array([ms.dq for ms in self.latest_lowstate.motor_state[:12]], dtype=np.float32)

                cmd = self.create_torque_command(target_pos_unitree,
                                                 current_pos_unitree,
                                                 current_vel_unitree)
                self.cmd_pub.Write(cmd)

                # stats
                loop_time = (time.perf_counter() - loop_start) * 1000
                self.step_count += 1
                self.total_inference_time += inf_time
                self.total_loop_time += loop_time

                now = time.time()
                if now - last_print >= 0.2:
                    elapsed = now - start_time
                    avg_inf = self.total_inference_time / self.step_count
                    avg_loop = self.total_loop_time / self.step_count
                    freq = self.step_count / elapsed
                    sport_str = sport_vel if sport_vel is not None else np.zeros(3, dtype=np.float32)
                    print(f"[{elapsed:6.1f}s] step {self.step_count:5d} | freq {freq:5.1f} Hz | "
                          f"inf {avg_inf:4.2f} ms | loop {avg_loop:4.2f} ms")
                    print(f"      vel(policy): {policy_vel} | kalman: {kalman_vel} | sport: {sport_str}")
                    print("      " + self.vel_estimator.status_string())
                    last_print = now

                # timing
                elapsed = time.perf_counter() - loop_start
                sleep = self.config.CONTROL_DT - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt - stopping robot...")

        finally:
            zero_cmd = self._create_zero_command()
            for _ in range(10):
                self.cmd_pub.Write(zero_cmd)
                time.sleep(0.01)

            if self.step_count > 0:
                total_time = time.time() - start_time
                print("\nExecution stats:")
                print(f"  steps      : {self.step_count}")
                print(f"  time       : {total_time:.1f}s")
                print(f"  freq       : {self.step_count/total_time:.1f} Hz")
                print(f"  inf avg    : {self.total_inference_time/self.step_count:.2f} ms")
                print(f"  loop avg   : {self.total_loop_time/self.step_count:.2f} ms")

            self.restore_robot()
            print("✓ Robot stopped and Sport mode restored")

        return True


# =============================================================================
# Entry point
# =============================================================================

def main():
    print("="*70)
    print("Unitree Go2 Policy Runner (Kalman velocity)")
    print("="*70)

    if len(sys.argv) >= 2:
        network_interface = sys.argv[1]
    else:
        network_interface = "default"
        print("⚠️  No network interface specified, using default")

    if len(sys.argv) >= 3:
        policy_path = sys.argv[2]
    else:
        policy_path = "./model_flat/exported/policy.onnx"
        print(f"⚠️  No policy path specified, using default: {policy_path}")

    if not os.path.exists(policy_path):
        print(f"\n❌ Policy file not found: {policy_path}")
        return 1

    # Safety prompt
    print("\n" + "⚠️ "*10)
    print("WARNING: This will control the REAL ROBOT.")
    print("Ensure the area is clear and E-stop is ready.")
    print("⚠️ "*10)
    resp = input("Type 'yes' to continue: ")
    if resp.lower() != "yes":
        print("Aborted.")
        return 0

    cfg = PolicyConfig()
    try:
        runner = PolicyRunner(policy_path, cfg, network_interface)
        ok = runner.run(duration=1.5)
        return 0 if ok else 1
    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
