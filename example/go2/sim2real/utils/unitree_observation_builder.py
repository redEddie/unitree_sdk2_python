"""
Unitree Go2 Observation Builder
================================

Standardized observation processing for policy deployment.

This module provides a consistent interface for building observation
vectors from robot sensor data, matching the format expected by
IsaacLab-trained policies.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

from . import unitree_joint_config as joint_cfg
from . import unitree_imu_utils as imu_utils


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    # Observation clipping
    obs_clip: float = 100.0

    # Noise configuration (typically disabled for real robot)
    enable_noise: bool = False
    noise_levels: Dict[str, tuple] = None

    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = {
                'base_lin_vel': (-0.1, 0.1),
                'base_ang_vel': (-0.2, 0.2),
                'projected_gravity': (-0.05, 0.05),
                'joint_pos': (-0.01, 0.01),
                'joint_vel': (-1.5, 1.5),
            }


class ObservationBuilder:
    """
    Build observation vectors from robot sensor data.

    This class processes raw sensor data (LowState, SportModeState) into
    observation vectors suitable for policy inference.
    """

    def __init__(self, config: Optional[ObservationConfig] = None):
        """
        Initialize observation builder.

        Args:
            config: Observation configuration
        """
        self.config = config or ObservationConfig()

        # IMU velocity integrator (fallback when SportModeState unavailable)
        self.imu_integrator = imu_utils.IMUVelocityIntegrator(decay_factor=0.99)
        self.use_imu_integration = False

        # Last actions (for action history in observation)
        self.last_actions = np.zeros(12, dtype=np.float32)

        # Velocity commands
        self.velocity_commands = np.zeros(3, dtype=np.float32)
        self.vel_command_b = np.zeros(3, dtype=np.float32)

        # Heading control
        self.heading_command = True
        self.heading_control_stiffness = 0.5
        self.heading_target = 0.0

    def set_use_imu_integration(self, use_imu: bool):
        """
        Enable or disable IMU integration for base velocity estimation.

        Args:
            use_imu: True to use IMU integration, False to use SportModeState
        """
        self.use_imu_integration = use_imu
        if use_imu:
            self.imu_integrator.reset()

    def set_velocity_commands(self, vx: float, vy: float, yaw: float):
        """
        Set velocity commands.

        Args:
            vx: Forward velocity command (m/s)
            vy: Lateral velocity command (m/s)
            yaw: Yaw rate or heading target (rad/s or rad)
        """
        self.velocity_commands = np.array([vx, vy, yaw], dtype=np.float32)
        self.vel_command_b[0] = vx
        self.vel_command_b[1] = vy

        if self.heading_command:
            self.heading_target = yaw
        else:
            self.vel_command_b[2] = yaw

    def update_last_actions(self, actions: np.ndarray):
        """
        Update last actions (for action history in observation).

        Args:
            actions: Action vector (12,) in IsaacLab order
        """
        self.last_actions = actions.copy()

    def process_lowstate(self, lowstate: LowState_) -> Dict[str, np.ndarray]:
        """
        Extract all components from LowState.

        Args:
            lowstate: LowState message from robot

        Returns:
            Dictionary of observation components
        """
        components = {}

        # IMU data
        quat = np.array(lowstate.imu_state.quaternion, dtype=np.float32)
        components['quaternion'] = quat
        components['base_ang_vel'] = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
        components['imu_accel'] = np.array(lowstate.imu_state.accelerometer, dtype=np.float32)

        # Projected gravity
        components['projected_gravity'] = imu_utils.compute_projected_gravity(quat)

        # Joint data (Unitree -> IsaacLab order)
        joint_pos_unitree = np.array([ms.q for ms in lowstate.motor_state[:12]], dtype=np.float32)
        joint_vel_unitree = np.array([ms.dq for ms in lowstate.motor_state[:12]], dtype=np.float32)

        components['joint_pos'] = joint_cfg.unitree_to_isaaclab(joint_pos_unitree)
        components['joint_vel'] = joint_cfg.unitree_to_isaaclab(joint_vel_unitree)

        # Relative joint positions
        components['joint_pos_rel'] = joint_cfg.get_relative_joint_positions(
            components['joint_pos'], order='isaaclab'
        )

        return components

    def build_observation(
        self,
        lowstate: LowState_,
        sportmode: Optional[SportModeState_] = None,
        velocity_override: Optional[np.ndarray] = None,
        add_noise: bool = False
    ) -> np.ndarray:
        """
        Build complete observation vector.

        Args:
            lowstate: LowState message
            sportmode: Optional SportModeState message (for base velocity)
            velocity_override: Optional manual velocity override (3,)
            add_noise: Whether to add observation noise

        Returns:
            Observation vector (48,) in IsaacLab format:
            [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
             vel_command_b(3), joint_pos_rel(12), joint_vel(12), last_actions(12)]
        """
        # Process lowstate
        components = self.process_lowstate(lowstate)

        # Base linear velocity
        if velocity_override is not None:
            base_lin_vel = velocity_override.copy()
        elif sportmode is not None and not self.use_imu_integration:
            base_lin_vel = np.array([
                float(sportmode.velocity[0]),
                float(sportmode.velocity[1]),
                float(sportmode.velocity[2])
            ], dtype=np.float32)
        elif self.use_imu_integration:
            base_lin_vel = self.imu_integrator.update(
                components['imu_accel'],
                components['quaternion']
            )
        else:
            # Default: zero velocity
            base_lin_vel = np.zeros(3, dtype=np.float32)

        # Update velocity commands based on heading mode
        if self.heading_command:
            self._update_vel_command_b(components['quaternion'])

        # Get components
        base_ang_vel = components['base_ang_vel']
        projected_gravity = components['projected_gravity']
        joint_pos_rel = components['joint_pos_rel']
        joint_vel = components['joint_vel']

        # Add noise if requested
        if add_noise and self.config.enable_noise:
            base_lin_vel = self._add_noise(base_lin_vel, 'base_lin_vel')
            base_ang_vel = self._add_noise(base_ang_vel, 'base_ang_vel')
            projected_gravity = self._add_noise(projected_gravity, 'projected_gravity')
            joint_pos_rel = self._add_noise(joint_pos_rel, 'joint_pos')
            joint_vel = self._add_noise(joint_vel, 'joint_vel')

        # Build observation vector
        obs = np.concatenate([
            base_lin_vel,           # (3,)
            base_ang_vel,           # (3,)
            projected_gravity,      # (3,)
            self.vel_command_b,     # (3,)
            joint_pos_rel,          # (12,)
            joint_vel,              # (12,)
            self.last_actions       # (12,)
        ]).astype(np.float32)

        # Clip observation
        obs = np.clip(obs, -self.config.obs_clip, self.config.obs_clip)

        return obs

    def validate_observation(self, obs: np.ndarray) -> bool:
        """
        Validate observation vector shape and values.

        Args:
            obs: Observation vector

        Returns:
            True if valid
        """
        if obs.shape != (48,):
            return False
        if not np.all(np.isfinite(obs)):
            return False
        return True

    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get metadata about observation components.

        Returns:
            Dictionary with observation structure information
        """
        return {
            'total_dim': 48,
            'components': {
                'base_lin_vel': {'start': 0, 'end': 3, 'dim': 3},
                'base_ang_vel': {'start': 3, 'end': 6, 'dim': 3},
                'projected_gravity': {'start': 6, 'end': 9, 'dim': 3},
                'vel_command_b': {'start': 9, 'end': 12, 'dim': 3},
                'joint_pos_rel': {'start': 12, 'end': 24, 'dim': 12},
                'joint_vel': {'start': 24, 'end': 36, 'dim': 12},
                'last_actions': {'start': 36, 'end': 48, 'dim': 12},
            },
            'joint_order': 'isaaclab',
            'use_imu_integration': self.use_imu_integration,
        }

    def _update_vel_command_b(self, quaternion: np.ndarray):
        """Update velocity commands based on heading mode."""
        if self.heading_command:
            current_heading = imu_utils.compute_heading_from_quaternion(quaternion)
            heading_error = imu_utils.wrap_angle_to_pi(self.heading_target - current_heading)
            angular_velocity = self.heading_control_stiffness * heading_error
            self.vel_command_b[2] = np.clip(angular_velocity, -1.0, 1.0)

    def _add_noise(self, data: np.ndarray, noise_type: str) -> np.ndarray:
        """Add uniform noise to data."""
        if noise_type not in self.config.noise_levels:
            return data

        n_min, n_max = self.config.noise_levels[noise_type]
        noise = np.random.uniform(n_min, n_max, size=data.shape).astype(np.float32)
        return data + noise
