"""
Unitree Go2 IMU Utilities
==========================

IMU data processing, transformations, and velocity estimation.
"""

import time
import numpy as np


# =============================================================================
# Quaternion and Rotation Utilities
# =============================================================================

def compute_projected_gravity(quaternion: np.ndarray) -> np.ndarray:
    """
    Compute normalized gravity vector in body frame from quaternion.

    Args:
        quaternion: Quaternion [w, x, y, z] (4,)

    Returns:
        Normalized gravity vector in body frame (3,)
    """
    w, x, y, z = quaternion

    # World-to-body rotation matrix (transpose of body-to-world)
    R_T = np.array([
        [1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)],
        [2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)],
        [2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]
    ], dtype=np.float32)

    # Normalized gravity in world frame (z-down convention)
    gravity_world_normalized = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # Transform to body frame
    gravity_body = R_T @ gravity_world_normalized

    # Ensure normalized
    gravity_body_norm = np.linalg.norm(gravity_body)
    if gravity_body_norm > 0:
        gravity_body = gravity_body / gravity_body_norm

    return gravity_body


def compute_heading_from_quaternion(quaternion: np.ndarray) -> float:
    """
    Extract heading (yaw) angle from quaternion.

    Args:
        quaternion: Quaternion [w, x, y, z] (4,)

    Returns:
        Heading angle in radians
    """
    w, x, y, z = quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(yaw)


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray,
    direction: str = 'body_to_world'
) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quaternion: Quaternion [w, x, y, z] (4,)
        direction: 'body_to_world' or 'world_to_body'

    Returns:
        Rotation matrix (3, 3)
    """
    w, x, y, z = quaternion

    if direction == 'body_to_world':
        # Body-to-world rotation matrix
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ], dtype=np.float32)
    elif direction == 'world_to_body':
        # World-to-body rotation matrix (transpose)
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)],
            [2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)],
            [2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return R


def wrap_angle_to_pi(angle: float) -> float:
    """
    Wrap angle to [-pi, pi] range.

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in [-pi, pi]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


# =============================================================================
# IMU Velocity Integration
# =============================================================================

class IMUVelocityIntegrator:
    """
    Estimate base velocity by integrating IMU acceleration.

    Note: This method accumulates drift over time. Use with caution
    and apply appropriate drift compensation strategies.
    """

    def __init__(self, decay_factor: float = 0.99):
        """
        Initialize IMU velocity integrator.

        Args:
            decay_factor: Decay factor to prevent unbounded drift (0-1)
                         1.0 = no decay, <1.0 = gradual decay to zero
        """
        self.estimated_velocity = np.zeros(3, dtype=np.float32)
        self.last_timestamp = None
        self.decay_factor = decay_factor

    def update(
        self,
        acceleration_body: np.ndarray,
        quaternion: np.ndarray
    ) -> np.ndarray:
        """
        Update velocity estimate from IMU data.

        Args:
            acceleration_body: Acceleration in body frame [x, y, z] (3,)
            quaternion: Quaternion [w, x, y, z] (4,)

        Returns:
            Estimated velocity in world frame (3,)
        """
        current_time = time.time()

        # Transform acceleration to world frame
        R = quaternion_to_rotation_matrix(quaternion, direction='body_to_world')
        accel_world = R @ acceleration_body

        # Remove gravity (assuming z-up world frame)
        # Note: IMU measures [0, 0, -9.81] when stationary, so we add 9.81 to z
        accel_world[2] += 9.81

        # Integrate
        if self.last_timestamp is not None:
            dt = current_time - self.last_timestamp

            # Simple integration with drift compensation
            self.estimated_velocity += accel_world * dt

            # Apply decay to prevent unbounded drift
            self.estimated_velocity *= self.decay_factor

        self.last_timestamp = current_time

        return self.estimated_velocity.copy()

    def reset(self):
        """Reset integrator state."""
        self.estimated_velocity = np.zeros(3, dtype=np.float32)
        self.last_timestamp = None

    def set_velocity(self, velocity: np.ndarray):
        """
        Manually set velocity (e.g., for initialization or correction).

        Args:
            velocity: Velocity in world frame (3,)
        """
        self.estimated_velocity = velocity.copy()

    def get_velocity(self) -> np.ndarray:
        """
        Get current velocity estimate.

        Returns:
            Estimated velocity in world frame (3,)
        """
        return self.estimated_velocity.copy()
