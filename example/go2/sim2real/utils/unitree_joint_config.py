"""
Unitree Go2 Joint Configuration
================================

Joint index mappings and default positions for converting between
Unitree SDK order and IsaacLab order.

Joint Ordering:
- Unitree/MuJoCo: FR, FL, RR, RL (front-right, front-left, rear-right, rear-left)
- IsaacLab: FL, FR, RL, RR (front-left, front-right, rear-left, rear-right)
"""

import numpy as np
from typing import Literal

# =============================================================================
# Joint Index Mappings
# =============================================================================

# Unitree/MuJoCo -> IsaacLab conversion
UNITREE_TO_ISAACLAB = np.array([
    3, 0, 9, 6,   # FL_hip, FR_hip, RL_hip, RR_hip
    4, 1, 10, 7,  # FL_thigh, FR_thigh, RL_thigh, RR_thigh
    5, 2, 11, 8   # FL_calf, FR_calf, RL_calf, RR_calf
], dtype=np.int32)

# IsaacLab -> Unitree/MuJoCo conversion
ISAACLAB_TO_UNITREE = np.array([
    1, 5, 9,   # FR_hip, FR_thigh, FR_calf
    0, 4, 8,   # FL_hip, FL_thigh, FL_calf
    3, 7, 11,  # RR_hip, RR_thigh, RR_calf
    2, 6, 10   # RL_hip, RL_thigh, RL_calf
], dtype=np.int32)

# =============================================================================
# Default Joint Positions
# =============================================================================

# Default standing position (Unitree order)
DEFAULT_JOINT_POS_UNITREE = np.array([
    -0.1, 0.8, -1.5,   # FR_hip, FR_thigh, FR_calf
    0.1, 0.8, -1.5,    # FL_hip, FL_thigh, FL_calf
    -0.1, 1.0, -1.5,   # RR_hip, RR_thigh, RR_calf
    0.1, 1.0, -1.5,    # RL_hip, RL_thigh, RL_calf
], dtype=np.float32)

# Default standing position (IsaacLab order)
DEFAULT_JOINT_POS_ISAACLAB = DEFAULT_JOINT_POS_UNITREE[UNITREE_TO_ISAACLAB]

# =============================================================================
# Joint Names
# =============================================================================

JOINT_NAMES_UNITREE = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

JOINT_NAMES_ISAACLAB = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]

# =============================================================================
# Conversion Functions
# =============================================================================

def unitree_to_isaaclab(joint_data: np.ndarray) -> np.ndarray:
    """
    Convert joint data from Unitree order to IsaacLab order.

    Args:
        joint_data: Joint data in Unitree order (12,)

    Returns:
        Joint data in IsaacLab order (12,)
    """
    if joint_data.shape[0] != 12:
        raise ValueError(f"Expected 12 joints, got {joint_data.shape[0]}")
    return joint_data[UNITREE_TO_ISAACLAB]


def isaaclab_to_unitree(joint_data: np.ndarray) -> np.ndarray:
    """
    Convert joint data from IsaacLab order to Unitree order.

    Args:
        joint_data: Joint data in IsaacLab order (12,)

    Returns:
        Joint data in Unitree order (12,)
    """
    if joint_data.shape[0] != 12:
        raise ValueError(f"Expected 12 joints, got {joint_data.shape[0]}")
    return joint_data[ISAACLAB_TO_UNITREE]


def get_relative_joint_positions(
    positions: np.ndarray,
    order: Literal['unitree', 'isaaclab'] = 'isaaclab'
) -> np.ndarray:
    """
    Get joint positions relative to default standing position.

    Args:
        positions: Absolute joint positions (12,)
        order: Joint order ('unitree' or 'isaaclab')

    Returns:
        Relative joint positions (12,)
    """
    if order == 'unitree':
        default = DEFAULT_JOINT_POS_UNITREE
    elif order == 'isaaclab':
        default = DEFAULT_JOINT_POS_ISAACLAB
    else:
        raise ValueError(f"Unknown order: {order}. Must be 'unitree' or 'isaaclab'")

    return positions - default


def get_absolute_joint_positions(
    relative_positions: np.ndarray,
    order: Literal['unitree', 'isaaclab'] = 'isaaclab'
) -> np.ndarray:
    """
    Get absolute joint positions from relative positions.

    Args:
        relative_positions: Relative joint positions (12,)
        order: Joint order ('unitree' or 'isaaclab')

    Returns:
        Absolute joint positions (12,)
    """
    if order == 'unitree':
        default = DEFAULT_JOINT_POS_UNITREE
    elif order == 'isaaclab':
        default = DEFAULT_JOINT_POS_ISAACLAB
    else:
        raise ValueError(f"Unknown order: {order}. Must be 'unitree' or 'isaaclab'")

    return relative_positions + default
