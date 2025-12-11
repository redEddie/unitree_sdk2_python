"""
Unitree Go2 Sim2Real Utility Modules
=====================================

Shared utility modules for Unitree Go2 robot control and observation.

Modules:
- unitree_joint_config: Joint index mappings and default positions
- unitree_imu_utils: IMU data processing and transformations
- unitree_mode_manager: Robot mode management (MCF/Sport mode switching)
- unitree_communication: Communication setup and channel management
- unitree_observation_builder: Observation processing for policy deployment
"""

__version__ = "1.0.0"
__all__ = [
    "unitree_joint_config",
    "unitree_imu_utils",
    "unitree_mode_manager",
    "unitree_communication",
    "unitree_observation_builder",
]
