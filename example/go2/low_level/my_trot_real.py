import time
import sys
import numpy as np
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: number of control steps
SECONDS = 0.2
DT = 0.002
STEPS = int(SECONDS / DT)  # set desired number of control steps

# Command : lin_vel_x, lin_vel_y, ang_vel_yaw, heading
COMMAND_BUF = np.array([0., 0, 0, 0])
ACTION_SCALE = 0.2

# Joint limits
HIP_LIMIT = np.array([np.deg2rad(-48), np.deg2rad(48)])
HIND_THIGH_LIMIT = np.array([np.deg2rad(-260), np.deg2rad(30)])
FORE_THIGH_LIMIT = np.array([np.deg2rad(-200), np.deg2rad(90)])
CALF_LIMIT = np.array([np.deg2rad(-156), np.deg2rad(-48)])

# LIMITS: list of (min, max) for each joint (hip, thigh, calf for each leg)
LIMITS = [
    tuple(HIP_LIMIT), tuple(HIND_THIGH_LIMIT), tuple(CALF_LIMIT),
    tuple(HIP_LIMIT), tuple(FORE_THIGH_LIMIT), tuple(CALF_LIMIT),
    tuple(HIP_LIMIT), tuple(HIND_THIGH_LIMIT), tuple(CALF_LIMIT),
    tuple(HIP_LIMIT), tuple(FORE_THIGH_LIMIT), tuple(CALF_LIMIT),
]

# KD Controller Gain
KP = 40
KD = 2
_DEFAULT_STAND_POS = np.array([ 0.0, 0.67, -1.3,    0.0, 0.67, -1.3,  0.0, 0.67, -1.3,    0.0, 0.67, -1.3  ])

# Scales
DT = 0.002  # 500 Hz
LIN_VEL_SCALE = torch.tensor(2.0)
ANG_VEL_SCALE = torch.tensor(0.25)
DOF_POS_SCALE = torch.tensor(1.0)
DOF_VEL_SCALE = torch.tensor(0.05)
COMMANDS_SCALE = torch.tensor(1.0)
DEFAULT_DOF_POS = torch.tensor([0.00571868, 0.608813, -1.21763, -0.00571868,
    0.608813, -1.21763, 0.00571868, 0.608813, -1.21763,
    -0.00571868, 0.608813, -1.21763])
# ─────────────────────────────────────────────────────────────────────────────

# Global state
latest_lowstate = None

def low_state_handler(msg: LowState_):
    global latest_lowstate
    latest_lowstate = msg

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [ ww + xx - yy - zz,   2*(xy - wz),       2*(xz + wy)    ],
        [ 2*(xy + wz),         ww - xx + yy - zz, 2*(yz - wx)    ],
        [ 2*(xz - wy),         2*(yz + wx),       ww - xx - yy + zz ]
    ], dtype=float)
    return R

class BaseVelocityEstimator:
    def __init__(self, dt: float, gravity: float = 9.81):
        self.dt = dt
        self.gravity = gravity
        self.lin_vel = np.zeros(3, dtype=float)

    def update(self, imu_state):
        q = imu_state.quaternion
        acc_body = np.array(imu_state.accelerometer, dtype=float)
        gyro_body = np.array(imu_state.gyroscope, dtype=float)
        R = quaternion_to_rotation_matrix(q)
        acc_world = R.dot(acc_body)
        acc_lin = acc_world - np.array([0.0, 0.0, self.gravity])
        self.lin_vel += acc_lin * self.dt
        return self.lin_vel, gyro_body.copy()


def compute_projected_gravity(quaternion: np.ndarray, gravity: float = 9.81) -> np.ndarray:
    w, x, y, z = quaternion
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [ ww + xx - yy - zz,   2*(xy - wz),       2*(xz + wy)    ],
        [ 2*(xy + wz),         ww - xx + yy - zz, 2*(yz - wx)    ],
        [ 2*(xz - wy),         2*(yz + wx),       ww - xx - yy + zz ]
    ], dtype=float)
    g_world = np.array([0.0, 0.0, -gravity], dtype=float)
    return R.T.dot(g_world)

# Stand-up sequence (simple three-phase)
def stand_up_sequence(cmd_pub, crc, duration=(500, 500, 1000), dt=0.002):
    latest = None
    def _wait():
        nonlocal latest
        while latest is None:
            time.sleep(0.005)
    def _handler(msg: LowState_):
        nonlocal latest
        latest = msg
    temp_sub = ChannelSubscriber("rt/lowstate", LowState_)
    temp_sub.Init(_handler, 10)
    _wait()
    start_q = [ms.q for ms in latest.motor_state[:12]]
    _p1 = [ 0.0, 1.36, -2.65,   0.0, 1.36, -2.65, -0.2, 1.36, -2.65,   0.2, 1.36, -2.65 ]
    _p2 = [ 0.0, 0.67, -1.3,    0.0, 0.67, -1.3,  0.0, 0.67, -1.3,    0.0, 0.67, -1.3  ]
    d1, d2, d3 = duration
    pct1 = pct2 = pct3 = 0.0
    cmd = unitree_go_msg_dds__LowCmd_()
    for i in range(12): cmd.motor_cmd[i].mode = 0x01
    while True:
        if pct1 < 1:
            pct1 = min(pct1 + 1/d1, 1)
            src, dst = start_q, _p1
        elif pct2 < 1:
            pct2 = min(pct2 + 1/d2, 1)
            src, dst = _p1, _p2
        elif pct3 < 1:
            pct3 = min(pct3 + 1/d3, 1)
            src, dst = _p2, _p2
        else:
            break
        alpha = pct1 if pct1<1 else pct2 if pct2<1 else pct3
        for i in range(12):
            q = (1-alpha)*src[i] + alpha*dst[i]
            cmd.motor_cmd[i].q  = q
            cmd.motor_cmd[i].kp = KP
            cmd.motor_cmd[i].kd = KD
            cmd.motor_cmd[i].dq = 0
            cmd.motor_cmd[i].tau= 0
        cmd.crc = crc.Crc(cmd)
        cmd_pub.Write(cmd)
        time.sleep(dt)



def sit_down_sequence(cmd_pub, crc, duration=(500,500,1000), dt=0.002):
    # wait for fresh state
    latest = None
    def _wait():
        nonlocal latest
        while latest is None:
            time.sleep(0.005)
    def _handler(msg: LowState_):
        nonlocal latest
        latest = msg
    temp_sub = ChannelSubscriber("rt/lowstate", LowState_)
    temp_sub.Init(_handler, 10)
    _wait()

    # current standing pose as src
    current_q = [ms.q for ms in latest.motor_state[:12]]
    # same intermediate targets as stand_up_sequence
    _p1 = [ 0.0, 0.67, -1.3,    0.0, 0.67, -1.3,  0.0, 0.67, -1.3,    0.0, 0.67, -1.3  ]
    _p2 = [ 0.0, 1.36, -2.65,   0.0, 1.36, -2.65, -0.2, 1.36, -2.65,   0.2, 1.36, -2.65 ]

    d1, d2, d3 = duration
    pct1 = pct2 = pct3 = 0.0

    cmd = unitree_go_msg_dds__LowCmd_()
    for i in range(12): cmd.motor_cmd[i].mode = 0x01

    while True:
        if pct1 < 1:
            pct1 = min(pct1 + 1/d1, 1)
            src, dst = current_q, _p1
            alpha = pct1
        elif pct2 < 1:
            pct2 = min(pct2 + 1/d2, 1)
            src, dst = _p1, _p2
            alpha = pct2
        else:
            break

        for i in range(12):
            q = (1-alpha)*src[i] + alpha*dst[i]
            cmd.motor_cmd[i].q  = q
            cmd.motor_cmd[i].kp = KP
            cmd.motor_cmd[i].kd = KD
            cmd.motor_cmd[i].dq = 0
            cmd.motor_cmd[i].tau= 0
        cmd.crc = crc.Crc(cmd)
        cmd_pub.Write(cmd)
        time.sleep(dt)


# Halt motors (Single phase)
def halt_motors(cmd_pub, crc, duration=(500), dt=0.002):
    latest = None
    def _wait():
        nonlocal latest
        while latest is None:
            time.sleep(0.005)
    def _handler(msg: LowState_):
        nonlocal latest
        latest = msg
    temp_sub = ChannelSubscriber("rt/lowstate", LowState_)
    temp_sub.Init(_handler, 10)
    _wait()
    start_q = [ms.q for ms in latest.motor_state[:12]]
    d1 = duration
    pct1 = 0.0
    cmd = unitree_go_msg_dds__LowCmd_()
    for i in range(12): cmd.motor_cmd[i].mode = 0x01

    while True:
        if pct1 < 1:
            pct1 = min(pct1 + 1/d1, 1)
        else:
            break

        alpha = pct1
        for i in range(12):
            q = (1-alpha)*start_q[i] + alpha*start_q[i]
            cmd.motor_cmd[i].q  = q
            cmd.motor_cmd[i].kp = (1 - alpha) * KP + alpha * 0
            cmd.motor_cmd[i].kd = (1 - alpha) * KD + alpha * 0
            cmd.motor_cmd[i].dq = 0
            cmd.motor_cmd[i].tau= 0
        cmd.crc = crc.Crc(cmd)
        cmd_pub.Write(cmd)
        time.sleep(dt)


def clamp_action_vec(act_vec):
    """
    Clamp the action vector to soft limits and enforce hard limits.
    If a hard limit is violated, exit the program.
    """
    # Hard-limit check with 1% margin
    margin = torch.tensor([(um - lm) * 0.01 for lm, um in LIMITS], dtype=act_vec.dtype)
    hard_mins = torch.tensor([lm for lm, _ in LIMITS], dtype=act_vec.dtype) + margin
    hard_maxs = torch.tensor([um for _, um in LIMITS], dtype=act_vec.dtype) - margin
    if torch.any(act_vec < hard_mins) or torch.any(act_vec > hard_maxs):
        print("❌ Hard limit (with 1% margin) exceeded! Shutting down.")
        sys.exit("❌ Hard limit (with 1% margin) exceeded! Shutting down.")
    # Clamp to soft limits
    soft_mins = []
    soft_maxs = []
    for hard_min, hard_max in LIMITS:
        delta = hard_max - hard_min
        soft_mins.append(hard_min + 0.1 * delta)
        soft_maxs.append(hard_max - 0.1 * delta)
    soft_mins = torch.tensor(soft_mins, dtype=act_vec.dtype)
    soft_maxs = torch.tensor(soft_maxs, dtype=act_vec.dtype)
    return torch.max(torch.min(act_vec, soft_maxs), soft_mins)




if __name__ == '__main__':
    # Initialize communication (real robot)
    ChannelFactoryInitialize(0)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(low_state_handler, 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()


    policy = torch.jit.load("./policy/policy.pt")
    policy.eval()

    print("WARNING: Ensure no obstacles around the robot before running.")
    input("Press Enter to continue...\n")

    try:
        # 1) Stand up
        stand_up_sequence(pub, crc)
        print("Stand up sequence completed")

        # 2) Wait for first state
        while latest_lowstate is None:
            time.sleep(0.001)

        # 3) Initialize estimator
        estimator = BaseVelocityEstimator(DT)
        commands_buf = torch.tensor(COMMAND_BUF).unsqueeze(0)
        actions_buf  = torch.zeros((1, 12))

        # 4) Control loop for fixed steps
        for step in range(STEPS):
            st = latest_lowstate
            lin_vel, ang_vel = estimator.update(st.imu_state)
            base_lin = torch.tensor(lin_vel).unsqueeze(0)
            base_ang = torch.tensor(ang_vel).unsqueeze(0)
            proj_g = torch.tensor(compute_projected_gravity(st.imu_state.quaternion)).unsqueeze(0)
            dof_pos = torch.tensor([m.q for m in st.motor_state[:12]]).unsqueeze(0)
            dof_vel = torch.tensor([m.dq for m in st.motor_state[:12]]).unsqueeze(0)
            obs = torch.cat((
                base_lin * lin_vel_scale,
                base_ang * ang_vel_scale,
                proj_g,
                commands_buf[:, :3] * commands_scale,
                (dof_pos - default_dof_pos) * dof_pos_scale,
                dof_vel * dof_vel_scale,
                actions_buf
            ), dim=-1).float()
            obs = torch.clamp(obs, -clip_obs, clip_obs)
            with torch.no_grad():
                act = default_dof_pos + policy(obs).cpu() * ACTION_SCALE


            act_vec = act[0]
            clamped = clamp_action_vec(act_vec)
            actions_buf[0] = clamped

            
            cmd = unitree_go_msg_dds__LowCmd_()
            cmd.head[0], cmd.head[1] = 0xFE, 0xEF
            cmd.level_flag = 0xFF
            cmd.gpio = 0
            for i in range(12):
                cmd.motor_cmd[i].mode = 0x01
                cmd.motor_cmd[i].q  = float(actions_buf[0,i])
                cmd.motor_cmd[i].kp = KP
                cmd.motor_cmd[i].kd = KD
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].tau= 0.0
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)
            time.sleep(dt)
        print(f"Completed {STEPS} steps, stopping control loop.")
    
    except Exception as e:
        print(f"Exception: {e}")

    finally:
        # 6) Sit down
        sit_down_sequence(pub, crc)
        print("Sit down sequence completed")
        
        # 7) Halt motors
        halt_motors(pub, crc)
        print("Control script finished.")

        time.sleep(1)
        print("Done!")
        sys.exit(-1) 