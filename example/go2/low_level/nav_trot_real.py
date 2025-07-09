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
SECONDS = 0.5
DT = 0.002
STEPS = int(SECONDS / DT)  # set desired number of control steps

# Command : lin_vel_x, lin_vel_y, ang_vel_yaw, heading
COMMAND_BUF = np.array([0.0, 0, 0, 0])
ACTION_SCALE = 0.1


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
def stand_up_sequence(cmd_pub, crc, duration=(500, 500, 1000), gains=(60, 5), dt=0.002):
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
            kp, kd = gains
        elif pct2 < 1:
            pct2 = min(pct2 + 1/d2, 1)
            src, dst = _p1, _p2
            kp, kd = gains
        elif pct3 < 1:
            pct3 = min(pct3 + 1/d3, 1)
            src, dst = _p2, _p2
            kp, kd = gains
        else:
            break
        alpha = pct1 if pct1<1 else pct2 if pct2<1 else pct3
        for i in range(12):
            q = (1-alpha)*src[i] + alpha*dst[i]
            cmd.motor_cmd[i].q  = q
            cmd.motor_cmd[i].kp = kp
            cmd.motor_cmd[i].kd = kd
            cmd.motor_cmd[i].dq = 0
            cmd.motor_cmd[i].tau= 0
        cmd.crc = crc.Crc(cmd)
        cmd_pub.Write(cmd)
        time.sleep(dt)

def sit_down_sequence(cmd_pub, crc, duration=(1000,1000,1000), gains=(60,5), dt=0.002):
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
            src, dst = _p1, _p2
            kp, kd = gains
            alpha = pct1
        elif pct2 < 1:
            pct2 = min(pct2 + 1/d2, 1)
            src, dst = _p2, _p2
            kp, kd = gains
            alpha = pct2
        else:
            break

        for i in range(12):
            q = (1-alpha)*src[i] + alpha*dst[i]
            cmd.motor_cmd[i].q  = q
            cmd.motor_cmd[i].kp = kp
            cmd.motor_cmd[i].kd = kd
            cmd.motor_cmd[i].dq = 0
            cmd.motor_cmd[i].tau= 0
        cmd.crc = crc.Crc(cmd)
        cmd_pub.Write(cmd)
        time.sleep(dt)


# Global state
latest_lowstate = None

def LowStateHandler(msg: LowState_):
    global latest_lowstate
    latest_lowstate = msg

if __name__ == '__main__':
    # Initialize communication (real robot)
    ChannelFactoryInitialize(0)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(LowStateHandler, 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = CRC()


    policy = torch.jit.load("./policy/navaneet.pt")
    policy.eval()
    temporary_height_data = torch.ones((1, 187)) * 0.3

    print("WARNING: Ensure no obstacles around the robot before running.")
    input("Press Enter to continue...\n")

    try:
        # 1) Stand up
        stand_up_sequence(pub, crc)
        print("Stand up sequence completed")

        # 2) Wait for first state
        while latest_lowstate is None:
            time.sleep(0.001)


        # 4) Initialize buffers & scales
        dt = 0.002  # 500 Hz
        estimator = BaseVelocityEstimator(dt)
        lin_vel_scale = torch.tensor(2.0)
        ang_vel_scale = torch.tensor(0.25)
        dof_pos_scale = torch.tensor(1.0)
        dof_vel_scale = torch.tensor(0.05)
        commands_scale = torch.tensor(1.0)
        default_dof_pos = torch.tensor([0.00571868, 0.608813, -1.21763, -0.00571868,
            0.608813, -1.21763, 0.00571868, 0.608813, -1.21763,
            -0.00571868, 0.608813, -1.21763])
        total_cmd_dims = 4
        clip_obs = 100.0
        clip_act = 100.0
        commands_buf = torch.tensor(COMMAND_BUF).unsqueeze(0)
        actions_buf  = torch.zeros((1, 12))

        # 5) Control loop for fixed steps
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
                actions_buf,
                temporary_height_data
            ), dim=-1).float()
            obs = torch.clamp(obs, -clip_obs, clip_obs)
            with torch.no_grad():
                act = default_dof_pos + policy(obs).cpu() * ACTION_SCALE
            actions_buf = act.clone().clamp(-clip_act, clip_act)
            cmd = unitree_go_msg_dds__LowCmd_()
            cmd.head[0], cmd.head[1] = 0xFE, 0xEF
            cmd.level_flag = 0xFF
            cmd.gpio = 0
            for i in range(12):
                cmd.motor_cmd[i].mode = 0x01
                cmd.motor_cmd[i].q  = float(act[0,i])
                cmd.motor_cmd[i].kp = 50.0
                cmd.motor_cmd[i].kd = 5.0
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
        
        # Halt motors after steps complete
        zero_cmd = unitree_go_msg_dds__LowCmd_()
        for i in range(12):
            zero_cmd.motor_cmd[i].mode = 0x01
            zero_cmd.motor_cmd[i].q  = 0.0
            zero_cmd.motor_cmd[i].kp = 0.0
            zero_cmd.motor_cmd[i].kd = 0.0
            zero_cmd.motor_cmd[i].dq = 0.0
            zero_cmd.motor_cmd[i].tau= 0.0
        zero_cmd.crc = crc.Crc(zero_cmd)
        pub.Write(zero_cmd)
        print("Control script finished.")