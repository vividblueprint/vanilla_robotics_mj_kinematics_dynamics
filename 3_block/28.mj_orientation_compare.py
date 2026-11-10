import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np
from scipy.linalg import svd

import utility as ram


XML_PATH = Path(__file__).resolve().parent / "three_blocks.xml"

SIM_END = 10.0
DT = 0.005
T_START, T_END = 2.0, 5.0

def linear_interp(t, t0, tf, p0, pf):
    s = np.clip((t - t0) / (tf - t0), 0.0, 1.0)
    return (1 - s) * p0 + s * pf

model = mj.MjModel.from_xml_path(str(XML_PATH))
data = mj.MjData(model)

euler_i = np.array([0.0, 0.0, 0.0])
euler_f = np.array([np.pi / 2, 0.0, 0.0])

quat_i = ram.euler2quat(euler_i)
quat_f = ram.euler2quat(euler_f)

R_i = ram.euler2rotation(euler_i)
R_f = ram.euler2rotation(euler_f)

pos1 = np.array([0.0, -0.4, 0.0])
pos2 = np.array([0.0, 0.0, 0.0])
pos3 = np.array([0.0, 0.4, 0.0])

q = np.hstack((pos1, quat_i, pos2, quat_i, pos3, quat_i))

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 167.51
    viewer.cam.elevation = -17.85
    viewer.cam.distance = 1.34
    viewer.cam.lookat[:] = [0.0, 0.0, 0.0]

    viewer.opt.frame = mj.mjtFrame.mjFRAME_BODY

    while viewer.is_running() and data.time < SIM_END:
        start = time.perf_counter()

        data.time += DT
        t = data.time

        # 1. Euler interpolation
        euler = linear_interp(t, T_START, T_END, euler_i, euler_f)
        q[3:7] = ram.euler2quat(euler)

        # 2. Quaternion interpolation (NLERP)
        quat = linear_interp(t, T_START, T_END, quat_i, quat_f)
        quat /= np.linalg.norm(quat)
        q[10:14] = quat

        # 3. Rotation matrix interpolation (SVD)
        R = linear_interp(t, T_START, T_END, R_i, R_f)

        U, _, Vt = svd(R)
        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        q[17:21] = ram.rotation2quat(R)

        data.qpos[:] = q
        mj.mj_forward(model, data)

        viewer.sync()

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, DT - elapsed))
