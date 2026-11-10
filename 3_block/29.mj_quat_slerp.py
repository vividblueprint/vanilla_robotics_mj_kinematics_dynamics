from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation, Slerp


# ---------------- CONFIG ----------------
MODEL_PATH = Path(__file__).parent / "block.xml"

T_START = 2.0
T_END = 5.0
SIM_END = 10.0
DT = 0.005

EULER_START = [0, 0, 0]
EULER_END = [np.pi / 2, np.pi / 2 + 0.5, np.pi / 2]


# ---------------- SLERP ----------------
def trajectory(t, slerp, axis, angle):

    T = T_END - T_START

    u = np.clip((t - T_START) / T, 0.0, 1.0)

    # Quintic time scaling
    s = np.clip(10*u**3 - 15*u**4 + 6*u**5, 0.0, 1.0)

    # Derivatives
    ds = (30*u**2 - 60*u**3 + 30*u**4) / T
    dds = (60*u - 180*u**2 + 120*u**3) / T**2

    # Quaternion
    quat = slerp([s]).as_quat(scalar_first=True)[0]

    # Angular velocity and acceleration
    omega = axis * angle * ds
    alpha = axis * angle * dds

    return quat, omega, alpha


# ---------------- INITIALIZE ----------------
model = mj.MjModel.from_xml_path(str(MODEL_PATH))
data = mj.MjData(model)

# Original code uses Bryant angles: Rx @ Ry @ Rz
r0 = Rotation.from_euler("XYZ", EULER_START)
rf = Rotation.from_euler("XYZ", EULER_END)

# SLERP uses shortest quaternion path
slerp = Slerp([0, 1], Rotation.concatenate([r0, rf]))

# Relative rotation in world coordinates
relative = rf * r0.inv()
rotvec = relative.as_rotvec()

angle = np.linalg.norm(rotvec)
axis = rotvec / angle if angle > 1e-10 else np.array([1.0, 0.0, 0.0])

# Recorded data: time, quaternion, omega, alpha
records = []


# ---------------- SIMULATION ----------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    viewer.cam.azimuth = 167.51
    viewer.cam.elevation = -17.85
    viewer.cam.distance = 1.34
    viewer.cam.lookat[:] = [0.0, 0.0, 0.0]

    viewer.opt.frame = mj.mjtFrame.mjFRAME_BODY

    while viewer.is_running() and data.time < SIM_END:

        start = time.time()

        t = data.time

        quat, omega, alpha = trajectory(t, slerp, axis, angle)

        # Set free-joint orientation
        data.qpos[3:7] = quat

        # Update kinematics without advancing physics
        mj.mj_forward(model, data)

        if T_START <= t <= T_END:
            records.append([t - T_START, *quat, *omega, *alpha])

        viewer.sync()

        # Advance prescribed trajectory time
        data.time += DT

        # Real-time playback
        remaining = DT - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)


# ---------------- PLOTS ----------------
records = np.asarray(records)

t = records[:, 0]
quat = records[:, 1:5]
omega = records[:, 5:8]
alpha = records[:, 8:11]


def plot_data(values, labels, title, ylabel):
    plt.figure()

    for i, label in enumerate(labels):
        plt.plot(t, values[:, i], label=label)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)


plot_data(quat, ["w", "x", "y", "z"], "Quaternion", "Quaternion")
plot_data(omega, ["wx", "wy", "wz"], "Angular Velocity", "rad/s")
plot_data(alpha, ["ax", "ay", "az"], "Angular Acceleration", "rad/s²")

# plt.show()
plt.show(block=False)
plt.pause(5)
plt.close()