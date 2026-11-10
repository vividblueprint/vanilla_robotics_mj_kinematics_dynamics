import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import time

from forward_kinematics import forward_kinematics
from ik import inverse_kinematics
from scipy.optimize import fsolve
import utility as ram


xml_path = '../universal_robots_ur5e/scene.xml'
simend = 5
print_camera_config = 0


def init_controller(model, data):
    # Initialize controller here
    pass


def controller(model, data):
    # Put controller here
    pass


# Get the full path
dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)


# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)


# Initialize controller
init_controller(model, data)


# Set controller
mj.set_mjcb_control(controller)


# Initial joint configuration
key_qpos = model.key("home").qpos.copy()

q = key_qpos.copy()
q = np.array([-1.23, -1.5, 0.5, -1.5708, -1.5708, 0])


# Reference
x_ref = 0.5
y_ref = 0.2
z_ref = 0.5

phi_ref = 3.14
theta_ref = 0
psi_ref = 0

X_ref = np.array([
    x_ref,
    y_ref,
    z_ref,
    phi_ref,
    theta_ref,
    psi_ref
])


# Initialize robot
data.qpos[:] = q
mj.mj_forward(model, data)


# Launch MuJoCo viewer
with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:

    # Camera configuration
    with viewer.lock():
        viewer.cam.azimuth = -130
        viewer.cam.elevation = -5
        viewer.cam.distance = 2
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.5])

        # Show site coordinate frames
        viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE

    while viewer.is_running() and data.time < simend:

        start = time.perf_counter()
        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:

            with viewer.lock():

                data.time += 0.02

                # //////////////////////| Inverse Kinematics |//////////////////////

                q = fsolve(
                    inverse_kinematics,
                    q,
                    args=(X_ref,)
                )

                # Apply calculated joint configuration
                data.qpos[:] = q

                # Update forward kinematics
                mj.mj_forward(model, data)

                # mj.mj_step(model, data)

                # //////////////////////| Forward Kinematics |//////////////////////

                mj_end_eff_pos = data.site('attachment_site').xpos.copy()
                mj_end_eff_mat = data.site('attachment_site').xmat.copy()

                mj_end_eff_quat = np.zeros(4)

                mj.mju_mat2Quat(
                    mj_end_eff_quat,
                    mj_end_eff_mat
                )

                print("mj_end_eff_pos =", mj_end_eff_pos)
                print("mj_end_eff_quat =", mj_end_eff_quat)

                print("ref end_eff_pos =", X_ref[:3])

                ref_euler = ram.euler2quat(
                    np.array([phi_ref, theta_ref, psi_ref])
                )

                print("ref end_eff_quat =", ref_euler)

                # sol, robot = forward_kinematics(q)

                # py_end_eff_pos = sol.end_eff_pos
                # py_end_eff_rot = sol.end_eff_rot
                # py_end_eff_quat = ram.rotation2quat(py_end_eff_rot)

                # print("py_end_eff_pos =", py_end_eff_pos)
                # print("py_end_eff_quat =", py_end_eff_quat)

                print(' ***** ')

        if data.time >= simend:
            break

        # Print camera configuration
        if print_camera_config == 1:

            with viewer.lock():

                print(
                    'cam.azimuth =', viewer.cam.azimuth,
                    '; cam.elevation =', viewer.cam.elevation,
                    '; cam.distance =', viewer.cam.distance
                )

                print(
                    'cam.lookat =',
                    viewer.cam.lookat.copy()
                )

        # Update viewer
        viewer.sync()

        # Limit viewer updates to approximately 60 FPS
        elapsed = time.perf_counter() - start
        remaining = 1.0 / 60.0 - elapsed

        if remaining > 0:
            time.sleep(remaining)


# Clear controller callback
mj.set_mjcb_control(None)