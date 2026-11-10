
import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import time
import utility as rm

from forward_kinematics import forward_kinematics
# from differential_fk import differential_kinematics
# from jac_com import jac_com
# from jac_end_effector import jac_end_effector


xml_path = '../universal_robots_ur5e/scene.xml'
simend = 100
print_camera_config = 0


def init_controller(model, data):
    # Initialize controller
    pass


def controller(model, data):
    # Controller
    pass


# Get full path
dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)


# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)


# Initialize controller
init_controller(model, data)


# Set controller
mj.set_mjcb_control(controller)


# Keyframe
key_id = model.key("home").id
key_qpos = model.key_qpos[key_id].copy()
key_qvel = model.key_qvel[key_id].copy()


# # Initialize robot
# data.qpos[:] = key_qpos
# data.qvel[:] = key_qvel
# mj.mj_forward(model, data)


# Launch MuJoCo viewer
with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:

    # Camera configuration
    with viewer.lock():
        viewer.cam.azimuth = -130.0
        viewer.cam.elevation = -5.0
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.5])

    while viewer.is_running() and data.time < simend:

        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:

            # mj.mj_step(model, data)

            with viewer.lock():
                data.time += 0.02
                data.qpos = key_qpos.copy()
                mj.mj_forward(model,data)
                #mj.mj_step(model, data)

                mj_end_eff_pos = data.site('attachment_site').xpos
                mj_end_eff_mat = data.site('attachment_site').xmat

                mj_end_eff_quat = np.zeros(4)
                mj.mju_mat2Quat(mj_end_eff_quat, mj_end_eff_mat)
                #print(mj_end_eff_quat)

                print(f"mj_end_eff_pos = {mj_end_eff_pos} mj_end_eff_quat = {mj_end_eff_quat}")

                robot, sol = forward_kinematics(key_qpos)
                py_end_eff_pos = sol.end_eff_pos
                py_end_eff_rot = sol.end_eff_rot
                py_end_eff_quat = rm.rotation2quat(py_end_eff_rot)

                print(f"py_end_eff_pos = {py_end_eff_pos} py_end_eff_quat = {py_end_eff_quat}")

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
                print('cam.lookat =', viewer.cam.lookat.copy())

        # Update viewer
        viewer.sync()

        time.sleep(1.0 / 60.0)


# Clear controller callback
mj.set_mjcb_control(None)

