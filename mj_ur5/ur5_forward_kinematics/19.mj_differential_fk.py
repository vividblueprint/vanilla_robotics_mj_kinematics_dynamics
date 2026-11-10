
import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import time
import utility as rm

from forward_kinematics import forward_kinematics
from differential_fk import differential_kinematics
# from jac_com import jac_com
# from jac_end_effector import jac_end_effector


xml_path = '../universal_robots_ur5e/scene.xml'
simend = 10
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
key_qpos = model.key("home").qpos
key_qvel = model.key("home").qvel
q = key_qpos.copy()
u = key_qvel.copy()

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

                robot, dfk = differential_kinematics(q,u)

                print('py bodyframe 3 lin/ang vel')
                print(dfk.v_globals[3])
                print(dfk.w_globals[3])
                print('mj bodyframe 3 lin/ang vel')
                print(data.sensor('forearm_link_frame_linvel').data)
                print(data.sensor('forearm_link_frame_angvel').data)

                print('py body 3 com linvel')
                print(dfk.v_com_globals[3])
                print('mj body 3 com linvel')
                print(data.sensor('forearm_link_com_linvel').data)

                print('py tip lin/ang vel')
                print(dfk.v_tip_global)
                print(dfk.w_tip_global)
                print('mj tip lin/ang vel')
                print(data.sensor('tip_linvel').data)
                print(data.sensor('tip_angvel').data)

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

