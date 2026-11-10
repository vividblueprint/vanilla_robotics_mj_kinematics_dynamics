import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import time

from jac_com import jac_com
from jac_end_effector import jac_end_effector


xml_path = 'planar_3R.xml'
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


# Initialize robot
data.qpos[:] = key_qpos
data.qvel[:] = key_qvel
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
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -90.0
        viewer.cam.distance = 6.0
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])

    while viewer.is_running() and data.time < simend:

        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:

            # mj.mj_step(model, data)

            with viewer.lock():

                data.time += 0.2
                data.qpos[:] = key_qpos
                data.qvel[:] = key_qvel

                # data.qpos[:] = np.array([1.5707, -1.5707, 1.5707])

                mj.mj_forward(model, data)

                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))

                l = np.array([1, 1, 0.25])
                q = data.qpos.copy()
                u = data.qvel.copy()

                # q = np.array([np.pi/2, -np.pi/2, np.pi/2])
                # fk = forward_kinematics(q, l)

                _, py_J_G2, _ = jac_com(q, l)

                print('py_J_G2')
                print(py_J_G2)

                # ///////////////| Jacobian |/////////////

                body_id = model.body('body2').id

                mj.mj_jacBodyCom(
                    model, data, jacp, jacr, body_id
                )

                mj_J_G2 = np.vstack((jacp, jacr))

                print("mj_J_G2")
                print(mj_J_G2)

                if (1):

                    py_J_E = jac_end_effector(q, l)

                    print("py_J_E")
                    print(py_J_E)

                    body_id = model.body("body3").id
                    mj_end_eff_pos = data.site("tip").xpos.copy()

                    mj.mj_jac(
                        model,
                        data,
                        jacp,
                        jacr,
                        mj_end_eff_pos,
                        body_id
                    )

                    mj_J_E = np.vstack((jacp, jacr))

                    print("mj_J_E")
                    print(mj_J_E)

                # ///////////////| Differential kinematics |/////////////

                # fk = differential_kinematics(q, u, l)

                # print('py frame 2 lin/angvel')
                # print(fk.w2)
                # print(fk.v2)

                # print('mj frame 2 lin/angvel')
                # print(data.sensor('bodyframe2_angvel').data)
                # print(data.sensor('bodyframe2_linvel').data)

                # print('py com 2 linvel')
                # print(fk.vC2)

                # print('mj com 2 linvel')
                # print(data.sensor('body2_linvel').data)

                # print('py end-eff lin/angvel')
                # print(fk.we)
                # print(fk.ve)

                # print('mj end-eff lin/angvel')
                # print(data.sensor('tip_angvel').data)
                # print(data.sensor('tip_linvel').data)

                # /////////////////////////////////////////////////////

                # mj_end_eff_pos = data.site("tip").xpos
                # print("mj_end_eff_pos: ", mj_end_eff_pos)
                # print("py_end_eff_pos: ", fk.e)

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