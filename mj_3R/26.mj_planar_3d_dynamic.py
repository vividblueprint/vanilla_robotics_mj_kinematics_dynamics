import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import time
import utility as rm

from forward_kinematics import forward_kinematics
from differential_fk import differential_kinematics
from jac_com import jac_com
from jac_end_effector import jac_end_effector
from RNEA import RNEA


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

model.opt.gravity = np.array([0, -9.81, 0])

q = np.array([0.5, -1.2, 1.57])
u = np.array([1.23, 6.54, 1.89])

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
                data.qpos = q.copy()

                mj.mj_forward(model, data)

                C_py,G_py,CG_py,M_py = RNEA(q,u)

                #toc = time.time()  # End timer
                #print(f"Elapsed time: {toc - tic:.4f} seconds")

                #M*qddot + qfrc_bias = tau
                #M*qddot + C + G = tau
                #so C+G = qfrc_bias
                print("Gravity and Coriolis")
                qfrc_bias = np.zeros((model.nv,1))
                for i in range(0,model.nv):
                    qfrc_bias[i]=data.qfrc_bias[i]
                rm.print_matrix(qfrc_bias,4,"CG_mj")
                rm.print_matrix(CG_py,4,"CG_py")
                rm.print_matrix(CG_py-qfrc_bias,6,"CG_py-CG_mj ")


                print("Mass matrix")
                M_mj = np.zeros((model.nv,model.nv ))
                mj.mj_fullM(model, data, M_mj)
                rm.print_matrix(M_mj,6,"M_mj")
                rm.print_matrix(M_py,6,"M_py")

                rm.print_matrix(M_py-M_mj,6,"M_py-M_mj")

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