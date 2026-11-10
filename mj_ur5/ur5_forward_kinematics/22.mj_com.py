import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import time

from forward_kinematics import forward_kinematics
from differential_fk import differential_kinematics
from jac_com import jac_end
import utility as rm


xml_path = '../universal_robots_ur5e/scene.xml'
simend = 5
print_camera_config = 0


def print_matrix(Mat, decimals=6, name="Mat"):
    if not isinstance(Mat, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray.")

    if Mat.ndim != 2:
        raise ValueError("Input must be a 2D numpy.ndarray (matrix).")

    format_string = f"{{:.{decimals}f}}"
    print(f"{name} = [ ...")

    for row in Mat:
        formatted_row = " ".join(format_string.format(elem) for elem in row)
        print(f"  {formatted_row};")

    print("];\n")


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


# Robot body names
body_names = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]

joint_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


site_id = model.site("attachment_site").id
body_ids = [model.body(name).id for name in body_names]
joint_ids = np.array([model.joint(name).id for name in joint_names])
actuator_ids = np.array([model.actuator(name).id for name in joint_names])

print("site_id = ", site_id)
print("body_ids = ", body_ids)
print("joint_ids = ", joint_ids)


# Keyframe
key_id = model.key("home").id
key_qpos = model.key("home").qpos.copy()
key_qvel = model.key("home").qvel.copy()
key_ctrl = model.key("home").ctrl.copy()

q = key_qpos.copy()
u = key_qvel.copy()

# q = np.array([-1.23, -1.5, 0.5, -1.5708, -1.5708, 0])
# u = np.array([-0.2, 0.45, -0.67, -1.3, 2.8, -0.768])


mj_jac_E = np.zeros((6, model.nv))
py_jac_E = np.zeros((6, model.nv))


# Initialize robot at home configuration
mj.mj_resetDataKeyframe(model, data, key_id)

data.qpos[:] = q
data.qvel[:] = u

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

    while viewer.is_running() and data.time < simend:

        start = time.perf_counter()
        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:

            with viewer.lock():

                data.time += 0.02
                data.qpos[:] = q
                data.qvel[:] = u

                J_g1, J_g2, J_g3, J_g4, J_g5, J_g6 = jac_end(q)

                mj.mj_forward(model, data)

                # mj.mj_step(model, data)

                # //////////////////////| COM Jacobian |//////////////////////

                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))

                for i in range(0, len(body_ids)):

                    mj.mj_jacBodyCom(
                        model,
                        data,
                        jacp,
                        jacr,
                        body_ids[i]
                    )

                    print("body_id = ", body_ids[i])
                    print("body_name = ", body_names[i])

                    jac = np.vstack([jacp, jacr])

                    if i == 0:
                        print_matrix(jac - J_g1, 4, "jac-J_g1")

                    if i == 1:
                        print_matrix(jac - J_g2, 4, "jac-J_g2")

                    if i == 2:
                        print_matrix(jac - J_g3, 4, "jac-J_g3")

                    if i == 3:
                        print_matrix(jac - J_g4, 4, "jac-J_g4")

                    if i == 4:
                        print_matrix(jac - J_g5, 4, "jac-J_g5")

                    if i == 5:
                        print_matrix(jac - J_g6, 4, "jac-J_g6")

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