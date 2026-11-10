import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import time

from sympy import true

from forward_kinematics import forward_kinematics
from ik import inverse_kinematics
from scipy.optimize import fsolve
import utility as rm
import matplotlib.pyplot as plt


xml_path = '../universal_robots_ur5e/scene.xml'
simend = 50
print_camera_config = 0

realtime_plot = true

t_init = 1
r = 0.1
f = 1


def circle(t, x_init, y_init, r, f):
    x = x_init + r * np.cos(2 * np.pi * f * t)
    y = y_init + r * np.sin(2 * np.pi * f * t)

    return x, y


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


# Initial joint configuration
# q = model.key("home").qpos.copy()
q = np.array([-1.23, -1.5, 0.5, -1.5708, -1.5708, 0])

X_ref = np.array([0.5, 0.2, 0.5, 3.14, 0, 0])


# Initialize robot
data.qpos[:] = q
mj.mj_forward(model, data)


# ////////////////////| Plotting |/////////////////////////////////

plot_counter = 0

time_all = []

x_ref_all = []
y_ref_all = []
z_ref_all = []

x_all = []
y_all = []
z_all = []

if realtime_plot:
    plt.ion()

    fig_traj, ax_traj = plt.subplots(figsize=(8, 6))

    ref_line, = ax_traj.plot(
        [], [], 'r--', linewidth=2, label='Reference'
    )

    actual_line, = ax_traj.plot(
        [], [], 'b-', linewidth=2, label='Actual'
    )

    ref_point, = ax_traj.plot([], [], 'ro', markersize=8)
    actual_point, = ax_traj.plot([], [], 'bo', markersize=8)

    ax_traj.set_title('End Effector XY Trajectory')
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.grid(True)
    ax_traj.axis('equal')
    ax_traj.set_xlim(0.2, 0.8)
    ax_traj.set_ylim(-0.2, 0.6)
    ax_traj.legend()

# ///////////////////////////////////////////////////////////////


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

                # mj.mj_step(model, data)

                if data.time < t_init:
                    x_init = X_ref[0] - r
                    y_init = X_ref[1]

                else:
                    X_ref[0], X_ref[1] = circle(
                        data.time - t_init,
                        x_init,
                        y_init,
                        r,
                        f
                    )

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

                # //////////////////////| Forward Kinematics |//////////////////////

                mj_end_eff_pos = data.site('attachment_site').xpos.copy()
                mj_end_eff_mat = data.site('attachment_site').xmat.copy()

                mj_end_eff_quat = np.zeros(4)
                mj.mju_mat2Quat(mj_end_eff_quat, mj_end_eff_mat)

                print("mj_end_eff_pos =", mj_end_eff_pos)
                print("mj_end_eff_quat =", mj_end_eff_quat)

                print("ref_end_eff_pos =", X_ref[:3])

                # Orientation is stored in X_ref[3:6]
                ref_quat = rm.euler2quat(X_ref[3:6])

                print("ref_end_eff_quat =", ref_quat)

                # robot, sol = forward_kinematics(q)

                # py_end_eff_pos = sol.end_eff_pos
                # py_end_eff_rot = sol.end_eff_rot
                # py_end_eff_quat = rm.rotation2quat(py_end_eff_rot)

                # print("py_end_eff_pos =", py_end_eff_pos)
                # print("py_end_eff_quat =", py_end_eff_quat)

                print(' ***** ')

                # //////////////////////| Trajectory Data |//////////////////////

                if data.time > t_init:

                    time_all.append(data.time - t_init)

                    x_ref_all.append(X_ref[0])
                    y_ref_all.append(X_ref[1])
                    z_ref_all.append(X_ref[2])

                    x_all.append(mj_end_eff_pos[0])
                    y_all.append(mj_end_eff_pos[1])
                    z_all.append(mj_end_eff_pos[2])

                    plot_counter += 1

            # //////////////////////| Realtime Plotting |//////////////////////
            # Keep Matplotlib updates outside the viewer lock.

            if (
                data.time > t_init
                and plot_counter % 1 == 0
                and realtime_plot
            ):

                # Trajectory history
                ref_line.set_data(x_ref_all, y_ref_all)
                actual_line.set_data(x_all, y_all)

                # Current positions
                ref_point.set_data(
                    [X_ref[0]],
                    [X_ref[1]]
                )

                actual_point.set_data(
                    [mj_end_eff_pos[0]],
                    [mj_end_eff_pos[1]]
                )

                ax_traj.relim()
                ax_traj.autoscale_view()

                fig_traj.canvas.draw_idle()
                fig_traj.canvas.flush_events()

                # plt.pause(0.001)

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


# //////////////////////| Final Plot |//////////////////////

plt.ioff()

plt.figure()

plt.plot(
    x_ref_all,
    y_ref_all,
    label='ref',
    color='black',
    linestyle='--',
    linewidth=2
)

plt.plot(
    x_all,
    y_all,
    label='act',
    color='red'
)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Trajectory')
plt.grid()
plt.axis('equal')
plt.show()