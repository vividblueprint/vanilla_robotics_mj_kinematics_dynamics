from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from scipy.linalg import solve_discrete_are

MODEL_PATH = Path(__file__).resolve().parent.parent / "../robot_rl/assets/balance_bot/scene.xml"

MOTOR_LIMIT = 3.0
INITIAL_PITCH = 0.08

PITCH_AXIS = np.array([0.0, 1.0, 0.0])

# LQR weights: [position, pitch, velocity, pitch_rate]
Q = np.diag([
    5.0,
    300.0,
    3.0,
    25.0,
])

# Penalty on the common wheel command
R_LQR = np.array([[0.5]])

# Finite-difference perturbation
EPS = 1e-4


# ============================================================
# INITIALIZE MUJOCO
# ============================================================

model = mj.MjModel.from_xml_path(str(MODEL_PATH))
data = mj.MjData(model)

if model.nu != 2:
    raise ValueError("Expected exactly two wheel actuators.")

freejoint_ids = [
    i
    for i in range(model.njnt)
    if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE
]

if len(freejoint_ids) != 1:
    raise ValueError("Expected exactly one free joint.")

freejoint_id = freejoint_ids[0]

chassis_id = int(model.jnt_bodyid[freejoint_id])
qpos_adr = int(model.jnt_qposadr[freejoint_id])
qvel_adr = int(model.jnt_dofadr[freejoint_id])

print("Chassis:", model.body(chassis_id).name)

def get_rotation_matrix(d):
    quat = d.qpos[qpos_adr + 3:qpos_adr + 7]

    rot = np.zeros(9)
    mj.mju_quat2Mat(rot, quat)

    return rot.reshape(3, 3)

def get_state(d):
    """
    State:
        [position_x, pitch, velocity_x, pitch_rate]

    Assumes planar motion near upright.
    """

    R = get_rotation_matrix(d)

    pitch = np.arctan2(-R[2, 0], R[0, 0])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mj.mj_jacBody(
        model,
        d,
        jacp,
        jacr,
        chassis_id,
    )

    linear_velocity = jacp @ d.qvel
    angular_velocity = jacr @ d.qvel

    position = d.qpos[qpos_adr]
    velocity = linear_velocity[0]
    pitch_rate = angular_velocity[1]

    return np.array([
        position,
        pitch,
        velocity,
        pitch_rate,
    ], dtype=float)

def reset_robot(pitch=INITIAL_PITCH):
    mj.mj_resetData(model, data)

    quat = np.zeros(4)

    mj.mju_axisAngle2Quat(
        quat,
        PITCH_AXIS,
        pitch,
    )

    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

    mj.mj_forward(model, data)

def set_state_perturbation(d, index, epsilon):
    """
    Perturb one reduced-state coordinate.

    0: base X position
    1: pitch
    2: base X velocity
    3: pitch angular velocity
    """

    if index == 0:
        d.qpos[qpos_adr] += epsilon

    elif index == 1:
        quat = np.zeros(4)

        mj.mju_axisAngle2Quat(
            quat,
            PITCH_AXIS,
            epsilon,
        )

        d.qpos[qpos_adr + 3:qpos_adr + 7] = quat

    elif index == 2:
        d.qvel[qvel_adr] += epsilon

    elif index == 3:
        d.qvel[qvel_adr + 4] += epsilon


def simulate_one_step(state_index=None, delta=0.0, torque=0.0):
    """
    Evaluate one MuJoCo step near the nominal upright state.
    """

    d = mj.MjData(model)

    mj.mj_resetData(model, d)

    if state_index is not None:
        set_state_perturbation(d, state_index, delta)

    mj.mj_forward(model, d)

    # Equal wheel commands
    d.ctrl[:] = torque

    mj.mj_step(model, d)

    return get_state(d)


def linearize():
    """
    Numerically estimate:

        x_next ≈ A*x + B*u + c

    using central finite differences.
    """

    n = 4
    m = 1

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        x_plus = simulate_one_step(
            state_index=i,
            delta=EPS,
        )

        x_minus = simulate_one_step(
            state_index=i,
            delta=-EPS,
        )

        A[:, i] = (x_plus - x_minus) / (2 * EPS)

    u_plus = simulate_one_step(torque=EPS)
    u_minus = simulate_one_step(torque=-EPS)

    B[:, 0] = (u_plus - u_minus) / (2 * EPS)

    return A, B


# ============================================================
# CALCULATE LQR GAIN
# ============================================================

def compute_lqr_gain(A, B):
    """
    Solve the discrete algebraic Riccati equation.
    """

    P = solve_discrete_are(
        A,
        B,
        Q,
        R_LQR,
    )

    K = np.linalg.solve(
        R_LQR + B.T @ P @ B,
        B.T @ P @ A,
    )

    return K


# ============================================================
# LQR CONTROLLER
# ============================================================

def lqr_controller(K):
    state = get_state(data)

    # Target: upright, stationary, at initial X position
    target = np.array([
        TARGET_POSITION,
        0.0,
        0.0,
        0.0,
    ])

    error = state - target

    torque = float(-(K @ error).item())

    torque = float(
        np.clip(
            torque,
            -MOTOR_LIMIT,
            MOTOR_LIMIT,
        )
    )

    return torque, state


# ============================================================
# MAIN
# ============================================================

def main():
    global TARGET_POSITION

    # Linearize around upright equilibrium
    A, B = linearize()

    print("\nA matrix:")
    print(A)

    print("\nB matrix:")
    print(B)

    K = compute_lqr_gain(A, B)

    print("\nLQR Gain K:")
    print(K)

    # Check local linear closed-loop stability
    eigenvalues = np.linalg.eigvals(A - B @ K)

    print("\nClosed-loop eigenvalues:")
    print(eigenvalues)

    if np.any(np.abs(eigenvalues) >= 1.0):
        print("WARNING: Linearized closed-loop system is not stable.")

    reset_robot()

    TARGET_POSITION = float(data.qpos[qpos_adr])

    step = 0

    with mj.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            start = time.perf_counter()

            torque, state = lqr_controller(K)

            data.ctrl[:] = torque

            mj.mj_step(model, data)

            if step % 100 == 0:
                print(
                    f"t={data.time:.2f} | "
                    f"x={state[0]:+.3f} m | "
                    f"pitch={np.degrees(state[1]):+.2f} deg | "
                    f"vx={state[2]:+.3f} m/s | "
                    f"pitch_rate={state[3]:+.3f} rad/s | "
                    f"u={torque:+.3f}"
                )

            viewer.sync()

            remaining = (
                model.opt.timestep
                - (time.perf_counter() - start)
            )

            if remaining > 0:
                time.sleep(remaining)

            step += 1


if __name__ == "__main__":
    main()