from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "../robot_rl/assets/balance_bot/scene.xml"

KP = 5.0          # Pitch proportional gain
KD = 0.3           # Pitch derivative gain
KV = 04.9           # Forward velocity damping

MOTOR_LIMIT = 3.0
CONTROL_SIGN = 1.0
INITIAL_PITCH = 0.08

# Assumes the robot moves forward along its local +X axis
FORWARD_AXIS = np.array([1.0, 0.0, 0.0])
PITCH_AXIS = np.array([0.0, 1.0, 0.0])

model = mj.MjModel.from_xml_path(str(MODEL_PATH))
data = mj.MjData(model)

if model.nu != 2:
    raise ValueError(f"Expected 2 wheel actuators, found {model.nu}")

left_id = 0
right_id = 1

freejoint_ids = [
    i
    for i in range(model.njnt)
    if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE
]

if len(freejoint_ids) != 1:
    raise ValueError("Expected exactly one chassis freejoint.")

freejoint_id = freejoint_ids[0]

chassis_id = int(model.jnt_bodyid[freejoint_id])
qpos_adr = int(model.jnt_qposadr[freejoint_id])

print("Chassis:", model.body(chassis_id).name)
print("Actuators:", [model.actuator(i).name for i in range(model.nu)])

def get_rotation_matrix():
    quat = data.qpos[qpos_adr + 3:qpos_adr + 7]

    rot = np.zeros(9)
    mj.mju_quat2Mat(rot, quat)

    return rot.reshape(3, 3)


def get_pitch():
    R = get_rotation_matrix()

    return float(
        np.arctan2(-R[2, 0], R[0, 0])
    )

def get_jacobians():
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mj.mj_jacBody(
        model,
        data,
        jacp,
        jacr,
        chassis_id,
    )

    return jacp, jacr

def get_robot_state():
    jacp, jacr = get_jacobians()

    v_world = jacp @ data.qvel
    omega_world = jacr @ data.qvel

    R = get_rotation_matrix()

    v_body = R.T @ v_world
    omega_body = R.T @ omega_world

    pitch = get_pitch()

    pitch_rate = float(PITCH_AXIS @ omega_body)

    forward_velocity = float(FORWARD_AXIS @ v_body)

    return pitch, pitch_rate, forward_velocity

def controller():
    pitch, pitch_rate, forward_velocity = get_robot_state()

    torque = CONTROL_SIGN * (
        KP * pitch
        + KD * pitch_rate
        + KV * forward_velocity
    )

    torque = float(
        np.clip(torque, -MOTOR_LIMIT, MOTOR_LIMIT)
    )

    return torque, pitch, pitch_rate, forward_velocity

def reset_robot():
    mj.mj_resetData(model, data)

    quat = np.zeros(4)

    mj.mju_axisAngle2Quat(
        quat,
        PITCH_AXIS,
        INITIAL_PITCH,
    )

    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

    mj.mj_forward(model, data)

def main():
    reset_robot()

    step = 0

    with mj.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()

            torque, pitch, pitch_rate, velocity = controller()

            data.ctrl[left_id] = torque
            data.ctrl[right_id] = torque

            mj.mj_step(model, data)

            if step % 100 == 0:
                print(
                    f"t={data.time:.2f} | "
                    f"pitch={np.degrees(pitch):+.2f} deg | "
                    f"pitch_rate={pitch_rate:+.3f} rad/s | "
                    f"velocity={velocity:+.3f} m/s | "
                    f"torque={torque:+.3f} Nm"
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