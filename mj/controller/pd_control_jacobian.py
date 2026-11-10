from pathlib import Path
import time

import mujoco as mj
import mujoco.viewer
import numpy as np


def get_chassis_jacobian(model, data, body_id):
    """Compute chassis linear and angular Jacobians."""

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mj.mj_jacBody(
        model,
        data,
        jacp,
        jacr,
        body_id,
    )

    return jacp, jacr

def get_chassis_velocity(model, data, body_id):
    jacp, jacr = get_chassis_jacobian(model, data, body_id)

    linear_velocity = jacp @ data.qvel
    angular_velocity = jacr @ data.qvel

    return linear_velocity, angular_velocity

MODEL_PATH = Path(__file__).resolve().parent.parent / "../robot_rl/assets/balance_bot/scene.xml"

KP = 5.0
KD = .3

MOTOR_LIMIT = 3.0
CONTROL_SIGN = 1.0

INITIAL_PITCH = 0.08
PRINT_INTERVAL = 100

model = mj.MjModel.from_xml_path(str(MODEL_PATH))
data = mj.MjData(model)

actuator_names =    [model.actuator(i).name for i in range(model.nu)]
actuator_id =       [model.actuator(i).id for i in range(model.nu)]

sensor_names =  [model.sensor(i).name for i in range(model.nsensor)]
sensor_ids =    [model.sensor(i).id for i in range(model.nsensor)]
sensor_adrs =   [model.sensor(i).adr[0] for i in range(model.nsensor)]

imu_site_id = model.site("imu_site").id

if len(actuator_names) < 2:
    raise ValueError("Robot must have at least two actuators.")

def find_sensor(sensor_type):
    for i in range(model.nsensor):
        if model.sensor_type[i] == sensor_type:
            return i

    raise ValueError(f"Sensor type {sensor_type} not found.")

quat_id = find_sensor(mj.mjtSensor.mjSENS_FRAMEQUAT)
gyro_id = find_sensor(mj.mjtSensor.mjSENS_GYRO)
accel_id = find_sensor(mj.mjtSensor.mjSENS_ACCELEROMETER)

quat_adr = model.sensor_adr[quat_id]
gyro_adr = model.sensor_adr[gyro_id]
accel_adr = model.sensor_adr[accel_id]

quat_dim = model.sensor_dim[quat_id]
gyro_dim = model.sensor_dim[gyro_id]
accel_dim = model.sensor_dim[accel_id]

imu_site_id = int(model.sensor_objid[gyro_id])
chassis_id = int(model.site_bodyid[imu_site_id])

imu_site = model.site(imu_site_id).name
chassis_id = model.body("chassis").id

print("Actuator_names:", actuator_names)
print("Actuator_ids:", actuator_id)

print("Sensors_names:", sensor_names)
print("Sensor_ids:", sensor_ids)

print("Sensor_adrs:", sensor_adrs)
print("Sensor_adrs:", model.sensor_adr)
print("Sensor_dims:", model.sensor_dim)
print("Imu Site: ", imu_site, imu_site_id)

def get_pitch():
    quat = data.sensordata[quat_adr:quat_adr + quat_dim]

    rot = np.zeros(9)
    mj.mju_quat2Mat(rot, quat)
    R = rot.reshape(3, 3)
    pitch = np.arctan2(-R[2, 0], R[0, 0])

    return float(pitch)

def pd_controller(pitch, pitch_rate):

    torque = CONTROL_SIGN * (
        KP * pitch + KD * pitch_rate
    )

    return float(np.clip(torque, -MOTOR_LIMIT, MOTOR_LIMIT))

def reset_robot():
    mj.mj_resetData(model, data)

    data.qpos[3:7] = [
        np.cos(INITIAL_PITCH / 2),
        0.0,
        np.sin(INITIAL_PITCH / 2),
        0.0,
    ]

    mj.mj_forward(model, data)

def main():
    reset_robot()

    step = 0

    with mj.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            start = time.perf_counter()

            mj.mj_forward(model, data)

            linear_vel, angular_vel = get_chassis_velocity(model, data, chassis_id)
            gyro_local = data.sensor(gyro_id).data.copy()
            R_imu = data.site_xmat[imu_site_id].reshape(3, 3)
            omega_imu = R_imu.T @ angular_vel

            error = np.linalg.norm(gyro_local - omega_imu)
            
            pitch = get_pitch()
            pitch_rate = float(angular_vel[1])

            torque = pd_controller(pitch, pitch_rate)
            data.ctrl[actuator_id] = torque

            if step % PRINT_INTERVAL == 0:
                print(
                    # f"Time: {data.time:.2f} s | "
                    # f"Pitch: {np.degrees(pitch):+.2f}° | "
                    # f"Pitch rate: {pitch_rate:+.3f} rad/s | "
                    # f"Torque: {torque:+.3f} Nm "
                    # f"lin_vel: {linear_vel} m/s "
                    f"gyro_vel  : {gyro_local} rad/s \n"
                    f"ang_vel   : {angular_vel} rad/s \n"
                    f"omega_vel : {omega_imu} rad/s "
                    # f"ang_vel: {angular_vel} rad/s "
                )

            mj.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)

            step += 1

if __name__ == "__main__":
    main()