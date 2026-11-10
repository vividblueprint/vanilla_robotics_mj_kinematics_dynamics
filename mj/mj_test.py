from xml.etree.ElementInclude import include

import mujoco as m
from mujoco import viewer
from pathlib import Path

import numpy as np

MJCF_file=Path(__file__).parent / "assets/robots/scene.xml"
DEFAULT_MODEL = Path(__file__).parent.parent / "robot_rl/assets/balance_bot/scene.xml"

# model = m.MjModel.from_xml_path(str(MJCF_file))
model = m.MjModel.from_xml_path(str(DEFAULT_MODEL))
data = m.MjData(model)

print("//////////| Model Dimensions |//////////")
print("nq (positions):", model.nq)
print("nv (velocities):", model.nv)
print("nu (actuators):", model.nu)
print("na (activations):", model.na)
print("nbody:", model.nbody)
print("njnt:", model.njnt)
print("ngeom:", model.ngeom)
print("nsensor:", model.nsensor)
print("nsite:", model.nsite)
print("ncon (contacts):", data.ncon)

print("//////////|joint names|//////////")

joint_names = [model.joint(i).name for i in range(model.njnt)]

print(f"Joints: {joint_names}")

print("//////////|actuator names|//////////")
actuator_names = [model.actuator(i).name for i in range(model.nu)]

print(f"Actuator: {actuator_names}")

print("//////////| sensoers |//////////")

sensor_names = [model.sensor(i).name for i in range(model.nsensor)]
# quat = data.sensordata[imu_quat_adr : imu_quat_adr + 4]
print(sensor_names)
print(model.sensor_adr)
print(model.sensor_dim)

print("//////////|joint names|//////////")
lwj_id  = m.mj_name2id(model, m.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
lw_dof_idx = model.jnt_dofadr[0]
# num   = model.jnt_dofnum[lwj_id]

print(model.joint(0))
print(lw_dof_idx)

m.mj_forward(model, data)    # to recompute states after manual updates

# with viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         m.mj_step(model, data)
#         viewer.sync()
print("//////////|joint|//////////")
print(data.qpos)
print(data.qvel)

print("//////////||//////////")

import numpy as np

for name in dir(model):
    if not name.startswith("_"):
        value = getattr(model, name)
        print(f"{name}: {value}")

for name in dir(model):
    if not name.startswith("_"):
        value = getattr(model, name)

        if isinstance(value, np.ndarray):
            print(f"{name}: shape={value.shape}")