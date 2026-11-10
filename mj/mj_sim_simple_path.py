import mujoco
from mujoco import viewer
from pathlib import Path

MJCF_file=Path(__file__).parent / "assets/robots/scene.xml"

model = mujoco.MjModel.from_xml_path(str(MJCF_file))
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()