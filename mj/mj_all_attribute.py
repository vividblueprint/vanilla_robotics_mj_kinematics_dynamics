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

for i, name in enumerate(dir(data)):
    if not name.startswith("_"):
        value = getattr(data, name)

        if not isinstance(value, np.ndarray):
            print(f"{i} {name}: {value}")

        if isinstance(value, np.ndarray):
            print(f"{i} {name}: shape={value.shape}")

actuators = model.actuator
print(dir(actuators))