from pathlib import Path
from sre_constants import ASSERT
import time

import mujoco
import mujoco.viewer

mesh_path=Path(__file__).parent / "assets/robots/meshes/wheel_left.stl"

XML = """
<mujoco model="mesh_test">

    <option timestep="0.005"/>

    <asset>
        <mesh name="wheel_left"
              file="wheel_left.stl"
              scale="0.001 0.001 0.001"/>

        <texture type="skybox"
                 builtin="gradient"
                 rgb1="0.5 0.7 1"
                 rgb2="0.95 0.95 1"
                 width="256"
                 height="256"/>

        <texture name="grid"
                 type="2d"
                 builtin="checker"
                 rgb1="0.2 0.3 0.4"
                 rgb2="0.3 0.4 0.5"
                 width="512"
                 height="512"
                 mark="cross"
                 markrgb="0.8 0.8 0.8"/>

        <material name="grid"
                  texture="grid"
                  texrepeat="5 5"/>
    </asset>

    <worldbody>
        <light pos="0 0 1.5"
               dir="0 0 -1"/>

        <geom name="ground"
              type="plane"
              size="5 5 0.1"
              material="grid"/>

        <body>
            <freejoint/>
            <geom type="mesh" name="gizmo" mesh="wheel_left"/>
        </body>
    </worldbody>

</mujoco>
"""

ASSETS=dict()

with open(mesh_path, 'rb') as f:
  ASSETS['wheel_left.stl'] = f.read()

# Compile MuJoCo model
model = mujoco.MjModel.from_xml_string(XML, ASSETS)
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()