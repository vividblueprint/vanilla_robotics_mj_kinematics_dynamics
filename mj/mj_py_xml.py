import mujoco
from mujoco import viewer
from pathlib import Path
import numpy as np
import mujoco_warp as mjw
import warp as wp

nworld = 4

# base spec: 1 body with 1 mesh geom, all mesh assets
spec = mujoco.MjSpec()

mesh_a = spec.add_mesh()
mesh_a.name = "mesh_a"
mesh_a.uservert = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

mesh_b = spec.add_mesh()
mesh_b.name = "mesh_b"
mesh_b.uservert = [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]

body = spec.worldbody.add_body()
body.pos = [0, 0, 1]
body.add_freejoint()
geom = body.add_geom()
geom.name = "obj"
geom.type = mujoco.mjtGeom.mjGEOM_MESH
geom.meshname = "mesh_a"

xml = spec.to_xml()
print(xml) 