import mujoco as mj
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
import os

xml_path = "two_link.xml"

dirname = os.path.dirname(__file__)
xml_path = os.path.join(dirname, xml_path)
model = mj.MjModel.from_xml_path(str(xml_path))

data = mj.MjData(model)

# Initial state
data.qpos[:] = [0, 0, 0]
data.qvel[:] = [0, 0, 0]

mj.mj_forward(model, data)

# t = []
# q = []
# qdot = []
# KE = []
# PE = []
# TE = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -90.0
        viewer.cam.distance = 6.0
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])

    while viewer.is_running() and data.time < 10:

        start = time.time()

        # MuJoCo solves forward dynamics and integrates
        mj.mj_step(model, data)

        # t.append(data.time)
        # q.append(data.qpos.copy())
        # qdot.append(data.qvel.copy())

        # Energy
        # KE.append(data.energy[1])
        # PE.append(data.energy[0])
        # TE.append(sum(data.energy))

        viewer.sync()

        remaining = model.opt.timestep - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
