import mujoco
import time
from mujoco import viewer
from pathlib import Path

MJCF_file=Path(__file__).parent / "assets/robots/scene.xml"

model = mujoco.MjModel.from_xml_path(str(MJCF_file))
data = mujoco.MjData(model)

paused = False
run_sim_time = 10

def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

with viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    start = time.perf_counter()
    while viewer.is_running() and time.perf_counter() - start < run_sim_time:

        if not paused:
           mujoco.mj_step(model, data)
           viewer.sync()

