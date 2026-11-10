import mujoco
import time
from mujoco import viewer
from pathlib import Path

MJCF_file=Path(__file__).parent / "assets/robots/scene.xml"

model = mujoco.MjModel.from_xml_path(str(MJCF_file))
data = mujoco.MjData(model)

run_sim_time = 10

with viewer.launch_passive(model, data) as viewer:
    start = time.perf_counter()
    while viewer.is_running() and time.perf_counter() - start < run_sim_time:
        step_start = time.perf_counter()

        mujoco.mj_step(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()
        
        time_until_next_step = model.opt.timestep - (time.perf_counter() - step_start)
        print(model.opt.timestep)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

