from pathlib import Path

import mujoco
from mujoco.usd import exporter

ROOT = Path(__file__).resolve().parent

MJCF_file = ROOT / "assets/balance_bot/scene.xml"
OUTPUT_DIR = ROOT / "usd_output"

model = mujoco.MjModel.from_xml_path(str(MJCF_file))
data = mujoco.MjData(model)

exp = exporter.USDExporter(
    model=model,
    output_directory_root=str(OUTPUT_DIR),
    output_directory="balance_bot",
)

duration = 5
framerate = 60

mujoco.mj_forward(model, data)

# Record initial state
exp.update_scene(data=data)

while data.time < duration:
    mujoco.mj_step(model, data)

    if exp.frame_count < data.time * framerate:
        exp.update_scene(data=data)

print("Recorded frames:", exp.frame_count)

exp.save_scene(filetype="usd")

print("USD export completed!")