# MuJoCo Robotics Workspace

Python examples for learning robot kinematics, dynamics, control, and trajectory generation. The workspace includes a planar arm with three revolute joints (3R), a Universal Robots UR5e, a balance bot, and block models for comparing rotation interpolation methods.

## Setup

Use Python 3.12 or newer and `uv`. From the workspace root, install the dependencies from [`uv.lock`](uv.lock).

```sh
uv sync --locked
```

Run each example by its file path from the workspace root. MuJoCo viewer demos need a graphical session with OpenGL support. Interactive plots need a graphical Matplotlib backend.

## Run an example

```sh
# Print a basic NumPy matrix example
uv run --locked python basic/01.matrix.py

# Open a basic MuJoCo scene
uv run --locked python mj/mj_sim_simple_path.py

# Plot the reachable workspace of a planar arm
uv run --locked python IK/10.main_planar_reachable_space_node.py

# Compare forward kinematics with MuJoCo for the planar 3R arm and UR5e
uv run --locked python mj_3R/15.mj_planar_3d_fk.py
uv run --locked python mj_ur5/ur5_forward_kinematics/18.mj_fk.py

# Compare Euler-angle, normalized quaternion, and rotation-matrix interpolation
uv run --locked python 3_block/28.mj_orientation_compare.py
```

## Workspace layout

| Directory | Contents |
| --- | --- |
| [`basic/`](basic/) | NumPy matrices, Matplotlib animations, Euler rotations, and gimbal lock. |
| [`IK/`](IK/) | Planar forward kinematics, reachable workspace plots, and analytic and numerical inverse kinematics. |
| [`mj/`](mj/) | MuJoCo viewer examples, robot models, balance-bot controllers, and USD export. |
| [`mj_template/`](mj_template/) | A starter MuJoCo simulation loop and its companion MJCF model. |
| [`mj_3R/`](mj_3R/) | Planar 3R forward kinematics, Jacobians, Lagrange and Newton–Euler dynamics, and Cartesian trajectory generation. |
| [`mj_ur5/`](mj_ur5/) | UR5e kinematics, circular path tracking, dynamics, joint-space trajectory generation, and the robot model. |
| [`3_block/`](3_block/) | Orientation interpolation comparisons and a quaternion SLERP demo. |
| [`joint_cartesian_profiles/`](joint_cartesian_profiles/) | Cubic and quintic polynomial profiles and cubic splines. |

## Notes for specific examples

- The balance-bot controller scripts in [`mj/controller/`](mj/controller/) point to `robot_rl/assets/balance_bot/scene.xml`, which is absent from this checkout. The bundled model is at [`mj/assets/balance_bot/scene.xml`](mj/assets/balance_bot/scene.xml).
- The [UR5e joint-space trajectory demo](mj_ur5/ur5_forward_kinematics/ur5_traj_joint/mj_ur5_traj_gen_joint.py) uses workspace-root imports. From the workspace root, run it with `PYTHONPATH=. uv run --locked python mj_ur5/ur5_forward_kinematics/ur5_traj_joint/mj_ur5_traj_gen_joint.py`.
- The [Warp example](mj/mj_py_xml.py) imports `mujoco_warp` and `warp`, and the [USD exporter](mj/mj_sim_usd_exporter.py) requires the `pxr` bindings. These optional dependencies are outside the base environment.

The bundled UR5e model has its own [model documentation](mj_ur5/universal_robots_ur5e/README.md) and [BSD 3-Clause license](mj_ur5/universal_robots_ur5e/LICENSE).
