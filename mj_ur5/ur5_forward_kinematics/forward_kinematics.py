import numpy as np
from types import SimpleNamespace

import utility as rm
from robot_ur5 import robot


def forward_kinematics(q):
    H_base = np.block([
        [rm.quat2rotation(robot.params.base_quat), np.zeros((3, 1))],
        [np.zeros((1, 3)), np.ones((1, 1))]
    ])

    iposes = {}
    joint_axises = {}
    p_locals = {}
    R_locals = {}
    H_globals = {}

    for i, body in robot.body.items():
        # Joint axis
        joint_axis = body.joint_axis / np.linalg.norm(
            body.joint_axis + 1e-10
        )

        # Joint rotation
        R = rm.rot_axis_angle(joint_axis, q[i - 1])

        # Local transformation
        R_local = rm.quat2rotation(body.quat) @ R
        p_local = body.pos.reshape(-1, 1)

        H_local = np.block([
            [R_local, p_local],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ])

        # Global transformation
        H_base = H_base @ H_local

        # Store results
        iposes[i] = body.ipos
        joint_axises[i] = joint_axis.copy()
        p_locals[i] = body.pos.copy()
        R_locals[i] = R_local.copy()
        H_globals[i] = H_base.copy()

        # Update robot body for RNEA and Jacobian calculations
        body.o_local = p_locals[i]
        body.R_local = R_locals[i]
        body.H_global = H_globals[i]

    end_eff_pos = (
        H_base @ np.append(robot.params.end_eff_pos_local, 1)
    )[:3]

    end_eff_rot = (
        H_base[:3, :3]
        @ rm.quat2rotation(robot.params.end_eff_quat_local)
    )

    fk = SimpleNamespace(
        iposes=iposes,
        end_eff_pos_local=robot.params.end_eff_pos_local,
        end_eff_quat_local=robot.params.end_eff_quat_local,
        end_eff_pos=end_eff_pos,
        end_eff_rot=end_eff_rot,
        joint_axises=joint_axises,
        p_locals=p_locals,
        R_locals=R_locals,
        H_globals=H_globals,
    )

    return robot, fk