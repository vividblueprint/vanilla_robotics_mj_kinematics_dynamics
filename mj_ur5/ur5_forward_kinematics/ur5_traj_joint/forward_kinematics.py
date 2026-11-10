import numpy as np
from mj_ur5.ur5_forward_kinematics.ur5_traj_joint.robot_data import robot
from types import SimpleNamespace
import mj_ur5.ur5_forward_kinematics.ur5_traj_joint.utility as ram

def forward_kinematics(q):

    base_quat = robot.params.base_quat
    R_base = ram.quat2rotation(base_quat);

    end_eff_pos_local = robot.params.end_eff_pos_local
    end_eff_quat_local = robot.params.end_eff_quat_local
    R_end_eff = ram.quat2rotation(end_eff_quat_local )

    for i, body in robot.body.items():
        quat = robot[i].quat
        joint_axis = robot[i].joint_axis
        axis_id = np.argmax(np.abs(joint_axis))
        angle = q[i-1]  # body_id goes from 1 to 6 but q goes from 0 to 5.

        R_q = ram.rotation(angle, axis_id)
        robot[i].R_local= ram.quat2rotation(quat) @ R_q
        robot[i].o_local = robot[i].pos

        robot[i].H_local = np.block([
            [robot[i].R_local, robot[i].o_local.reshape(-1, 1)], #does 3x1
            [np.zeros((1, 3)), 1]
            ])

    # Compute global homogeneous matrices
    H_base = np.block([
        [R_base, np.zeros((3, 1))],
        [np.zeros((1, 3)), 1]
    ])
    temp = H_base
    for i, body in robot.body.items():
        # Compute global transformation matrix
        robot[i].H_global = temp @ robot[i].H_local
        # Update temp for the next iteration
        temp = robot[i].H_global

    #end_eff stuff
    end_eff_pos = robot[6].H_global @ np.append(end_eff_pos_local, 1)  # Homogeneous coordinates
    end_eff_pos = end_eff_pos[:3]  # Extract translation

    # Extract the rotation matrix from the global transformation matrix and R_end_eff
    end_eff_rot = robot[6].H_global[:3, :3] @ R_end_eff

    sol = SimpleNamespace(
        end_eff_pos=np.array(end_eff_pos),
        end_eff_rot=np.array(end_eff_rot),
        )

    return robot,sol
