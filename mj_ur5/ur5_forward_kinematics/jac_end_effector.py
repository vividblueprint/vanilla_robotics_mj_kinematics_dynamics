import numpy as np
import utility as rm
from forward_kinematics import forward_kinematics

def jac_end_effector(q):
    robot, fk = forward_kinematics(q)

    o01 = fk.H_globals[1][:3, 3]
    o02 = fk.H_globals[2][:3, 3]
    o03 = fk.H_globals[3][:3, 3]
    o04 = fk.H_globals[4][:3, 3]
    o05 = fk.H_globals[5][:3, 3]
    o06 = fk.H_globals[6][:3, 3]

    e0 = fk.end_eff_pos

    n1 = fk.joint_axises[1]
    n2 = fk.joint_axises[2]
    n3 = fk.joint_axises[3]
    n4 = fk.joint_axises[4]
    n5 = fk.joint_axises[5]
    n6 = fk.joint_axises[6]

    R00 = np.eye(3)
    R01 = fk.H_globals[1][:3, :3]
    R02 = fk.H_globals[2][:3, :3]
    R03 = fk.H_globals[3][:3, :3]
    R04 = fk.H_globals[4][:3, :3]
    R05 = fk.H_globals[5][:3, :3]
    R06 = fk.H_globals[6][:3, :3]

    Jv_E = np.column_stack([
        rm.vec2skew(R00@n1)@(e0 - o01),
        rm.vec2skew(R01@n2)@(e0 - o02),
        rm.vec2skew(R02@n3)@(e0 - o03),
        rm.vec2skew(R03@n4)@(e0 - o04),
        rm.vec2skew(R04@n5)@(e0 - o05),
        rm.vec2skew(R05@n6)@(e0 - o06) 
    ])

    Jw_E = np.column_stack([
        R00 @ n1,
        R01 @ n2,
        R02 @ n3,
        R03 @ n4,
        R04 @ n5,
        R05 @ n6
    ])

    J_E = np.vstack([Jv_E, Jw_E])

    return J_E