import numpy as np
import utility as rm
from forward_kinematics import forward_kinematics

def jac_end(q):
    robot, fk = forward_kinematics(q)

    o01 = fk.H_globals[1][:3, 3]
    o02 = fk.H_globals[2][:3, 3]
    o03 = fk.H_globals[3][:3, 3]
    o04 = fk.H_globals[4][:3, 3]
    o05 = fk.H_globals[5][:3, 3]
    o06 = fk.H_globals[6][:3, 3]

    g1 = (fk.H_globals[1]@np.append(fk.iposes[1], 1))[:3]
    g2 = (fk.H_globals[2]@np.append(fk.iposes[2], 1))[:3]
    g3 = (fk.H_globals[3]@np.append(fk.iposes[3], 1))[:3]
    g4 = (fk.H_globals[4]@np.append(fk.iposes[4], 1))[:3]
    g5 = (fk.H_globals[5]@np.append(fk.iposes[5], 1))[:3]
    g6 = (fk.H_globals[6]@np.append(fk.iposes[6], 1))[:3]

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

    Jv_g6 = np.column_stack([
        rm.vec2skew(R00@n1)@(g6 - o01),
        rm.vec2skew(R01@n2)@(g6 - o02),
        rm.vec2skew(R02@n3)@(g6 - o03),
        rm.vec2skew(R03@n4)@(g6 - o04),
        rm.vec2skew(R04@n5)@(g6 - o05),
        rm.vec2skew(R05@n6)@(g6 - o06) 
    ])

    Jv_g5 = np.column_stack([
        rm.vec2skew(R00@n1)@(g5 - o01),
        rm.vec2skew(R01@n2)@(g5 - o02),
        rm.vec2skew(R02@n3)@(g5 - o03),
        rm.vec2skew(R03@n4)@(g5 - o04),
        rm.vec2skew(R04@n5)@(g5 - o05),
        np.zeros((3, 1))
    ])

    Jv_g4 = np.column_stack([
        rm.vec2skew(R00@n1)@(g4 - o01),
        rm.vec2skew(R01@n2)@(g4 - o02),
        rm.vec2skew(R02@n3)@(g4 - o03),
        rm.vec2skew(R02@n3)@(g4 - o04),
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    Jv_g3 = np.column_stack([
        rm.vec2skew(R00@n1)@(g3 - o01),
        rm.vec2skew(R01@n2)@(g3 - o02),
        rm.vec2skew(R02@n3)@(g3 - o03),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    Jv_g2 = np.column_stack([
        rm.vec2skew(R00@n1)@(g2 - o01),
        rm.vec2skew(R01@n2)@(g2 - o02),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    Jv_g1 = np.column_stack([
        rm.vec2skew(R00@n1)@(g1 - o01),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    Jw_g6 = np.column_stack([
        R00 @ n1,
        R01 @ n2,
        R02 @ n3,
        R03 @ n4,
        R04 @ n5,
        R04 @ n6
    ])

    Jw_g5 = np.column_stack([
        R00 @ n1,
        R01 @ n2,
        R02 @ n3,
        R03 @ n4,
        R04 @ n5,
        np.zeros((3, 1))
    ])

    Jw_g4 = np.column_stack([
        R00 @ n1,
        R01 @ n2,
        R02 @ n3,
        R03 @ n4,
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    Jw_g3 = np.column_stack([
        R00 @ n1,
        R01 @ n2,
        R02 @ n3,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    Jw_g2 = np.column_stack([
        R00 @ n1,
        R01 @ n2,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    Jw_g1 = np.column_stack([
        R00 @ n1,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1))
    ])

    J_g6 = np.vstack([Jv_g6, Jw_g6])
    J_g5 = np.vstack([Jv_g5, Jw_g5])
    J_g4 = np.vstack([Jv_g4, Jw_g4])
    J_g3 = np.vstack([Jv_g3, Jw_g3])
    J_g2 = np.vstack([Jv_g2, Jw_g2])
    J_g1 = np.vstack([Jv_g1, Jw_g1])
    
    F_g6 = robot.body[6].mass * robot.params.gravity
    F_g5 = robot.body[5].mass * robot.params.gravity
    F_g4 = robot.body[4].mass * robot.params.gravity
    F_g3 = robot.body[3].mass * robot.params.gravity
    F_g2 = robot.body[2].mass * robot.params.gravity
    F_g1 = robot.body[1].mass * robot.params.gravity

    tau = Jv_g1.T@F_g1 + Jv_g2.T@F_g2 + Jv_g3.T@F_g3 + Jv_g4.T@F_g4 + Jv_g5.T@F_g5 + Jv_g6.T@F_g6

    return J_g1, J_g2, J_g3, J_g4, J_g5, J_g6