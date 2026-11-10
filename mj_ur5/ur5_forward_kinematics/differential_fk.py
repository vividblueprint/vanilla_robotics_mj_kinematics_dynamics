import numpy as np
import utility as rm
from types import SimpleNamespace
from forward_kinematics import forward_kinematics

def differential_kinematics(q, u):

    robot, fk = forward_kinematics(q)

    w00 = np.zeros(3)
    v00 = np.zeros(3)

    w_locals = {}
    w_globals = {}
    v_locals = {}
    v_globals = {}
    v_com_locals = {}
    v_com_globals = {}

    for i, _ in robot.body.items():
        w_local = fk.R_locals[i].T@w00 + u[i-1]*fk.joint_axises[i]
        v_local = fk.R_locals[i].T@(v00 + rm.vec2skew(w00)@fk.p_locals[i])

        v_com_locals[i] = v_local + rm.vec2skew(w_local)@fk.iposes[i]

        w_locals[i] = w_local
        v_locals[i] = v_local

        v_com_globals[i] = fk.H_globals[i][:3, :3] @ v_com_locals[i]
        w_globals[i] = fk.H_globals[i][:3, :3] @ w_locals[i]
        v_globals[i] = fk.H_globals[i][:3, :3] @ v_locals[i]

        v00 = v_local
        w00 = w_local
        
        # print(v_com_locals[i])
        # print(v_com_globals[i])
        # print(v_locals[i])
        # print(v_globals[i])
        # print(w_locals[i])
        # print(w_globals[i])

    v_tip_local = v_locals[6] + rm.vec2skew(w_locals[6])@fk.end_eff_pos_local
    v_tip_global = fk.H_globals[6][:3, :3]@v_tip_local
    w_tip_global = w_globals[6]

    # print(v_tip_local)
    # print(v_tip_global)
    # print(w_tip_global)

    dfk = SimpleNamespace(
        v_tip_local=v_tip_local,
        v_tip_global=v_tip_global,
        w_tip_global=w_tip_global,
        v_com_locals=v_com_locals,
        v_com_globals=v_com_globals,
        v_locals=v_locals,
        v_globals=v_globals,
        w_locals=w_locals,
        w_globals=w_globals
    )

    return robot, dfk

# /////////////////////////////////////////
# q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
# u = np.array([-0.2, 0.45, -0.67, -1.3, 2.8, -0.768])

# X_ref = np.array([0.5, 0.2, 0.5, 3.14, 0, 0])
# differential_kinematics(q, u)
# print(differential_kinematics(q))