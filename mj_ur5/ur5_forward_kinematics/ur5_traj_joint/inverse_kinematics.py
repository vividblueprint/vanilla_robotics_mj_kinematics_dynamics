import numpy as np
from mj_ur5.ur5_forward_kinematics.ur5_traj_joint.forward_kinematics import forward_kinematics
import mj_ur5.ur5_forward_kinematics.ur5_traj_joint.utility as ram


def inverse_kinematics(q,X_ref):

    x_ref = X_ref[0];
    y_ref = X_ref[1];
    z_ref = X_ref[2];
    phi_ref = X_ref[3];
    theta_ref = X_ref[4];
    psi_ref = X_ref[5];

    rot_ref = ram.euler2rotation(np.array([phi_ref,theta_ref,psi_ref]))

    robot,sol = forward_kinematics(q)

    end_eff_pos = sol.end_eff_pos
    end_eff_rot = sol.end_eff_rot


    end_eff_euler = ram.rotation2euler(end_eff_rot)
    end_eff_quat = ram.rotation2quat(end_eff_rot)

    RRt = np.dot(end_eff_rot, rot_ref.T);

    x = end_eff_pos[0]
    y = end_eff_pos[1]
    z = end_eff_pos[2]
    phi = end_eff_euler[0];
    theta = end_eff_euler[1];
    psi = end_eff_euler[2];

    #return x-x_ref,y-y_ref,z-z_ref,phi-phi_ref,theta-theta_ref,psi-psi_ref
    return x-x_ref,y-y_ref,z-z_ref,RRt[0][0]-1,RRt[1][1]-1,RRt[2][2]-1
