import numpy as np
import utility as rm
from forward_kinematics import forward_kinematics
from animate import animate

q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
fk = forward_kinematics(q)

print("Position of the end-effector:")
print(fk[1].end_eff_pos)
print('**********')

print("Rotation matrix of the end-effector:")
print(fk[1].end_eff_rot)
print('**********')

print("Euler angles of the end-effector:")
print(rm.rotation2euler(fk[1].end_eff_rot))
print('**********')

print("Quaternion of the end-effector:")
print(rm.rotation2quat(fk[1].end_eff_rot))

animate(fk)