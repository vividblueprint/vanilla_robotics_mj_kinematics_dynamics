import numpy as np
import utility as rm

class Robot:
    class Body:
        def __init__(self, parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range):
            self.parent = parent
            self.name = name
            self.pos = np.array(pos)
            self.quat = np.array(quat)
            self.ipos = np.array(ipos)
            self.iquat = np.array(iquat)
            self.mass = mass
            self.inertia = np.array(inertia)
            self.joint_axis = np.array(joint_axis)
            self.joint_range = np.array(joint_range)

    class Params:
        def __init__(self):
            self.base_quat = rm.quat_normalize(np.array([1, 0, 0, 1]))
            self.end_eff_pos_local = np.array([0, 0.1, 0])
            self.end_eff_quat_local = rm.quat_normalize(np.array([-1, 1, 1, 1]))
            self.gravity = np.array([0, 0, -9.81])
            
    def __init__(self):
        self.body = {}
        self.params = Robot.Params()

        
    def add_body(self, body_id, parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range):
        self.body[body_id] = Robot.Body(parent, name, pos, quat, ipos, iquat, mass, inertia, joint_axis, joint_range)


robot = Robot()

robot.add_body(1, parent="ground", name="shoulder_link", pos=[0, 0, 0.163], quat=[1, 0, 0, 0], ipos=[0, 0, 0], iquat=[1, 0, 0, 0], mass=3.7, inertia=[0.0102675, 0.0102675, 0.0066], joint_axis=[0, 0, 1], joint_range=[-6.28319, 6.28319])
robot.add_body(2, parent='shoulder_link', name='upper_arm_link', pos=[0, 0.138, 0], quat=[1, 0, 1, 0], ipos=[0, 0, 0.2125], iquat=[1, 0, 0, 0], mass=8.393, inertia=[0.133886, 0.133886, 0.0151074], joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319])
robot.add_body(3, parent='upper_arm_link', name='forearm_link', pos=[0, -0.131, 0.425], quat=[1, 0, 0, 0], ipos=[0, 0, 0.196], iquat=[1, 0, 0, 0], mass=2.275, inertia=[0.0311796, 0.0311796, 0.004095], joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319])
robot.add_body(4, parent='forearm_link', name='wrist_1_link', pos=[0, 0, 0.392], quat=[1, 0, 1, 0], ipos=[0, 0.127, 0], iquat=[1, 0, 0, 0], mass=1.219, inertia=[0.0025599, 0.0025599, 0.0021942], joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319])
robot.add_body(5, parent='wrist_1_link', name='wrist_2_link', pos=[0, 0.127, 0], quat=[1, 0, 0, 0], ipos=[0, 0, 0.1], iquat=[1, 0, 0, 0], mass=1.219, inertia=[0.0025599, 0.0025599, 0.0021942], joint_axis=[0, 0, 1], joint_range=[-6.28319, 6.28319])
robot.add_body(6, parent='wrist_2_link', name='wrist_3_link', pos=[0, 0, 0.1], quat=[1, 0, 0, 0], ipos=[0, 0.0771683, 0], iquat=[1, 0, 0, 1], mass=0.1889, inertia=[0.000132134, 9.90863e-05, 9.90863e-05], joint_axis=[0, 1, 0], joint_range=[-6.28319, 6.28319])

# id = 2
# print(robot.body[id].parent)
# print(robot.body[id].name)
# print(robot.body[id].pos)
# print(robot.body[id].quat)
# print(robot.body[id].ipos)
# print(robot.body[id].iquat)
# print(robot.body[id].mass)
# print(robot.body[id].inertia)
# print(robot.body[id].joint_axis)
# print(robot.body[id].joint_range)

for body_id, body in robot.body.items():
    body.quat = rm.quat_normalize(body.quat)
    body.iquat = rm.quat_normalize(body.iquat)

# print(robot.body[id].quat)
# print(robot.body[id].iquat)

# print(robot.params.base_quat)
# print(robot.params.end_eff_pos_local)
# print(robot.params.end_eff_quat_local)

# print(robot.body.items())