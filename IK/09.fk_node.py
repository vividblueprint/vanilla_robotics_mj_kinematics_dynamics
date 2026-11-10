import numpy as np
from fk import forward_kinematics
from animate import animate

l1 = 1
l2 = 1
l3 = 0.25

q = np.array([np.pi/2, -np.pi/2, np.pi/2])
sol = forward_kinematics(q)
print(sol)

#print(sol.e)
#print(sol.q)
animate(sol)