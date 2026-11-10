import numpy as np
from fk import forward_kinematics
from matplotlib import pyplot as plt

l1 = 1; l2 = 1; l3 = 0.25

theta_range = np.array(
    [
        [0, np.pi/2],
        [-np.pi/2, np.pi/2],
        [-np.pi/2,np.pi/2]
    ]
)

end_eff_x = []
end_eff_y = []

n = 40
theta1 = np.linspace(theta_range[0,0],theta_range[0,1],n)
theta2 = np.linspace(theta_range[1,0],theta_range[1,1],n)
theta3 = np.linspace(theta_range[2,0],theta_range[2,1],n)

for i in range(0,len(theta1)):
    for j in range(0,len(theta2)):
        for k in range(0,len(theta3)):
            sol = forward_kinematics((theta1[i],theta2[j],theta3[k]))
            end_eff_x.append(sol.e[0])
            end_eff_y.append(sol.e[1])

# Create the plot with markers
plt.figure(figsize=(8, 6))
plt.scatter(end_eff_x,end_eff_y, marker='o', color='blue', label='reachable space')

# Add labels, title, and legend
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Reachable space')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
