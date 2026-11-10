import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utility as rm

def animate(fk):
    robot, fk = fk
    
    H_globals = fk.H_globals
    end_eff_pos = fk.end_eff_pos
    end_eff_rot = fk.end_eff_rot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot Animation')
    ax.grid(True)

    # Set axis limits as [-1, 1, -1, 1, -1, 1]
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=33)

    prev = np.zeros(3)
    for i, transform in H_globals.items():
        curr = transform[:-1, -1] # [:3, 3]
        plt.plot([prev[0], curr[0]], [prev[1], curr[1]], [prev[2], curr[2]], linewidth=5, color="red")
        prev = curr
        
        # ///////////////////////////////////////////////////
        # if i == len(fH_globals):
        #     end_eff_pos = (transform @ np.append(robot.params.end_eff_pos_local, 1))[:3]
        #     end_eff_rot = transform[:3, :3] @ rm.quat2rotation(robot.params.end_eff_quat_local)
        #     plt.plot([prev[0], end_eff_pos[0]], [prev[1], end_eff_pos[1]], [prev[2], end_eff_pos[2]], linewidth=5, color="red")
        # ///////////////////////////////////////////////////
    
    # ///////////////////////////////////////////////////
    # H_end_local = np.block([
    #     [rm.quat2rotation(robot.params.end_eff_quat_local), robot.params.end_eff_pos_local.reshape(-1, 1)],
    #     [np.zeros((1, 3)), np.ones((1, 1))]
    # ])

    # H_end = transform @ H_end_local

    # end_eff_pos = H_end[:-1, -1]
    # end_eff_rot = H_end[:3, :3]
    # ///////////////////////////////////////////////////

    plt.plot([prev[0], end_eff_pos[0]], [prev[1], end_eff_pos[1]], [prev[2], end_eff_pos[2]], linewidth=5, color="red")

    R = end_eff_rot
    origin = np.zeros(3)

    arrow_length = 0.2

    dir_x = R.dot(np.array([1, 0, 0]))
    dir_y = R.dot(np.array([0, 1, 0]))
    dir_z = R.dot(np.array([0, 0, 1]))

    ax.quiver(end_eff_pos[0], end_eff_pos[1], end_eff_pos[2], dir_x[0], dir_x[1], dir_x[2], length=arrow_length, arrow_length_ratio=0.1, normalize=True, color="red")
    ax.quiver(end_eff_pos[0], end_eff_pos[1], end_eff_pos[2], dir_y[0], dir_y[1], dir_y[2], length=arrow_length, arrow_length_ratio=0.1, normalize=True, color="green")
    ax.quiver(end_eff_pos[0], end_eff_pos[1], end_eff_pos[2], dir_z[0], dir_z[1], dir_z[2], length=arrow_length, arrow_length_ratio=0.1, normalize=True, color="blue")
    ax.scatter(end_eff_pos[0], end_eff_pos[1], end_eff_pos[2], color="orange", s=50, label="end_eff")
    
    
    # plt.show(block=False)
    # plt.pause(3)
    # plt.close()
    plt.show()