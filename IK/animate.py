from matplotlib import pyplot as plt
#from types import SimpleNamespace

def animate(sol, X_ref=None):

    plt.plot([sol.o[0], sol.p[0]],[sol.o[1], sol.p[1]],linewidth=5, color='red')
    plt.plot([sol.p[0], sol.q[0]],[sol.p[1], sol.q[1]],linewidth=5, color='blue')
    plt.plot([sol.q[0], sol.e[0]],[sol.q[1], sol.e[1]],linewidth=5, color='green')


    # %Draw line from end of link 2 to end of link 3
    plt.plot([sol.e[0], sol.e_left1[0]],[sol.e[1], sol.e_left1[1]],linewidth=5, color='green')
    plt.plot([sol.e[0], sol.e_right1[0]],[sol.e[1], sol.e_right1[1]],linewidth=5, color='green')
    plt.plot([sol.e_left1[0], sol.e_left2[0]],[sol.e_left1[1], sol.e_left2[1]],linewidth=5, color='green')
    plt.plot([sol.e_right1[0], sol.e_right2[0]],[sol.e_right1[1], sol.e_right2[1]],linewidth=5, color='green')

    if X_ref is not None:
        x_ref = X_ref[0];
        y_ref = X_ref[1];
        plt.plot(x_ref,y_ref,marker='o',markersize=5,color='k')

    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.gca().set_aspect('equal')

    # plt.axis('square')
    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()
    plt.show()
