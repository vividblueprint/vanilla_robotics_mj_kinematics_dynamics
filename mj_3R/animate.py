import matplotlib.pyplot as plt

def animate(sol):
    plt.plot([sol.o[0], sol.p[0]], [sol.o[1], sol.p[1]], linewidth=5, color="red")
    plt.plot([sol.p[0], sol.q[0]], [sol.p[1], sol.q[1]], linewidth=5, color="blue")
    plt.plot([sol.q[0], sol.e[0]], [sol.q[1], sol.e[1]], linewidth=5, color="green")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect("equal")

    # plt.axis('square')

    plt.show(block=False)
    plt.pause(5)
    plt.close()