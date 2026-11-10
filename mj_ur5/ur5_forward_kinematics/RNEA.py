import numpy as np

import utility as ram
from robot_ur5 import robot
from forward_kinematics import forward_kinematics


def forward_backward_recursion(q, qdot, qddot, robot, g):
    robot, _ = forward_kinematics(q)
    n = len(robot.body)

    # Inertia tensors
    for body in robot.body.values():
        R = ram.quat2rotation(body.iquat)
        body.I_body = R @ np.diag(body.inertia) @ R.T

    # Forward recursion
    w = np.zeros(3)
    v = np.zeros(3)
    wdot = np.zeros(3)
    vdot = np.zeros(3)

    for i in range(1, n + 1):
        body = robot.body[i]

        axis = body.joint_axis
        com = body.ipos
        p = body.o_local
        R = body.R_local.T

        body.w = R @ w + qdot[i - 1] * axis
        body.v = R @ (v + ram.vec2skew(w) @ p)

        body.wdot = (
            R @ wdot
            + ram.vec2skew(R @ w) @ (qdot[i - 1] * axis)
            + qddot[i - 1] * axis
        )

        body.vdot = R @ (
            vdot
            + ram.vec2skew(wdot) @ p
            + ram.vec2skew(w) @ (ram.vec2skew(w) @ p)
        )

        body.vdotC = (
            body.vdot
            + ram.vec2skew(body.wdot) @ com
            + ram.vec2skew(body.w) @ (ram.vec2skew(body.w) @ com)
        )

        w, v = body.w, body.v
        wdot, vdot = body.wdot, body.vdot

    # Backward recursion
    f = np.zeros(3)
    moment = np.zeros(3)
    R_next = np.eye(3)
    p_next = robot.params.end_eff_pos_local
    gravity = np.array([0.0, 0.0, g])

    for i in range(n, 0, -1):
        body = robot.body[i]

        m = body.mass
        I = body.I_body
        com = body.ipos

        R_world = body.H_global[:3, :3]
        g_body = R_world.T @ gravity

        F = m * body.vdotC
        N = I @ body.wdot + ram.vec2skew(body.w) @ (I @ body.w)

        body.f = R_next @ f + F - m * g_body

        body.n = (
            N
            + R_next @ moment
            + ram.vec2skew(com) @ body.f
            + ram.vec2skew(p_next - com) @ (R_next @ f)
        )

        f = body.f
        moment = body.n
        R_next = body.R_local
        p_next = body.o_local

    # Joint torques
    tau = np.zeros(n)

    for i in range(1, n + 1):
        body = robot.body[i]
        j = int(np.dot(body.joint_axis, [1, 2, 3]))
        tau[i - 1] = body.n[j - 1]

    return tau


def RNEA(q, qdot, flag):
    n = len(qdot)
    g = robot.params.gravity[2]
    qddot = np.zeros(n)

    # Coriolis and centrifugal
    C = forward_backward_recursion(
        q, qdot, qddot, robot, 0
    ).reshape(n, 1)

    # Coriolis and gravity
    CG = forward_backward_recursion(
        q, qdot, qddot, robot, g
    ).reshape(n, 1)

    G = CG - C

    # Mass matrix
    M = np.zeros((n, n))

    if flag == 1:
        for i in range(n):
            qddot[i] = 1.0

            tau = forward_backward_recursion(
                q, qdot, qddot, robot, g
            )

            M[:, i] = tau - CG[:, 0]
            qddot[i] = 0.0

    return C, G, CG, M