import numpy as np

import utility as ram
from robot_3R import robot
from forward_kinematics import forward_kinematics


def forward_backward_recursion(q, qdot, qddot, robot, g):
    robot, _ = forward_kinematics(q)
    n = len(robot.body)

    # ==================================================
    # Initialize inertia tensors
    # ==================================================
    for i in range(1, n + 1):
        body = robot.body[i]
        I = body.inertia

        body.I_body = np.array([
            [I[0], I[3], I[4]],
            [I[3], I[1], I[5]],
            [I[4], I[5], I[2]],
        ])

    # ==================================================
    # Forward recursion
    # ==================================================
    w_prev = np.zeros(3)
    v_prev = np.zeros(3)
    wdot_prev = np.zeros(3)
    vdot_prev = np.zeros(3)

    for i in range(1, n + 1):
        body = robot.body[i]

        axis = body.joint_axis
        com = body.ipos
        position = body.o_local
        R = body.R_local.T

        # Angular velocity
        body.w = R @ w_prev + qdot[i - 1] * axis

        # Linear velocity
        body.v = R @ (
            v_prev + ram.vec2skew(w_prev) @ position
        )

        # Angular acceleration
        body.wdot = (
            R @ wdot_prev
            + ram.vec2skew(R @ w_prev) @ (qdot[i - 1] * axis)
            + qddot[i - 1] * axis
        )

        # Linear acceleration
        body.vdot = R @ (
            vdot_prev
            + ram.vec2skew(wdot_prev) @ position
            + ram.vec2skew(w_prev) @ (
                ram.vec2skew(w_prev) @ position
            )
        )

        # Center of mass acceleration
        body.vdotC = (
            body.vdot
            + ram.vec2skew(body.wdot) @ com
            + ram.vec2skew(body.w) @ (
                ram.vec2skew(body.w) @ com
            )
        )

        # Update previous link state
        w_prev = body.w
        v_prev = body.v
        wdot_prev = body.wdot
        vdot_prev = body.vdot

    # ==================================================
    # Backward recursion
    # ==================================================
    f_next = np.zeros(3)
    n_next = np.zeros(3)
    R_next = np.eye(3)

    position_next = robot.params.end_eff_pos_local
    gravity = np.array([0.0, g, 0.0])

    for i in range(n, 0, -1):
        body = robot.body[i]

        mass = body.mass
        inertia = body.I_body
        com = body.ipos

        # Gravity in body frame
        R_world = body.H_global[:3, :3]
        g_body = R_world.T @ gravity

        # Inertial force
        F = mass * body.vdotC

        # Inertial moment
        N = (
            inertia @ body.wdot
            + ram.vec2skew(body.w) @ (inertia @ body.w)
        )

        # Force recursion
        body.f = (
            R_next @ f_next
            + F
            - mass * g_body
        )

        # Moment recursion
        body.n = (
            N
            + R_next @ n_next
            + ram.vec2skew(com) @ body.f
            + ram.vec2skew(position_next - com) @ (
                R_next @ f_next
            )
        )

        # Update next link state
        f_next = body.f
        n_next = body.n
        R_next = body.R_local
        position_next = body.o_local

    # ==================================================
    # Joint torques
    # ==================================================
    tau = np.zeros(n)

    for i in range(1, n + 1):
        body = robot.body[i]
        axis = body.joint_axis

        j = int(np.dot(axis, [1, 2, 3]))
        tau[i - 1] = body.n[j - 1]

    return tau


def RNEA(q, qdot):
    nv = len(qdot)
    g = robot.params.gravity[1]

    qddot = np.zeros(nv)

    # ==================================================
    # Coriolis and centrifugal forces
    # ==================================================
    tau_C = forward_backward_recursion(
        q, qdot, qddot, robot, g=0
    )

    C = tau_C.reshape(nv, 1)

    # ==================================================
    # Coriolis + gravity
    # ==================================================
    tau_CG = forward_backward_recursion(
        q, qdot, qddot, robot, g
    )

    CG = tau_CG.reshape(nv, 1)

    # ==================================================
    # Gravity
    # ==================================================
    G = CG - C

    # ==================================================
    # Mass matrix
    # ==================================================
    M = np.zeros((nv, nv))

    for i in range(nv):
        qddot[i] = 1.0

        tau = forward_backward_recursion(
            q, qdot, qddot, robot, g
        )

        M[:, i] = tau - tau_CG

        qddot[i] = 0.0

    return C, G, CG, M