import numpy as np
from scipy.spatial.transform import Rotation as R


# 1. Rotation matrix around X, Y, or Z
def rotation(angle, axis):
    axes = ["x", "y", "z"]
    return R.from_euler(axes[axis], angle).as_matrix()


# 2. Quaternion → Rotation matrix
def quat2rotation(q):
    return R.from_quat(q, scalar_first=True).as_matrix()


# 3. Rotation matrix → Quaternion
def rotation2quat(matrix):
    return R.from_matrix(matrix).as_quat(scalar_first=True)


# 4. Euler angles → Rotation matrix
def euler2rotation(euler):
    return R.from_euler("XYZ", euler).as_matrix()


# 5. Rotation matrix → Euler angles
def rotation2euler(matrix):
    return R.from_matrix(matrix).as_euler("XYZ")


# 6. Euler angles → Quaternion
def euler2quat(euler):
    return R.from_euler("XYZ", euler).as_quat(scalar_first=True)


# 7. Quaternion → Euler angles
def quat2euler(q):
    return R.from_quat(q, scalar_first=True).as_euler("XYZ")


# 8. Quaternion → Axis-angle
def quat2axisangle(q):
    rotvec = R.from_quat(q, scalar_first=True).as_rotvec()
    angle = np.linalg.norm(rotvec)
    axis = rotvec / angle if angle > 1e-8 else np.array([1, 0, 0])
    return axis, angle


# 9. Quaternion conjugate
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


# 10. Quaternion multiplication
def quat_product(q, p):
    result = (
        R.from_quat(q, scalar_first=True)
        * R.from_quat(p, scalar_first=True)
    )
    return result.as_quat(scalar_first=True)


# 11. Normalize quaternion
def quat_normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# 12. Vector → Skew-symmetric matrix
def vec2skew(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


# 13. Angular velocity in body frame
def quat2angvelBody(q, qdot):
    omega = 2 * quat_product(quat_conjugate(q), qdot)
    return omega[1:]


# 14. Angular velocity in world frame
def quat2angvelWorld(q, qdot):
    omega = 2 * quat_product(qdot, quat_conjugate(q))
    return omega[1:]


# 15. Angular acceleration in body frame
def quat2angaccBody(q, qdot, qddot):
    alpha = 2 * (
        quat_product(quat_conjugate(q), qddot)
        + quat_product(quat_conjugate(qdot), qdot)
    )
    return alpha[1:]


# 16. Angular acceleration in world frame
def quat2angaccWorld(q, qdot, qddot):
    alpha = 2 * (
        quat_product(qddot, quat_conjugate(q))
        + quat_product(qdot, quat_conjugate(qdot))
    )
    return alpha[1:]


# 17. Print matrix
def print_matrix(matrix, decimals=4):
    print(np.round(matrix, decimals))