import sympy as sy

def rotation(angle, axis):
    if axis not in [1, 2, 3]:
        raise ValueError("axis can only be 1, 2 or 3")
    
    if axis == 1:
        R = sy.Matrix([
            [1,     0,          0],
            [0,     sy.cos(angle),   -sy.sin(angle)],
            [0,     sy.sin(angle),   sy.cos(angle)]
        ])
    elif axis == 2:
        R = sy.Matrix([
            [sy.cos(angle),     0,      sy.sin(angle)],
            [0,                 1,      0],
            [-sy.sin(angle),    0,      sy.cos(angle)]
        ])
    elif axis == 3:
        R = sy.Matrix([
            [sy.cos(angle),     -sy.sin(angle),     0],
            [sy.sin(angle),     sy.cos(angle),      0],
            [0,                 0,                  1]
        ])

    return R
