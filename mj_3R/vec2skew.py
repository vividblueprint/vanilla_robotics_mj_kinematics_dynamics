import sympy as sy

def vec2skew(v):
    # Ensure v is a 3-element vector
    assert len(v) == 3, "Input must be a 3-element vector"
    # Create the skew-symmetric matrix
    A = sy.Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return A

# Example usage
# v = sy.symbols('v1 v2 v3')  # Define symbolic vector elements
# skew_matrix = vec2skew(v)
# sy.pprint(skew_matrix)
