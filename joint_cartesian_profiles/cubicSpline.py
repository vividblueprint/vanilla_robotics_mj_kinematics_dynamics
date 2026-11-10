import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Data points
x = np.array([1, 2, 3, 4])
y = np.array([3, 6, 5, 8])


# Generate cubic spline interpolation (This is like spline)
#try these options and see how the 1st and 2nd derivative changes
#cs = CubicSpline(x,y,bc_type='not-a-knot') #third derivative at the second and second to last knot point are continouous
#cs = CubicSpline(x, y,bc_type='clamped') #first derivative at the ends are zero
cs = CubicSpline(x, y,bc_type='natural') #second derivative at the ends are zero

# Access breakpoints
breakpoints = cs.x
print("Breakpoints (knots):", breakpoints)

# Access coefficients
coefficients = cs.c
print("Coefficients (per interval):")
print(coefficients)

# Evaluate the spline at new points
x_new = np.linspace(x[0], x[-1], 100)
y_new = cs(x_new)

# First derivative of the spline
y_prime = cs(x_new, 1)

# Second derivative of the spline
y_double_prime = cs(x_new, 2)


# Plot the result
plt.plot(x, y, 'o', label='data points')  # Original data points
plt.plot(x_new, y_new, label='cubic spline')  # Interpolated spline
plt.plot(x_new, y_prime, label="1st derivative")    # First derivative
plt.plot(x_new, y_double_prime, label="2nd derivative")  # Second derivative
plt.legend()
#plt.show();
plt.show(block=False)
plt.pause(10)
plt.close()
