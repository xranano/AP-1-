import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


np.random.seed(42)


#vector norms
def vector_norm_2(v):
    return np.sqrt(np.sum(v ** 2))
def vector_norm_inf(v):
    return np.max(np.abs(v))



#induced matrix norms
def matrix_norm_2(A):
    return np.linalg.norm(A, 2)
def matrix_norm_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))


#generate random vectors and matrices
v1 = np.random.randn(4)
v2 = np.random.randn(4)

A1 = v1.reshape(2, 2)
A2 = v2.reshape(2, 2)


print("generated vectors")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print("\ngenerated matrices")
print(f"A1 (reshaped from v1):\n{A1}")
print(f"\nA2 (reshaped from v2):\n{A2}")


#distances
v_diff = v1 - v2
dist_v_2 = vector_norm_2(v_diff)
dist_v_inf = vector_norm_inf(v_diff)

print("\nvector distances")
print(f"||v1 - v2||_2= {dist_v_2:.6f}")
print(f"||v1 - v2||_∞= {dist_v_inf:.6f}")


A_diff = A1 - A2
dist_A_2 = matrix_norm_2(A_diff)
dist_A_inf = matrix_norm_inf(A_diff)

print("matrix distances")
print(f"||A1 - A2||_2= {dist_A_2:.6f}")
print(f"||A1 - A2||_∞= {dist_A_inf:.6f}")




#visualization
def draw_unit_ball_2d(ax, xr_2d, norm_type='2', color='blue', label=''):
    theta = np.linspace(0, 2 * np.pi, 1000)

    if norm_type == '2':
        # Circle for L² norm
        x = xr_2d[0] + np.cos(theta)
        y = xr_2d[1] + np.sin(theta)
    else:  # infinity norm
        # Square for L∞ norm
        x = xr_2d[0] + np.array([1, 1, -1, -1, 1])
        y = xr_2d[1] + np.array([1, -1, -1, 1, 1])

    ax.plot(x, y, color=color, linewidth=2.5, label=label)
    ax.fill(x, y, color=color, alpha=0.2)
    ax.plot(xr_2d[0], xr_2d[1], 'o', color=color, markersize=10,
            markeredgecolor='black', markeredgewidth=1.5)


# Create visualization using only first 2 components of each vector
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Unit Balls in 2D: ||x - xᵣ|| ≤ 1 (First Two Components)',
             fontsize=16, fontweight='bold')

# Extract 2D coordinates (first two components)
v1_2d = v1[:2]
v2_2d = v2[:2]

# Left plot: L² norm
ax = axes[0]
draw_unit_ball_2d(ax, v1_2d, '2', '#3498db', f'v1 = [{v1_2d[0]:.2f}, {v1_2d[1]:.2f}]')
draw_unit_ball_2d(ax, v2_2d, '2', '#2ecc71', f'v2 = [{v2_2d[0]:.2f}, {v2_2d[1]:.2f}]')
ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('L² Norm (Euclidean) - Circles', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.legend(loc='best')
ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

# Right plot: L∞ norm
ax = axes[1]
draw_unit_ball_2d(ax, v1_2d, 'inf', '#e74c3c', f'v1 = [{v1_2d[0]:.2f}, {v1_2d[1]:.2f}]')
draw_unit_ball_2d(ax, v2_2d, 'inf', '#f39c12', f'v2 = [{v2_2d[0]:.2f}, {v2_2d[1]:.2f}]')
ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('L∞ Norm (Maximum) - Squares', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.legend(loc='best')
ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.show()
