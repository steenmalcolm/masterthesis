import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plot_params

# --- Parameters ---
N1, N2, N3 = 1.0, 1.0, 1.0

# These interaction parameters are chosen to give a strongly non-convex landscape.
# You may want to tune them for the visual style you prefer.
chi = 2.9
chi12 = chi
chi13 = chi
chi23 = chi

# Small cutoff to avoid log(0)
eps = 1e-9


def fh_free_energy(phi1, phi2, N1=1.0, N2=1.0, N3=1.0, chi12=3.2, chi13=3.0, chi23=2.8):
    """Flory-Huggins free energy density for an incompressible ternary mixture."""
    phi3 = 1.0 - phi1 - phi2

    # Valid region: phi1 >= 0, phi2 >= 0, phi3 >= 0
    valid = (phi1 >= 0.0) & (phi2 >= 0.0) & (phi3 >= 0.0)

    f = np.full_like(phi1, np.nan, dtype=float)

    p1 = np.clip(phi1[valid], eps, 1.0)
    p2 = np.clip(phi2[valid], eps, 1.0)
    p3 = np.clip(phi3[valid], eps, 1.0)

    entropic = (p1 / N1) * np.log(p1) + (p2 / N2) * np.log(p2) + (p3 / N3) * np.log(p3)

    enthalpic = chi12 * p1 * p2 + chi13 * p1 * p3 + chi23 * p2 * p3

    f[valid] = entropic + enthalpic
    return f


# --- Grid in composition space ---
n = 300
phi1_vals = np.linspace(0.0, 1.0, n)
phi2_vals = np.linspace(0.0, 1.0, n)
PHI1, PHI2 = np.meshgrid(phi1_vals, phi2_vals)

F = fh_free_energy(
    PHI1, PHI2, N1=N1, N2=N2, N3=N3, chi12=chi12, chi13=chi13, chi23=chi23
)

PHI3 = 1.0 - PHI1 - PHI2
v = 0.09
valid = (PHI1 >= v) & (PHI2 >= v) & (PHI3 >= v)
v = 0.116
# valid = (PHI1 >= 0) & (PHI2 >= 0) & (PHI3 >= 0)
F_masked = np.ma.masked_where(~valid, F)

# --- Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    PHI1,
    PHI2,
    F_masked,
    cmap=mpl.cm.cividis,
    linewidth=0,
    antialiased=True,
    alpha=0.9,
    label="Free energy surface",
)


# Optional: draw the boundary of the allowed composition triangle
# Edges: (phi1, phi2) = (0,0)->(1,0)->(0,1)->(0,0)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Corner vertices of the allowed composition triangle
triangle_xy = np.array([[v, v], [v, 1 - 2 * v], [1 - 2 * v, v]])

triangle_z = fh_free_energy(
    triangle_xy[:, 0],
    triangle_xy[:, 1],
    N1=N1,
    N2=N2,
    N3=N3,
    chi12=chi12,
    chi13=chi13,
    chi23=chi23,
)

# 3D vertices of the triangle
verts3d = [list(zip(triangle_xy[:, 0], triangle_xy[:, 1], triangle_z))]

# Add transparent triangular patch
triangle_patch = Poly3DCollection(
    verts3d,
    facecolors="red",
    edgecolors="black",  # or 'black' if you want an outline
    linewidths=0.5,
    linestyle="--",
    alpha=0.5,
    label="Three-phase construction",
)

ax.add_collection3d(triangle_patch)
x = np.linspace(1 / chi, 0.99, 1001)
for i, x0 in enumerate(np.linspace(v + 0.02, 1 / chi - 0.02, 17)):
    f = np.log(x / x0) - chi * (x - x0)
    root_idx = np.where(np.diff(np.sign(f)))[0][0]
    x1 = x[root_idx]
    y = 1 - x0 - x1

    edge = np.array([[x0, x1], [y, y]])
    # Plot an edge line
    z = fh_free_energy(
        edge[0],
        edge[1],
        N1=N1,
        N2=N2,
        N3=N3,
        chi12=chi12,
        chi13=chi13,
        chi23=chi23,
    )
    if i == 0:
        ax.plot(edge[0], edge[1], z, color="black", linewidth=1, label="Tie lines")
    else:
        ax.plot(edge[0], edge[1], z, color="black", linewidth=1)


# Set viewing angle
ax.view_init(elev=10, azim=-96)

# Remove panes (the gray background planes)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove pane edges
ax.xaxis.pane.set_edgecolor("w")
ax.yaxis.pane.set_edgecolor("w")
ax.zaxis.pane.set_edgecolor("w")

# Remove grid
ax.grid(False)

# Remove axis lines
# ax.xaxis.line.set_color((1, 1, 1, 0))
# ax.yaxis.line.set_color((1, 1, 1, 0))
# ax.zaxis.line.set_color((1, 1, 1, 0))

# Remove ticks
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_zticks([])

# Optional: remove labels too
ax.set_xlabel(r"$\phi_1$", labelpad=10)
ax.set_ylabel(r"$\phi_2$", labelpad=10)
# ax.set_ylabel(r"$f(\phi)$", labelpad=10)

# Make background fully transparent
fig.patch.set_alpha(0)
ax.set_facecolor((1, 1, 1, 0))
plt.tight_layout()
# Optionally, further reduce margins:
plt.subplots_adjust(top=0.98, bottom=0.02)
# plt.legend(loc="upper right", frameon=False)
plt.savefig("figures/three_phase_intuition.png", bbox_inches="tight", dpi=300)
plt.show()
