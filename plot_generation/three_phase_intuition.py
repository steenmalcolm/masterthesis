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

cutoff = -0.10
surf = ax.plot_surface(
    PHI1,
    PHI2,
    F_masked,
    cmap="Grays_r",
    linewidth=0,
    antialiased=True,
    alpha=0.0,
)
cs = ax.contour(
    PHI1,
    PHI2,
    F_masked,
    levels=np.linspace(np.nanmin(F_masked), cutoff, 100),
    linewidths=0.5,
    cmap="Grays_r",
    zorder=0,
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
    facecolors="orange",
    edgecolors="black",  # or 'black' if you want an outline
    linewidths=3.0,
    linestyle="-",
    alpha=0.5,
    label="3-phase region",
)

ax.add_collection3d(triangle_patch)
ax.scatter(
    triangle_xy[:, 0],
    triangle_xy[:, 1],
    triangle_z,
    color="orange",
    s=300,
    zorder=6,
    edgecolor="black",
)
x = np.linspace(1 / chi, 0.99, 1001)
x0 = 0.18
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
ax.plot(
    edge[0], edge[1], z, color="blue", linewidth=4, label="2-phase tie-line", zorder=3
)
ax.scatter(edge[0], edge[1], z, color="blue", s=300, zorder=4, edgecolor="black")


# Set viewing angle
ax.view_init(elev=10, azim=-102)

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
print(F_masked.min(), F_masked.max())
ax.set_zlim(np.nanmin(F_masked), cutoff)

# Remove ticks
ax.set_xticks([0, 1])
ax.set_yticks([0])
_tri_z = np.round(triangle_z, 8)
_tie_z = np.round(z, 8)
_z_ticks = np.unique(np.concatenate([_tri_z, _tie_z]))
ax.set_zticks(_z_ticks)
ax.set_zticklabels([r"$f_3$", r"$f_2$"])
_tick_colors = ["orange" if v in _tri_z else "blue" for v in _z_ticks]
for tick, color in zip(ax.zaxis.get_major_ticks(), _tick_colors):
    tick.label1.set_color(color)
    tick.tick1line.set_color(color)

# Optional: remove labels too
ax.set_xlabel(r"$\phi_1$", labelpad=10)
ax.set_ylabel(r"$\phi_2$", labelpad=10)
ax.set_zlabel(r"$f$", labelpad=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_ylabel(r"$f(\phi)$", labelpad=10)

# Make background fully transparent
fig.patch.set_alpha(0)
ax.set_facecolor((1, 1, 1, 0))
plt.tight_layout()
from matplotlib.lines import Line2D

# Optionally, further reduce margins:
plt.subplots_adjust(top=0.98, bottom=0.02)
_contour_proxy = Line2D([0], [0], color="gray", label="free-energy surface")
_handles, _labels = ax.get_legend_handles_labels()
plt.legend(handles=[_contour_proxy] + _handles, loc="upper right", frameon=False)
plt.savefig("figures/three_phase_intuition.png", bbox_inches="tight", dpi=300)
