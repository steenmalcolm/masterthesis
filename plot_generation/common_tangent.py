import matplotlib.pyplot as plt
import numpy as np
import plot_params
from matplotlib.colors import LightSource
from scipy.optimize import fsolve

# Binary-mixture example for the Maxwell construction.
phi_left = 0.22
phi_right = 0.78
slope = -0.18
intercept = 0.20


def common_tangent_line(phi):
    return intercept + slope * phi


def binary_free_energy(phi, stiffness=10.0):
    """Free-energy density with an exact common tangent line at phi_left and phi_right."""
    return (
        common_tangent_line(phi)
        + stiffness * (phi - phi_left) ** 2 * (phi - phi_right) ** 2
    )


def dfdphi(phi, stiffness=10.0):
    # Derivative of the free energy
    return slope + stiffness * (4 * phi - 2 * (phi_left + phi_right)) * (
        phi - phi_left
    ) * (phi - phi_right)


def d2fdphi2(phi, stiffness=10.0):
    # Second derivative (curvature)
    return slope + stiffness * (
        12 * (phi - phi_left) * (phi - phi_right)
        + 2 * (phi - phi_left) ** 2
        + 2 * (phi - phi_right) ** 2
        - 2 * (phi - phi_left) * (phi - phi_right)
    )


phi = np.linspace(0.0, 1.0, 500)
f = binary_free_energy(phi)
line = common_tangent_line(phi)
mu = dfdphi(phi)
mu_tangent = np.full_like(phi, slope)

f_left = binary_free_energy(phi_left)
f_right = binary_free_energy(phi_right)

# Find spinodal points (where d2f/dphi2 = 0)
from scipy.optimize import brentq


def find_spinodal_points():
    # Scan the interval for sign changes in d2fdphi2 and use brentq on each
    scan_phi = np.linspace(0.01, 0.99, 500)
    d2f = d2fdphi2(scan_phi)
    spinodals = []
    for i in range(len(scan_phi) - 1):
        if d2f[i] * d2f[i + 1] < 0:
            a, b = scan_phi[i], scan_phi[i + 1]
            try:
                root = brentq(lambda x: d2fdphi2(x), a, b)
                spinodals.append(root)
            except ValueError:
                pass
    return np.array(spinodals)


spinodal_points = find_spinodal_points()
f_spinodal = binary_free_energy(spinodal_points)
spin1, spin2 = np.sort(spinodal_points)

fig, (ax_curve, ax_mu) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
# Add 'A' and 'B' labels to subplots
ax_curve.text(
    -0.13,
    0.98,
    "A",
    transform=ax_curve.transAxes,
    fontsize=36,
    fontweight="bold",
    va="top",
    ha="left",
)
ax_mu.text(
    -0.13,
    0.98,
    "B",
    transform=ax_mu.transAxes,
    fontsize=36,
    fontweight="bold",
    va="top",
    ha="left",
)

# --- Left: Free energy and tangent ---
ax_curve.plot(
    phi[phi < spin1], f[phi < spin1], color="#1565c0", linewidth=4.5, label=r"$f(\phi)$"
)
ax_curve.plot(
    phi[(phi >= spin1) & (phi <= spin2)],
    f[(phi >= spin1) & (phi <= spin2)],
    color="#1565c0",
    linewidth=4.5,
    linestyle="--",
)
ax_curve.plot(phi[phi > spin2], f[phi > spin2], color="#1565c0", linewidth=4.5)
# Common tangent remains orange dashed
ax_curve.plot(
    phi, line, color="black", linewidth=4.8, linestyle="-", label="common tangent"
)
ax_curve.scatter(
    [phi_left, phi_right],
    [f_left, f_right],
    color="black",
    s=250,
    zorder=3,
    label="coexisting states",
)
# Vertical lines for phi_left and phi_right (bottom to marker)
ax_curve.plot(
    [phi_left, phi_left],
    [ax_curve.get_ylim()[0], f_left],
    color="0.3",
    linestyle=":",
    linewidth=4.2,
)
ax_curve.plot(
    [phi_right, phi_right],
    [ax_curve.get_ylim()[0], f_right],
    color="0.3",
    linestyle=":",
    linewidth=4.2,
)
ax_curve.text(phi_left - 0.03, f_left + 0.045, r"$\phi^{(1)}$", fontsize=25)
ax_curve.text(phi_right - 0.03, f_right + 0.045, r"$\phi^{(2)}$", fontsize=25)
ax_curve.set_xlabel(r"$\phi$")
ax_curve.set_ylabel(r"$f(\phi)$")
ax_curve.set_xlim(0, 1)
ax_curve.set_ylim(0, None)
ax_curve.set_yticks([])
# ax_curve.set_title("Free energy and common tangent")
# ax_curve.legend(loc="upper center")

# --- Right: Chemical potential and equal-area ---
ax_mu.plot(
    phi[phi < spin1],
    mu[phi < spin1],
    color="#1565c0",
    linewidth=4.5,
    label=r"$\mu(\phi) = f'(\phi)$",
)
ax_mu.plot(
    phi[(phi >= spin1) & (phi <= spin2)],
    mu[(phi >= spin1) & (phi <= spin2)],
    color="#1565c0",
    linewidth=4.5,
    linestyle="--",
)
ax_mu.plot(phi[phi > spin2], mu[phi > spin2], color="#1565c0", linewidth=4.5)
ax_mu.plot(
    phi,
    mu_tangent,
    color="black",
    linewidth=4.8,
    linestyle="-",
    label="tangent slope",
)
# Shade equal-area regions between mu and tangent slope
mask = (phi >= phi_left) & (phi <= phi_right)
area_above = np.where(mu > slope, mu, slope)
area_below = np.where(mu < slope, mu, slope)
ax_mu.fill_between(
    phi,
    slope,
    area_above,
    where=(mask & (mu > slope)),
    color="#e76f51",
    alpha=0.5,
    label="Area above slope",
)
ax_mu.fill_between(
    phi,
    slope,
    area_below,
    where=(mask & (mu < slope)),
    color="#457b9d",
    alpha=0.5,
    label="Area below slope",
)
# Vertical lines for phi_left and phi_right (bottom to marker)
ax_mu.plot(
    [phi_left, phi_left],
    [ax_mu.get_ylim()[0], slope],
    color="0.3",
    linestyle=":",
    linewidth=2.2,
)
ax_mu.plot(
    [phi_right, phi_right],
    [ax_mu.get_ylim()[0], slope],
    color="0.3",
    linestyle=":",
    linewidth=2.2,
)
ax_mu.scatter(
    [phi_left, phi_right],
    [slope, slope],
    color="black",
    s=250,
    zorder=3,
    label="coexisting states",
)
ax_mu.text(
    phi_left - 0.03,
    slope + 0.055,
    r"$\phi^{(1)}$",
    fontsize=25,
    ha="center",
    va="bottom",
)
ax_mu.text(
    phi_right - 0.03,
    slope + 0.055,
    r"$\phi^{(2)}$",
    fontsize=25,
    ha="center",
    va="bottom",
)
ax_mu.set_yticks([])
ax_mu.set_ylim(slope - 1, slope + 1)
ax_mu.set_xlim(0.0, 1)

ax_mu.set_xlabel(r"$\phi$")
ax_mu.set_ylabel(r"$\mu(\phi)$")


# Spinodal points
ax_curve.scatter(
    spinodal_points,
    f_spinodal,
    color="orange",
    edgecolor="black",
    s=250,
    zorder=4,
    label="spinodal points",
)

for x, y in zip(spinodal_points, f_spinodal):
    # Free energy plot: vertical line from bottom to spinodal marker
    print(f"ax_curve.get_ylim() = {ax_curve.get_ylim()}")
    ax_curve.plot(
        [x, x],
        [0, y],
        color="orange",
        linestyle="--",
        linewidth=4.2,
        alpha=0.7,
    )
    # Chemical potential plot: vertical line from bottom to spinodal marker
    mu_spinodal = dfdphi(x)
    print(f"ax_mu.get_ylim() = {ax_mu.get_ylim()}")
    ax_mu.plot(
        [x, x],
        [ax_mu.get_ylim()[0], mu_spinodal],
        color="orange",
        linestyle="--",
        linewidth=4.2,
        alpha=0.7,
    )
    ax_mu.scatter(x, mu_spinodal, color="orange", edgecolor="black", s=250, zorder=4)

# ax_mu.set_title("Chemical potential and equal-area (pressure) requirement")
# ax_mu.legend(loc="upper center")

fig.tight_layout()
fig.savefig("figures/common_tangent.png", dpi=300)
