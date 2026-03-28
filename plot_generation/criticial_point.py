import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

plt.rcParams.update(
    {
        "figure.figsize": (15, 5.2),
        "figure.dpi": 140,
        "axes.grid": False,
        "font.size": 22,
        "axes.labelsize": 28,
        "axes.titlesize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "axes.linewidth": 3.5,
        "xtick.major.width": 3.5,
        "ytick.major.width": 3.5,
        "xtick.minor.width": 2.2,
        "ytick.minor.width": 2.2,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.minor.size": 7,
        "ytick.minor.size": 7,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "legend.title_fontsize": 24,
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
            "sans-serif",
        ],
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)

chi_01 = 3
chi_2 = 1.5
phi2_c = 1 - 2 / chi_01

panel_labels = ["A", "B", "C"]
phi2_labels = [r"$\phi_2 > \phi_2^c$", r"$\phi_2 = \phi_2^c$", r"$\phi_2 < \phi_2^c$"]
phi2_values = [phi2_c + 0.1, phi2_c, phi2_c - 0.1]


def flory_huggins_free_energy(phi1, phi2):
    phi0 = 1 - phi1 - phi2
    return (
        phi0 * np.log(phi0)
        + phi1 * np.log(phi1)
        + phi2 * np.log(phi2)
        + chi_01 * phi1 * phi0
        + chi_2 * phi2 * (phi0 + phi1)
    )


def d2fdphi12(phi1, phi2):
    """Second derivative of f with respect to phi1 at fixed phi2."""
    phi0 = 1 - phi1 - phi2
    return 1 / phi1 + 1 / phi0 - 2 * chi_01


def find_spinodal_points(phi1_grid, phi2):
    """Find roots of d2f/dphi1^2 = 0 by scanning for sign changes."""
    d2f = d2fdphi12(phi1_grid, phi2)
    roots = []

    for i in range(len(phi1_grid) - 1):
        if d2f[i] == 0:
            roots.append(phi1_grid[i])
        elif d2f[i] * d2f[i + 1] < 0:
            a, b = phi1_grid[i], phi1_grid[i + 1]
            try:
                root = brentq(lambda x: d2fdphi12(x, phi2), a, b)
                roots.append(root)
            except ValueError:
                pass

    # remove duplicates that can occur if a root is found more than once
    if len(roots) == 0:
        return np.array([])

    roots = np.array(roots)
    roots = np.unique(np.round(roots, decimals=10))
    return roots


fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)

for ax, phi2, phi2_l, label in zip(axes, phi2_values, phi2_labels, panel_labels):
    phi1 = np.linspace(1e-4, 1 - phi2 - 1e-4, 1200)
    f = flory_huggins_free_energy(phi1, phi2)

    # Shift each curve vertically so the minimum is near zero
    f_shifted = f - np.min(f) + 0.01

    ax.plot(phi1, f_shifted, color="#1565c0", linewidth=3.0)

    # Panel label
    ax.text(
        -0.13,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=36,
        fontweight="bold",
        va="top",
        ha="left",
    )
    # Phi2 label
    ax.text(
        0.5,
        0.9,
        phi2_l,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="center",
    )

    # Find and plot d2f/dphi1^2 = 0 points
    spinodal_points = find_spinodal_points(phi1, phi2)

    if len(spinodal_points) > 0:
        f_spinodal = flory_huggins_free_energy(spinodal_points, phi2) - np.min(f) + 0.01

        ax.scatter(
            spinodal_points,
            f_spinodal,
            color="orange",
            edgecolor="black",
            s=80,
            zorder=4,
        )

        for x, y in zip(spinodal_points, f_spinodal):
            ax.plot(
                [x, x],
                [0, y],
                color="orange",
                linestyle="--",
                linewidth=2.2,
                alpha=0.8,
            )
    if phi2 == phi2_c:
        phi1_c = 1 / chi_01
        f_spinodal = flory_huggins_free_energy(phi1_c, phi2_c) - np.min(f) + 0.01
        ax.scatter(
            phi1_c,
            f_spinodal,
            color="orange",
            edgecolor="black",
            s=80,
            zorder=4,
        )

        ax.plot(
            [phi1_c, phi1_c],
            [0, f_spinodal],
            color="orange",
            linestyle="--",
            linewidth=2.2,
            alpha=0.8,
        )
    if phi2 < phi2_c:
        # Show text of binodal points
        min_idx = np.argmin(f[: len(f) // 2])
        print(min_idx)
        ax.text(
            phi1[min_idx] + 0.002,
            f_shifted[min_idx] + 0.01,
            r"$\phi_1^{(1)}$",
            fontsize=18,
        )
        ax.scatter(
            phi1[min_idx],
            f_shifted[min_idx],
            color="black",
            s=70,
            zorder=3,
        )
        min_idx = np.argmin(f[len(f) // 2 :]) + len(f) // 2
        print(min_idx)
        ax.scatter(
            phi1[min_idx],
            f_shifted[min_idx],
            color="black",
            s=70,
            zorder=3,
        )
        ax.text(
            phi1[min_idx] + 0.002,
            f_shifted[min_idx] + 0.01,
            r"$\phi_1^{(2)}$",
            fontsize=18,
        )

    ax.set_xlabel(r"$\phi_1$")
    ax.set_xlim(phi1.min(), phi1.max())
    ax.set_yticks([0.01])
    ax.set_yticklabels([None])
    ax.set_xticks([phi1.min(), 1 / chi_01, phi1.max()])
    # ax.set_xticklabels([r"$\phi_1^c$"])
    ax.set_xticklabels([""] * 3)
    ax.set_ylim(0, None)

axes[0].set_yticklabels([r"$f_\text{min}$"])
axes[0].set_ylabel(r"$f(\phi_1,\phi_2)$")

fig.savefig("figures/critical_point_schematic.png", dpi=300)
