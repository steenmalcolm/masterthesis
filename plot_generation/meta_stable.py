import matplotlib.pyplot as plt
import numpy as np
import plot_params
from scipy.optimize import brentq

chi_01 = 3


def f(phi, phi0):
    return np.log(phi / phi0) + chi_01 * (phi0 - phi)


fig, ax = plt.subplots(1, 1, figsize=(7, 5.2), constrained_layout=True)
# Set background color to gray
# fig.patch.set_facecolor("#e5e5e5")
ax.set_facecolor("#e5e5e5")

phi0s = np.linspace(0, 1 / chi_01, 1001)
from scipy.optimize import root_scalar

phi1s = np.zeros_like(phi0s)
for i, phi0 in enumerate(phi0s):
    try:
        sol = root_scalar(
            f, args=(phi0,), bracket=[1 / chi_01, 1 - 1e-6], method="bisect"
        )
        if sol.converged:
            phi1s[i] = sol.root
    except ValueError:
        pass

mask = phi1s > 0
phi0s = phi0s[mask]
phi1s = phi1s[mask]
phi2s = 1 - phi0s - phi1s
ax.plot(phi2s, phi1s, color="black", linewidth=4.0, label="binodal() curve")
ax.plot(phi2s, phi0s, color="black", linewidth=4.0, label="spinodal curve")
ax.scatter(
    1 - 2 / chi_01,
    1 / chi_01,
    color="gold",
    marker="*",
    s=200,
    edgecolor="black",
    label="critical point",
    zorder=10,
)
phi1_spinodal = (phi1s - 1 / chi_01) / 2 + 1 / chi_01
phi0_spinodal = (phi0s - 1 / chi_01) / 2 + 1 / chi_01
ax.plot(
    phi2s,
    phi1_spinodal,
    color="black",
    linestyle="--",
    linewidth=4.0,
    label="spinodal points",
)
ax.plot(
    phi2s,
    phi0_spinodal,
    color="black",
    linestyle="--",
    linewidth=4.0,
    label="spinodal points",
)

## Fill unstable region (between spinodal lines)
ax.fill_between(
    phi2s,
    phi0_spinodal,
    phi1_spinodal,
    where=phi1_spinodal > phi0_spinodal,
    color="blue",
    alpha=0.5,
    label="unstable region",
)

## Fill metastable region (between binodal and spinodal)
ax.fill_between(
    phi2s,
    phi1_spinodal,
    phi1s,
    where=phi1s > phi1_spinodal,
    color="#3ec3e8",
    alpha=0.7,
    label="metastable region upper",
)
ax.fill_between(
    phi2s,
    phi0s,
    phi0_spinodal,
    where=phi0_spinodal > phi0s,
    color="#3ec3e8",
    alpha=0.7,
    label="metastable region lower",
)

# Add region labels
ax.text(
    0.3,
    0.38,
    "unstable",
    transform=ax.transAxes,
    fontsize=30,
    color="#000000",
    ha="center",
    va="center",
    alpha=0.8,
    fontweight="bold",
)
ax.text(
    0.3,
    0.65,
    "metastable",
    transform=ax.transAxes,
    fontsize=30,
    color="#000000",
    ha="center",
    va="center",
    alpha=0.8,
    fontweight="bold",
)
ax.text(
    0.3,
    0.155,
    "metastable",
    transform=ax.transAxes,
    fontsize=30,
    color="#000000",
    ha="center",
    va="center",
    alpha=0.8,
    fontweight="bold",
)
ax.text(
    0.3,
    0.88,
    "stable",
    transform=ax.transAxes,
    fontsize=30,
    color="#000000",
    ha="center",
    va="center",
    alpha=0.8,
    fontweight="bold",
)
ax.set_xticks([0, 1 - 2 / chi_01])
ax.set_xticklabels([r"$0$", r"$\phi_{2,c}$"])
ax.set_yticks([0, 1])
ax.set_xlabel(r"$\phi_2$")
ax.set_ylabel(r"$\phi_1$")
ax.set_xlim(0, 1 / chi_01 + 0.02)
ax.set_ylim(0, 1)
fig.savefig("figures/metastable.png", dpi=300)
