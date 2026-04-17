import matplotlib.pyplot as plt
import mpltern
import numpy as np
import plot_params
from pyphasediagram.diagram import PhaseDiagram
from pyphasediagram.spinodal import Spinodal

np.random.seed(4)


chi_01, chi_02, chi_12 = 2.5840, 2.8667, 2.70

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 16),
    subplot_kw={"projection": "ternary"},
)
ax = axes[0]
ax.grid(True)
ax.taxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.laxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.raxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.scatter(
    0.2, 0.5, 0.3, marker="*", color="red", s=1200, label="Initial point", zorder=10
)
ax.scatter(
    0.9, 0.1, 0.0, marker="*", color="green", s=1200, label="Initial point", zorder=10
)

for ax, l in zip(axes, ["A", "B"]):
    ax.set_tlabel(r"$\phi_0$", fontsize=26)
    ax.set_llabel(r"$\phi_1$", fontsize=26)
    ax.set_rlabel(r"$\phi_2$", fontsize=26)
    ax.text(-0.15, 1.2, l, transform=ax.transAxes, fontsize=52, fontweight="bold")

chis = np.array([[0, chi_01, chi_02], [chi_01, 0, chi_12], [chi_02, chi_12, 0]])
diagram = PhaseDiagram(chis)
diagram.build()
b = diagram.binodal
s = diagram.spinodal

ax = axes[1]
for i, poly in enumerate(b.three_phase_polygons):
    phi1, phi2 = poly.exterior.xy
    phi1, phi2 = np.array(phi1), np.array(phi2)
    phi0 = 1 - phi1 - phi2
    ax.fill(
        phi0, phi1, phi2, color="orange", zorder=5, label="3-phase" if i == 0 else None
    )
for s_id, section in enumerate(b.binodal_sections):
    for i, phis in enumerate(section.phis):
        phi1, phi2 = phis
        phi0 = 1 - phi1 - phi2
        ax.plot(
            phi0,
            phi1,
            phi2,
            color="blue",
            zorder=2,
            linewidth=4.5,
            label="binodal" if s_id == 0 and i == 0 else None,
        )

for i, (phi1_sp, phi2_sp) in enumerate(zip(*s.get_spinodal_coords())):
    phi0_sp = 1 - phi1_sp - phi2_sp
    ax.plot(
        phi0_sp,
        phi1_sp,
        phi2_sp,
        color="black",
        linewidth=4.5,
        linestyle="--",
        zorder=1,
        label="spinodal" if i == 0 else None,
    )
for cp in diagram.spinodal.critical_points:
    phi1, phi2 = cp.phi1, cp.phi2
    phi0 = 1 - phi1 - phi2
    ax.scatter(
        phi0,
        phi1,
        phi2,
        marker="*",
        color="gold",
        edgecolors="k",
        s=500,
        zorder=6,
        label="Critical point" if cp == diagram.spinodal.critical_points[0] else None,
    )
num_tie_lines = 15
for s_id, section in enumerate(b.binodal_sections):
    phis_sel = np.arange(
        0,
        section.phis.shape[-1],
        max(1, section.phis.shape[-1] // num_tie_lines),
    )
    for i, idx in enumerate(phis_sel):
        phi1_pair = section.phis[:, 0, idx]
        phi2_pair = section.phis[:, 1, idx]
        phi0_pair = 1 - phi1_pair - phi2_pair
        ax.plot(
            phi0_pair,
            phi1_pair,
            phi2_pair,
            linestyle="--",
            linewidth=0.8,
            color="gray",
            zorder=0,
            label="tie line" if s_id + i == 0 else None,
        )
    # ax.taxis.set_ticks([0, 1])
    # ax.laxis.set_ticks([0, 1])
    # ax.raxis.set_ticks([0, 1])
    # ax.taxis.set_ticks([])
    # ax.laxis.set_ticks([])
    # ax.raxis.set_ticks([])

    # ax.set_tlabel(r"$\phi_0$", fontsize=18)
    # ax.set_llabel(r"$\phi_1$", fontsize=18)
    # ax.set_rlabel(r"$\phi_2$", fontsize=18)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.74, 1.25),
    bbox_transform=ax.transAxes,
    frameon=False,
)
fig.tight_layout()
fig.savefig(f"figures/algo_example_diagram.png", dpi=300)
