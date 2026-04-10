import matplotlib.pyplot as plt
import mpltern
import numpy as np
import plot_params
from pyphasediagram.diagram import PhaseDiagram
from pyphasediagram.spinodal import Spinodal

np.random.seed(42)

chi_ops = np.linspace(-4.5, -1, 4)
chi = -5
delta_ops = np.array([-0.5, 0, 0.5, 1.1, 10.0, 60.0])
delta_labels = ["-0.5,0.0", "0.5", "1.0", "10", "60"]
subplot_labels = ["A", "B", "C", "D", "E", "F"]
fig, axes = plt.subplots(
    2,
    len(delta_ops) // 2,
    figsize=(12, 8),
    subplot_kw={"projection": "ternary"},
)

for j, delta in enumerate(delta_ops):
    print(f"{j+1}/{len(delta_ops)}: {delta:.2f}")
    ax = axes[j // 3, j % 3]

    chis_reduced = np.array([[delta, -delta], [-delta, delta]]) + chi
    chi_01, chi_02 = -1 / 2 * np.diag(chis_reduced)
    chi_12 = chis_reduced[0, 1] - chis_reduced[0, 0]
    # chis_3x3 = np.array([[0, chi_01, chi_02], [chi_01, 0, chi_12], [chi_02, chi_12, 0]])
    print(chi_01, chi_02, chi_12)
    print(chis_reduced)
    diagram = PhaseDiagram(chis_reduced)
    diagram.build()
    b = diagram.binodal
    s = diagram.spinodal

    for poly in b.three_phase_polygons:
        phi1, phi2 = poly.exterior.xy
        phi1, phi2 = np.array(phi1), np.array(phi2)
        phi0 = 1 - phi1 - phi2
        ax.fill(phi0, phi1, phi2, color="orange", zorder=5)
    for s_id, section in enumerate(b.binodal_sections):
        for phis in section.phis:
            phi1, phi2 = phis
            phi0 = 1 - phi1 - phi2
            ax.plot(phi0, phi1, phi2, color="blue", zorder=2, linewidth=2)

    for phi1_sp, phi2_sp in zip(*s.get_spinodal_coords()):
        phi0_sp = 1 - phi1_sp - phi2_sp
        ax.plot(
            phi0_sp,
            phi1_sp,
            phi2_sp,
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=1,
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
            s=150,
            zorder=6,
            label="Critical point",
        )
    num_tie_lines = 15
    for s_id, section in enumerate(b.binodal_sections):
        phis_sel = np.arange(
            0,
            section.phis.shape[-1],
            max(1, section.phis.shape[-1] // num_tie_lines),
        )
        for idx in phis_sel:
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
            )
    ax.text(
        0.00,
        1.0,
        subplot_labels[j],
        transform=ax.transAxes,
        fontsize=36,
        fontweight="bold",
        va="top",
    )

    ax.taxis.set_ticks([0, 1])
    ax.laxis.set_ticks([0, 1])
    ax.raxis.set_ticks([0, 1])
    ax.taxis.set_ticks([])
    ax.laxis.set_ticks([])
    ax.raxis.set_ticks([])

    ax.set_tlabel(r"$\phi_0$", fontsize=18)
    ax.set_llabel(r"$\phi_1$", fontsize=18)
    ax.set_rlabel(r"$\phi_2$", fontsize=18)

fig.tight_layout()
fig.savefig("figures/interesting_case.png", dpi=300)
# plt.show()
