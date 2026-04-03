import matplotlib.pyplot as plt
import mpltern
import numpy as np
import plot_params
from pyphasediagram.diagram import PhaseDiagram
from pyphasediagram.spinodal import Spinodal

np.random.seed(42)

chi_ops = np.linspace(-5, -1, 4)
delta_ops = np.linspace(-5, -3.1, 5)
fig, axes = plt.subplots(
    len(chi_ops),
    len(delta_ops),
    figsize=(4 * len(delta_ops), 4 * len(chi_ops)),
    subplot_kw={"projection": "ternary"},
)

for i, chi in enumerate(chi_ops):
    print(f"{i+1}/{len(chi_ops)}")
    delta_ops = np.linspace(-5 - chi, -2.1 - chi, 5)
    for j, delta in enumerate(delta_ops):
        ax = axes[i, j]

        chis_reduced = np.array([[delta, -delta], [-delta, delta]]) + chi
        chi_01, chi_02 = -1 / 2 * np.diag(chis_reduced)
        chi_12 = chis_reduced[0, 1] - chi_01
        # chis_3x3 = np.array([[0, chi_01, chi_02], [chi_01, 0, chi_12], [chi_02, chi_12, 0]])
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
        if i == 0:
            ax.set_title(f"chi = {chi:.1f}")
fig.savefig("interesting_case_2.png", dpi=300)
# plt.show()
