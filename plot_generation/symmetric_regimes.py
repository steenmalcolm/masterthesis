import matplotlib.pyplot as plt
import mpltern
import numpy as np
import plot_params
from pyphasediagram.diagram import PhaseDiagram

# Create 6 subplots with ternary projections
chi_values = [2.4, 2.65, 2.67, 2.7, 2.746, 3.0]
subplot_labels = ["A", "B", "C", "D", "E", "F"]
fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={"projection": "ternary"})

for chi_idx, chi in enumerate([2.4, 2.65, 2.67, 2.7, 2.746, 3.0]):
    print(f"Processing chi={chi:.3f}...")
    chis = np.array([[0, chi, chi], [chi, 0, chi], [chi, chi, 0]])
    ax = axes[chi_idx // 3, chi_idx % 3]
    ax.text(
        0.00,
        1.0,
        subplot_labels[chi_idx],
        transform=ax.transAxes,
        fontsize=36,
        fontweight="bold",
        va="top",
    )

    pd = PhaseDiagram(chis)
    pd.build()
    b = pd.binodal
    s = pd.spinodal
    num_tie_lines = 1

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

        # pick a subset of tie lines
        phis_sel = np.random.choice(
            range(section.phis.shape[-1]),
            size=min(num_tie_lines, len(section.phis)),
            replace=False,
        )

        # plot each tie line separately
        for i in phis_sel:
            phi1_pair = section.phis[:, 0, i]
            phi2_pair = section.phis[:, 1, i]
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
    for cp in pd.spinodal.critical_points:
        phi1, phi2 = cp.phi1, cp.phi2
        phi0 = 1 - phi1 - phi2
        ax.scatter(
            phi0,
            phi1,
            phi2,
            marker="*",
            color="gold",
            edgecolor="black",
            s=100,
            zorder=10,
        )

    # only show 0 and 1 on each ternary axis
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


fig.savefig(f"figures/symmetric_chis.png", dpi=300)
plt.close()
