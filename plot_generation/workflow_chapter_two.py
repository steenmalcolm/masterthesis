import os

import matplotlib.pyplot as plt
import numpy as np
import plot_params
from matplotlib.patches import ConnectionPatch
from pyphasediagram.diagram import PhaseDiagram

num_tie_lines = 30
chis = np.array([[0, 1.5, 2.5], [1.5, 0, 1.5], [2.5, 1.5, 0]])
diagram = PhaseDiagram(chis)
diagram.build()
bin_phis = diagram.binodal.binodal_sections[0].phis
bin_phis = bin_phis[:, :, 10 :: max(1, bin_phis.shape[-1] // num_tie_lines)]
fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
ax.scatter(
    bin_phis[0, 0],
    bin_phis[0, 1],
    s=250,
    color="blue",
    edgecolor="black",
    zorder=2,
)
ax.scatter(
    bin_phis[1, 0],
    bin_phis[1, 1],
    s=250,
    color="blue",
    edgecolor="black",
    zorder=2,
)
ax.plot(
    bin_phis[:, 0, :], bin_phis[:, 1, :], color="black", linestyle="--", linewidth=0.5
)
ax.set_xlim(0, 1 - 2 / 2.5 + 1e-2)
ax.set_xlabel(r"$\phi_2$")
ax.set_ylabel(r"$\phi_1$")
ax.set_xticks([0, 1 - 2 / 2.5])
ax.set_xticklabels([r"$0$", r"$\phi_{2,c}$"])
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$0$", r"$1$"])

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L = 12.0
x0 = -0.1  # nonzero interface location
w = 1.0
s = 0.38
inset_locations = [(0.03, 0.28, s, s), (0.58, 0.565, s, s)]
mcs = [(0.65, 0.3), (0.5, 0.3)]
data_idx = [0, 10]
main_ax = ax
for mc, loc, d_idx in zip(mcs, inset_locations, data_idx):
    phi1_left, phi1_right = mc[0], mc[1]
    phi2_left, phi2_right = mc[1], mc[0]
    phi0_left, phi0_right = 1 - phi1_left - phi2_left, 1 - phi1_right - phi2_right

    x = np.linspace(-L / 2, L / 2, 1200)

    # ------------------------------------------------------------
    # Diffuse profiles
    # Species 1 decreases, species 2 increases across the interface
    # ------------------------------------------------------------
    phi1 = 0.5 * (phi1_left + phi1_right) - 0.5 * (phi1_left - phi1_right) * np.tanh(
        (x - x0) / w
    )
    phi1 = np.concatenate((phi1, phi1[::-1]))
    phi2 = 0.5 * (phi2_left + phi2_right) + 0.5 * (phi2_right - phi2_left) * np.tanh(
        (x + x0) / w
    )
    phi2 = np.concatenate((phi2, phi2[::-1]))
    phi0 = 1 - phi1 - phi2

    # Hard-interface counterparts
    phi1_hard = np.where(x < 0, phi1_left, phi1_right)
    phi2_hard = np.where(x < 0, phi2_left, phi2_right)
    phi1_hard = np.concatenate((phi1_hard, phi1_hard[::-1]))
    phi2_hard = np.concatenate((phi2_hard, phi2_hard[::-1]))
    phi0_hard = 1 - phi1_hard - phi2_hard
    x = np.linspace(-L / 2, L / 2, 2400)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    print(f"Creating inset at location {loc} for MC={mc}")
    ax = main_ax.inset_axes(loc)
    # Species 1 (blue)
    (line1,) = ax.plot(x, phi1, linewidth=6.0, label=r"$\phi_1(x)$")
    # ax.plot(x, phi1_hard, linestyle="--", linewidth=2.5, color=line1.get_color())

    # Species 2 (orange)
    (line2,) = ax.plot(x, phi2, linewidth=6.0, label=r"$\phi_2(x)$")
    # ax.plot(x, phi2_hard, linestyle="--", linewidth=2.5, color=line2.get_color())

    (line0,) = ax.plot(x, phi0, linewidth=6.0, label=r"$\phi_0(x)$")
    # ax.plot(x, phi0_hard, linestyle="--", linewidth=2.5, color=line0.get_color())

    # ax.axvline(0, color="0.3", linestyle=":", linewidth=2.5)

    # Bulk-value labels
    x_left_text = -L / 2 - 1.0
    x_right_text = 0

    # ax.text(x_left_text, phi1_left + 0.02, r"$\phi_1^{(1)}$", color=line1.get_color())
    # ax.text(x_left_text, phi2_left + 0.02, r"$\phi_0^{(1)}$", color=line2.get_color())
    # ax.text(x_left_text, phi0_left + 0.02, r"$\phi_2^{(1)}$", color=line0.get_color())

    # ax.text(x_right_text, phi1_right - 0.05, r"$\phi_1^{(2)}$", color=line1.get_color())
    # ax.text(x_right_text, phi2_right - 0.05, r"$\phi_0^{(2)}$", color=line2.get_color())
    # ax.text(x_right_text, phi0_right - 0.05, r"$\phi_2^{(2)}$", color=line0.get_color())

    ax.set_ylim(0, 0.7)
    ax.set_xlim(-L / 2, L / 2)
    ax.set_yticks([])
    ax.set_xticks([])

    if d_idx == 0:
        con = ConnectionPatch(
            xyA=(0.5, 1.0),
            coordsA=ax.transAxes,
            xyB=(bin_phis[0, 0, d_idx], bin_phis[0, 1, d_idx]),
            coordsB=main_ax.transData,
            color="black",
            lw=3.0,
            zorder=1,
        )
        fig.add_artist(con)
        con = ConnectionPatch(
            xyA=(0.5, 0.0),
            coordsA=ax.transAxes,
            xyB=(bin_phis[1, 0, d_idx], bin_phis[1, 1, d_idx]),
            coordsB=main_ax.transData,
            color="black",
            lw=3.0,
            zorder=1,
        )
        fig.add_artist(con)
    else:
        con = ConnectionPatch(
            xyA=(0.0, 1.0),
            coordsA=ax.transAxes,
            xyB=(bin_phis[0, 0, d_idx], bin_phis[0, 1, d_idx]),
            coordsB=main_ax.transData,
            color="black",
            lw=3.0,
            zorder=1,
        )
        fig.add_artist(con)
        con = ConnectionPatch(
            xyA=(0.50, 0.0),
            coordsA=ax.transAxes,
            xyB=(bin_phis[1, 0, d_idx], bin_phis[1, 1, d_idx]),
            coordsB=main_ax.transData,
            color="black",
            lw=3.0,
            zorder=1,
        )
        fig.add_artist(con)

fig.savefig("figures/workflow_chapter_two.png", dpi=300)
