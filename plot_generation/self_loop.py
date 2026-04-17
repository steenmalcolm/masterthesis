import matplotlib.pyplot as plt
import mpltern
import numpy as np
import plot_params
from pyphasediagram.binodal import Binodal
from pyphasediagram.diagram import PhaseDiagram
from pyphasediagram.spinodal import Spinodal

np.random.seed(4)


chi = 2.7
chis = np.array([[0, chi, chi], [chi, 0, chi], [chi, chi, 0]])
chis_r = np.array([[-2 * chi, -chi], [-chi, -2 * chi]])
binodal = Binodal(chis_r)
spinodal = Spinodal(chis_r)
binodal._find_phase_polygons = lambda: None
binodal.build()
# spinodal.build()
b = binodal
s = spinodal

fig, ax = plt.subplots(
    figsize=(8, 8),
    subplot_kw={"projection": "ternary"},
)

is_b_label = True
for s_id, section in enumerate(b.binodal_sections):
    if len(section) < 3000:
        continue
    for i, phis in enumerate(section.phis):
        phi1, phi2 = phis
        phi0 = 1 - phi1 - phi2

        ax.scatter(
            phi0[-1],
            phi1[-1],
            phi2[-1],
            marker="*",
            color="red",
            edgecolors="black",
            zorder=3,
            s=300,
            label="initial/final point" if is_b_label else None,
        )
        ax.plot(
            phi0,
            phi1,
            phi2,
            color="blue",
            zorder=2,
            linewidth=3.5,
            label="binodal" if is_b_label else None,
        )
        is_b_label = False

num_tie_lines = 15
is_tl_label = True
for s_id, section in enumerate(b.binodal_sections):
    if len(section) < 3000:
        continue
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
            label="tie-lines" if is_tl_label else None,
        )
        is_tl_label = False

for i, (phi1_sp, phi2_sp) in enumerate(zip(*s.get_spinodal_coords())):
    phi0_sp = 1 - phi1_sp - phi2_sp
    if phi1_sp[0] > 0.1 and phi2_sp[0] > 0.1 and phi0_sp[0] > 0.1:
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
ax.taxis.set_ticks([0, 1])
ax.laxis.set_ticks([0, 1])
ax.raxis.set_ticks([0, 1])
# ax.taxis.set_ticks([])
# ax.laxis.set_ticks([])
# ax.raxis.set_ticks([])

ax.set_tlabel(r"$\phi_0$", fontsize=18)
ax.set_llabel(r"$\phi_1$", fontsize=18)
ax.set_rlabel(r"$\phi_2$", fontsize=18)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.74, 1.25),
    bbox_transform=ax.transAxes,
    frameon=False,
)
fig.tight_layout()
fig.savefig(f"figures/self_loop.png", dpi=300)
