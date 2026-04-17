import json
import os

import flory
import matplotlib.pyplot as plt
import mpltern
import numpy as np
import plot_params
import shapely
from pyphasediagram.diagram import PhaseDiagram


def get_volumes(three_phase_phis: np.ndarray, phi_mean: np.ndarray):

    v = np.linalg.solve(three_phase_phis, phi_mean)
    v /= v.sum()  # Normalize to get volume fractions
    return v


def flory_energy(chis: np.ndarray, phi: np.ndarray) -> float:
    """Flory-Huggins free energy of mixing in the reduced two-component form.

    Parameters
    ----------
    chis : (2, 2) array
        Reduced chi-parameter matrix.
    phi : (2,) array
        Volume fractions of components 1 and 2 (component 0 is 1 - phi[0] - phi[1]).

    Returns
    -------
    float
        f = sum_i phi_i ln(phi_i) + 0.5 * phi^T chi phi
        where phi = (phi_1, phi_2) and the solvent fraction is implicit.
    """
    phi = np.asarray(phi, dtype=float)
    phi0 = 1.0 - phi[0] - phi[1]
    # Entropic term over all three components
    all_phi = np.array([phi0, phi[0], phi[1]])
    entropic = np.sum(all_phi * np.log(all_phi))
    # Enthalpic term in reduced form
    enthalpic = 0.5 * phi @ chis @ phi
    return float(entropic + enthalpic)


chi = 3.5
# chis_3x3 = np.array(
#     [
#         [0.0, 2.3684210526315788, 3.947368421052632],
#         [2.3684210526315788, 0.0, 3.947368421052632],
#         [3.947368421052632, 3.947368421052632, 0.0],
#     ]
# )
chis_3x3 = np.array([[0, chi, chi], [chi, 0, chi], [chi, chi, 0]])
c01, c02, c12 = chis_3x3[0, 1], chis_3x3[0, 2], chis_3x3[1, 2]
chis_reduced = np.array([[-2 * c01, c12 - c01 - c02], [c12 - c01 - c02, -2 * c02]])
diagram = PhaseDiagram(chis_3x3)
diagram.build()

fig, ax = plt.subplots(subplot_kw={"projection": "ternary"})

# Sample random points inside the three-phase region
phi1, phi2 = diagram.binodal.three_phase_polygons[0].exterior.xy
phi0 = 1 - np.array(phi1) - np.array(phi2)
ax.fill(
    phi0,
    phi1,
    phi2,
    color="orange",
    alpha=0.5,
    zorder=1,
    edgecolor="black",
    linewidth=1,
    label="3-phase region",
)
phi1, phi2 = (
    phi1[:-1],
    phi2[:-1],
)  # Remove the closing point which is the same as the first

phi0 = 1 - np.array(phi1) - np.array(phi2)
bin_phis = np.array([phi1, phi2]).T

# Plot the three-phase polygon boundary
ax.plot(phi0, np.array(phi1), np.array(phi2), color="black", linewidth=1)

failed_count = 0
count = 0
success_label, failed_label = True, True
if os.path.exists("plot_generation/tpp_cache.json"):
    with open("plot_generation/tpp_cache.json", "r") as f:
        tpp_cache = json.load(f)

    for entry in tpp_cache:
        tpp = np.array(entry["tpp"])
        success = entry["success"]
        tpp0 = 1 - tpp.sum()
        if success:
            ax.scatter(
                tpp0,
                tpp[0],
                tpp[1],
                color="green",
                marker="o",
                s=100,
                edgecolor="black",
                label="3 phases found" if success_label else None,
            )
            success_label = False
        else:
            ax.scatter(
                tpp0,
                tpp[0],
                tpp[1],
                color="red",
                marker="x",
                s=100,
                label="2 phases found" if failed_label else None,
            )
            failed_label = False


else:
    tpp_cache = []
    while count < 30:
        bin_volumes = np.random.dirichlet(np.ones(3))  # Random volumes that sum to 1
        if failed_count < 15:
            while (
                bin_volumes.min() > 0.01
            ):  # Ensure we are close to the edge of the three-phase region
                bin_volumes = np.random.dirichlet(np.ones(3))
        tpp = bin_volumes @ bin_phis
        # Compute the free energy at the test point and the coexisting phases
        f_test = flory_energy(chis_reduced, tpp)
        bin_volumes = get_volumes(
            np.array([phi0, phi1, phi2]), np.r_[1 - tpp.sum(), tpp]
        )
        f_bin = sum(
            flory_energy(chis_reduced, phi) * v for phi, v in zip(bin_phis, bin_volumes)
        )

        phases = flory.find_coexisting_phases(
            3, chis_3x3, np.r_[1 - tpp.sum(), tpp], progress=False
        )
        flory_phis = phases.fractions[:, 1:]
        n_phases = phases.fractions.shape[0]
        flory_volumes = phases.volumes
        f_flory = sum(
            flory_energy(chis_reduced, phi) * v
            for phi, v in zip(flory_phis, flory_volumes)
        )
        tpp0 = 1 - tpp.sum()
        if f_bin - f_flory < -1e-4 or n_phases < 3:
            ax.scatter(tpp0, tpp[0], tpp[1], color="red", marker="x", s=100)
            tpp_cache.append((tpp, False))
            failed_count += 1
            count += 1
        elif abs(f_bin - f_flory) < 1e-4:
            if failed_count >= 15:
                ax.scatter(
                    tpp0,
                    tpp[0],
                    tpp[1],
                    color="green",
                    marker="o",
                    s=100,
                    edgecolor="black",
                )
                tpp_cache.append((tpp, True))
                count += 1
        else:
            raise ValueError(
                f"Flory found a higher free energy than the binodal! f_bin={f_bin}, f_flory={f_flory} at tpp={tpp}, volumes={flory_volumes}, phases={flory_phis}"
            )

    # store tpp_cache
    with open("plot_generation/tpp_cache.json", "w") as f:
        json.dump(
            [{"tpp": tpp.tolist(), "success": success} for tpp, success in tpp_cache],
            f,
            indent=2,
        )
ax.taxis.set_ticks([0, 1])
ax.laxis.set_ticks([0, 1])
ax.raxis.set_ticks([0, 1])

ax.set_tlabel(r"$\phi_0$", fontsize=20)
ax.set_llabel(r"$\phi_1$", fontsize=20)
ax.set_rlabel(r"$\phi_2$", fontsize=20)


# ax.legend(
#     loc="upper left",
#     bbox_to_anchor=(1.05, 1.1),
#     bbox_transform=ax.transAxes,
#     frameon=False,
# )
fig.tight_layout()
plt.savefig("figures/flory_fails.png", dpi=300)

plt.show()
