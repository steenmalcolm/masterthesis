import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plot_params
import torch
from pyphasediagram.spinodal import Spinodal
from surfacetension.solvers import ComputationBox, GradientExplicit
from tqdm import tqdm

np.random.seed(42)
chi01, chi02, chi12 = [], [], []
LMD = 1.25
debug_count = 0
fig, ax = plt.subplots(figsize=(12, 8))
file_idx = 0
mu_r_first = None
try:
    for fn in os.listdir("chapter_two_data"):
        with h5py.File(os.path.join("chapter_two_data", fn), "r") as f:
            if len(f.keys()) == 0:
                continue
            if len(f.keys()) > 1:
                raise ValueError(
                    f"Expected only one key in {fn}, but found {len(f.keys())}"
                )
            key = list(f.keys())[0]
            data = f[key]
            eps = data["eps"][:]
            chis = np.array(
                [
                    [
                        eps[i, j] - 1 / 2 * (eps[i, i] + eps[j, j])
                        for j in range(eps.shape[1])
                    ]
                    for i in range(eps.shape[0])
                ]
            )
            chi_DR, chi_DS, chi_RS = chis[0, 1], chis[0, 2], chis[1, 2]
            chis_r = np.array(
                [
                    [-2 * chi_DS, chi_DR - chi_DS - chi_RS],
                    [chi_DR - chi_DS - chi_RS, -2 * chi_RS],
                ],
            )

            sp = Spinodal(chis_r)
            sp.build()
            assert (
                len(sp.critical_points) >= 1
            ), f"Expected exactly one critical point, but found {len(sp.critical_points)} in {fn}"
            cp = sp.critical_points[0]
            chi01.append(chis[0, 1])
            chi02.append(chis[0, 2])
            chi12.append(chis[1, 2])
            bs = data["box_size"][:]
            phis = data["profile_eq"][:]
            # Data may not have zeta so access safely
            zeta = data.attrs.get("zeta", 0)
            if zeta != 0:
                raise ValueError(f"Expected zeta=0, but found zeta={zeta} in {fn}")

            N = phis.shape[-1]
            colormap = plt.get_cmap("Blues")
            # for i in range(phis.shape[0]):

            phis_torch = torch.tensor(phis)
            st = np.zeros(phis.shape[0])
            st_approx = np.zeros_like(st)
            mu_r = np.zeros_like(st)
            for i in range(phis.shape[0]):
                box = ComputationBox((N,), (bs[i],))
                model = GradientExplicit(box, chis, lmd=LMD, dt=0.01)
                model(phis_torch[i])
                if model.error_max.item() > 1e-8:
                    raise ValueError(
                        f"Expected simulation to be converged, but found error_max={model.error_max.item()} in {fn} at index {i}"
                    )
                phi_bulk = phis[i, :, [0, N // 2]]
                del_phi = phi_bulk[:, 1] - phi_bulk[:, 0]
                phi_bar = phi_bulk[:, 0] + del_phi / 2
                mu = model.chemical_potential(phis_torch[i])[:, 0].numpy()
                f_bulk = model._free_energy_local(torch.tensor(phi_bulk[:, 0])).item()
                f_if = model._free_energy_local(torch.tensor(phi_bar)).item()
                first_term = del_phi @ (model.kappas_r.numpy() @ del_phi)
                second_term = f_if - f_bulk - mu @ (phi_bar - phi_bulk[:, 0])
                st_approx[i] = np.sqrt(-2 * first_term * second_term)
                st[i] = model.surface_tension(phis_torch[i]).item()
                mu_r[i] = -model.chemical_potential(phis_torch[i])[1, 0].item()

            mu_rc = model.chemical_potential(
                np.array([[cp.phi1, cp.phi2] for cp in sp.critical_points]).T,
                is_gradient=False,
            )[1]
            mu_rc = mu_rc[np.argmin(np.abs(mu_rc + mu_r[-1]))].item()
            mu_r += mu_rc
            plt.plot(mu_r, st)
            plt.plot(mu_r, st_approx, "--")
            plt.xscale("log")
            plt.yscale("log")
            if mu_r.min() < 2e-4:
                continue
            if file_idx == 30:
                mu_r_first = mu_r
                print(f"chi01, chi02, chi12: {chi_DS}, {chi_RS}, {chi_DR}")
                ax.scatter(
                    mu_r,
                    st,
                    s=150,
                    color="C0",
                    edgecolor="black",
                    zorder=10,
                )
                ax.plot(mu_r, st, "-", color="C0", linewidth=1.5, zorder=9)
            else:
                ax.scatter(
                    mu_r,
                    st,
                    s=20,
                    color="gray",
                    edgecolor="none",
                    alpha=0.3,
                    zorder=2,
                )
                ax.plot(mu_r, st, "-", color="gray", linewidth=0.8, alpha=0.3, zorder=1)
            file_idx += 1
except ValueError as e:
    print(f"Error processing file: {e}")

if mu_r_first is not None:
    mu_plot = np.sort(mu_r_first)
    ax.plot(
        mu_plot,
        mu_plot ** (3 / 2) * np.sqrt(8 / 9 * LMD),
        color="black",
        linestyle="--",
        linewidth=1.5,
        zorder=8,
    )
    ax.text(
        mu_plot[-1] * 0.01,
        (mu_plot[-1] * 0.01) ** (3 / 2) * np.sqrt(8 / 9 * LMD) * 1.5,
        r"$ \sim (\mu_{2,c} - \mu_2)^{3/2}$",
        fontsize=32,
        ha="right",
    )

ax.set_xlabel(r"$\mu_{2,c} - \mu_2$")
ax.set_ylabel(r"$\gamma$")
ax.set_xscale("log")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("figures/st_simulation_results.png", dpi=300)
plt.show()

print(file_idx)
