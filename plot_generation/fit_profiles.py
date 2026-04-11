import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plot_params
import torch
from pyphasediagram.spinodal import Spinodal
from scipy.optimize import curve_fit
from surfacetension.solvers import ComputationBox, GradientExplicit

np.random.seed(42)
LMD = 1.25
N_REPRESENTATIVES = 5


def fit_tanh(x, del_x, w, x0, bar_x):
    return bar_x + del_x * np.tanh((x - x0) / w)


def compute_results():
    results = None
    for fn in sorted(os.listdir("chapter_two_data")):
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
            bs = data["box_size"][:]
            phis = data["profile_eq"][:]
            zeta = data.attrs.get("zeta", 0)
            if zeta != 0:
                raise ValueError(f"Expected zeta=0, but found zeta={zeta} in {fn}")

            N = phis.shape[-1]
            phis_torch = torch.tensor(phis)
            st_error = np.zeros(phis.shape[0])
            mu_r = np.zeros_like(st_error)
            x0_fit = np.zeros((st_error.shape[0], 2))
            fit_params = []

            for i in range(phis.shape[0]):
                box = ComputationBox((N,), (bs[i],))
                model = GradientExplicit(box, chis, lmd=LMD, dt=0.01)
                model(phis_torch[i])
                if model.error_max.item() > 1e-8:
                    raise ValueError(
                        f"Expected simulation to be converged, but found error_max={model.error_max.item()} in {fn} at index {i}"
                    )
                del_phi = phis[i, :, 0] - phis[i, :, N // 2]
                phi_bar = (phis[i, :, 0] + phis[i, :, N // 2]) / 2
                mu = model.chemical_potential(phis_torch[i])[:, 0].numpy()
                f_bulk = model._free_energy_local(phis_torch[i, :, 0]).item()
                f_if = model._free_energy_local(torch.tensor(phi_bar)).item()
                first_term = del_phi @ (model.kappas_r.numpy() @ del_phi)
                second_term = f_if - f_bulk - mu @ (phi_bar - phis[i, :, 0])
                st_approx = np.sqrt(first_term * second_term)
                st = model.surface_tension(phis_torch[i]).item()
                st_error[i] = np.abs(st - st_approx) / st
                mu_r[i] = -model.chemical_potential(phis_torch[i])[1, 0].item()
                # Fit tanh to the profiles `phis[i, 0, :N//2]` and `1 - phis[i, 0, :N//2] - phis[i, 1, :N//2]`
                x = np.arange(N // 2)
                profile1 = phis[i, 0, : N // 2]
                profile2 = 1 - phis[i, 0, : N // 2] - phis[i, 1, : N // 2]
                p0_1 = [
                    -(profile1[0] - profile1[-1]) / 2,
                    N // 50,
                    N // 4,
                    np.mean(profile1),
                ]
                popt1, _ = curve_fit(fit_tanh, x, profile1, p0=p0_1, maxfev=10000)
                p0_2 = [
                    -(profile2[0] - profile2[-1]) / 2,
                    N // 50,
                    N // 4,
                    np.mean(profile2),
                ]
                popt2, _ = curve_fit(fit_tanh, x, profile2, p0=p0_2, maxfev=10000)
                x0_fit[i, 0] = popt1[2] / x[-1] * bs[i]
                x0_fit[i, 1] = popt2[2] / x[-1] * bs[i]
                fit_params.append(
                    {
                        "popt1": popt1.tolist(),
                        "popt2": popt2.tolist(),
                        "profile1": profile1.tolist(),
                        "profile2": profile2.tolist(),
                        "N": int(N),
                    }
                )

            mu_rc = model.chemical_potential(
                np.array([[cp.phi1, cp.phi2] for cp in sp.critical_points]).T,
                is_gradient=False,
            )[1]
            mu_rc = mu_rc[np.argmin(np.abs(mu_rc + mu_r[-1]))].item()
            mu_r += mu_rc
            if mu_r.min() < 2e-4:
                continue
            if np.isnan(st).any():
                isnan_mask = np.isnan(st)
                mu_r = mu_r[~isnan_mask]
                st = st[~isnan_mask]
            results = {
                "fn": fn,
                "st_error": st_error,
                "mu_r": mu_r,
                "del_x0_fit": np.abs(x0_fit[:, 0] - x0_fit[:, 1]),
                "fit_params": fit_params,
                "chi_DS": float(chi_DS),
                "chi_RS": float(chi_RS),
                "chi_DR": float(chi_DR),
            }
            break

    return results


results = compute_results()
print(f"File: {results['fn']}")

mu_r = results["mu_r"]
del_x0 = results["del_x0_fit"]
fit_params = results["fit_params"]

# Pick one representative profile (middle index) to show in the first subplot
mid_idx = len(fit_params) // 2
fp = fit_params[mid_idx]

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# --- Panel A: profiles + fits ---
ax = axes[0]
N = fp["N"]
x = np.arange(100, 160)
x_fine = np.linspace(100, 160, 500)
profile1 = np.array(fp["profile1"])[100:160]
profile2 = np.array(fp["profile2"])[100:160]
popt1 = np.array(fp["popt1"])
popt2 = np.array(fp["popt2"])
popt1[2] += 2
popt2[2] -= 2
# ax.plot(x, profile1, "o", color="C0", markersize=3, alpha=0.6)
ax.plot(x_fine, fit_tanh(x_fine, *popt1), "-", color="C0", linewidth=7)
# ax.plot(x, profile2, "s", color="C1", markersize=3, alpha=0.6)
ax.plot(x_fine, fit_tanh(x_fine, *popt2), "-", color="C1", linewidth=7)
ax.text(x[0] + 1, fit_tanh(x[0], *popt1) + 0.015, r"$\phi_1$", fontsize=30, color="C0")
ax.text(x[0] + 1, fit_tanh(x[0], *popt2) - 0.03, r"$\phi_0$", fontsize=30, color="C1")
ax.axvline(
    popt1[2], color="C0", linestyle=":", label=r"$x_0$ fit for $\phi_1$", linewidth=7
)
ax.axvline(
    popt2[2], color="C1", linestyle=":", label=r"$x_0$ fit for $\phi_2$", linewidth=7
)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\phi$")
ax.set_xlim(100, 160)
ax.set_xticks([100, popt1[2], popt2[2], 160])
ax.set_xticklabels(["-L/2", "", "", "L/2"])
ax.text(127, 0.305, r"$\Delta x_0$", fontsize=28, color="green")
ax.text(80, 0.53, "A", color="black", fontsize=36)

# --- Panel B: del_x0 vs mu_c - mu ---
ax = axes[1]
ax.scatter(mu_r, del_x0, s=100, color="green", edgecolor="black", zorder=10, marker="^")
ax.plot(mu_r, del_x0, "-", color="green", linewidth=7, zorder=9)
ax.set_xlabel(r"$\mu_c - \mu$")
ax.set_ylabel(r"$|\Delta x_0|$", color="green")
ax.tick_params(axis="y", labelcolor="green")
ax.set_xscale("log")
ax.set_ylim(0, None)
ax.text(1.6e-5, 1.13, "B", color="black", fontsize=36)

ax2 = ax.twinx()
st_error = results["st_error"]
ax2.scatter(
    mu_r, 100 * st_error, s=100, color="blue", edgecolor="black", zorder=10, alpha=0.7
)
ax2.plot(mu_r, 100 * st_error, "-", color="blue", linewidth=7, zorder=9)
ax2.set_ylabel(r"$\epsilon_\mathrm{approx.}$ (%)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
ax2.set_ylim(0, None)

plt.tight_layout()
plt.savefig("figures/fit_profiles.png", dpi=300)
