"""
Plot the scaling of the two factors in the surface tension approximation with the chemical potential.
First factor: |\\Delta \\phi| \\sim |\\mu_{2,c} - \\mu_2|^{1/2}
Second factor: f(\\bar{\\phi}) - f(\\phi^{(1)}) - \\mu_i  (\\bar{\\phi}_i - \\phi_i^{(1)}) \\sim |\\mu_{2,c} - \\mu_2|^2
"""

import json
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
LMD = 1.25
CACHE_FILE = "plot_generation/factor_scaling.json"
N_REPRESENTATIVES = 5


def compute_all_results():
    results = []
    for fn in tqdm(sorted(os.listdir("chapter_two_data"))):
        try:
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
                factor1 = np.zeros(phis.shape[0])
                factor2 = np.zeros(phis.shape[0])
                mu_r = np.zeros_like(factor1)
                for i in range(phis.shape[0]):
                    box = ComputationBox((N,), (bs[i],))
                    model = GradientExplicit(box, chis, lmd=LMD, dt=0.01)
                    model(phis_torch[i])
                    if model.error_max.item() > 1e-8:
                        raise ValueError(
                            f"Expected simulation to be converged, but found error_max={model.error_max.item()} in {fn} at index {i}"
                        )
                    del_phis = phis[i, :, 0] - phis[i, :, N // 2]
                    phi_bar = (phis[i, :, 0] + phis[i, :, N // 2]) / 2
                    f_1 = model._free_energy_local(phis_torch[i, :, 0])
                    f_bar = model._free_energy_local(torch.tensor(phi_bar))
                    mu = model.chemical_potential(phis_torch[i])[:, 0].numpy()
                    factor1[i] = np.linalg.norm(del_phis) ** 2
                    factor2[i] = (
                        f_bar.item() - f_1.item() - np.dot(mu, phi_bar - phis[i, :, 0])
                    )
                    mu_r[i] = -model.chemical_potential(phis_torch[i])[1, 0].item()
                mu_rc = model.chemical_potential(
                    np.array([[cp.phi1, cp.phi2] for cp in sp.critical_points]).T,
                    is_gradient=False,
                )[1]
                mu_rc = mu_rc[np.argmin(np.abs(mu_rc + mu_r[-1]))].item()
                mu_r += mu_rc
                if mu_r.min() < 2e-4:
                    continue
                if np.isnan(factor1).any() or np.isnan(factor2).any():
                    isnan_mask = np.isnan(factor1) | np.isnan(factor2)
                    mu_r = mu_r[~isnan_mask]
                    factor1 = factor1[~isnan_mask]
                    factor2 = factor2[~isnan_mask]
                results.append(
                    {
                        "fn": fn,
                        "factor1": factor1.tolist(),
                        "factor2": factor2.tolist(),
                        "mu_r": mu_r.tolist(),
                        "chi_DS": float(chi_DS),
                        "chi_RS": float(chi_RS),
                        "chi_DR": float(chi_DR),
                    }
                )
        except ValueError as e:
            print(f"Error processing file {fn}: {e}")
            break
    return results


def load_or_compute_results():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached results from {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    print("Computing results from chapter_two_data/...")
    results = compute_all_results()
    with open(CACHE_FILE, "w") as f:
        json.dump(results, f)
    print(f"Cached {len(results)} results to {CACHE_FILE}")
    return results


def select_representatives(results, n=N_REPRESENTATIVES):
    results_valid = [r for r in results if not np.isnan(r["factor1"][0])]
    f1_0_values = np.array([r["factor1"][0] for r in results_valid])
    lo, hi = f1_0_values.min(), f1_0_values.max()
    targets = np.linspace(lo, hi, n)
    selected_indices = []
    for t in targets:
        dists = np.abs(f1_0_values - t)
        # Avoid picking the same index twice
        for idx in np.argsort(dists):
            if idx not in selected_indices:
                selected_indices.append(int(idx))
                break
    return selected_indices


results = load_or_compute_results()
# Get min max for all chi values
chi_DS_values = np.array([r["chi_DS"] for r in results])
chi_RS_values = np.array([r["chi_RS"] for r in results])
chi_DR_values = np.array([r["chi_DR"] for r in results])
print(f"chi_DS: min={chi_DS_values.min()}, max={chi_DS_values.max()}")
print(f"chi_RS: min={chi_RS_values.min()}, max={chi_RS_values.max()}")
print(f"chi_DR: min={chi_DR_values.min()}, max={chi_DR_values.max()}")
print(f"Total valid files: {len(results)}")

rep_indices = select_representatives(results)
print(
    f"Selected representatives (factor1[0] values): {[results[i]['factor1'][0] for i in rep_indices]}"
)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
colors = [f"C{i}" for i in range(N_REPRESENTATIVES)]

mu_r_first = None
for plot_idx, ri in enumerate(rep_indices):
    r = results[ri]
    mu_r = np.array(r["mu_r"])
    factor1 = np.array(r["factor1"])
    factor2 = np.array(r["factor2"])
    c = colors[plot_idx]
    axes[0].scatter(mu_r, factor1, s=100, color=c, edgecolor="black", zorder=10)
    axes[0].plot(mu_r, factor1, "-", color=c, linewidth=4.5, zorder=9)
    axes[1].scatter(mu_r, factor2, s=100, color=c, edgecolor="black", zorder=10)
    axes[1].plot(mu_r, factor2, "-", color=c, linewidth=4.5, zorder=9)
    if mu_r_first is None:
        mu_r_first = mu_r
        print(f"chi_DS, chi_RS, chi_DR: {r['chi_DS']}, {r['chi_RS']}, {r['chi_DR']}")

if mu_r_first is not None:
    mu_plot = np.linspace(3e-4, 1e-1, 2)
    axes[0].plot(
        mu_plot,
        mu_plot * 2,
        color="black",
        linestyle="--",
        linewidth=3.5,
        zorder=8,
    )
    axes[0].text(
        mu_plot[-1] * 0.1,
        (mu_plot[-1] * 0.1) * 3,
        r"$ \sim |\mu_{c} - \mu|$",
        fontsize=32,
        ha="right",
    )
    axes[1].plot(
        mu_plot,
        mu_plot**2 * 1.5,
        color="black",
        linestyle="--",
        linewidth=3.5,
        zorder=8,
    )
    axes[1].text(
        mu_plot[-1] * 0.1,
        (mu_plot[-1] * 0.1) ** 2 * 1.5**2,
        r"$ \sim |\mu_{c} - \mu|^{2}$",
        fontsize=32,
        ha="right",
    )

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xticks([])
axes[0].set_ylabel(r"first factor")
axes[0].text(7e-6, 6e-1, "A", color="black", fontsize=36)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel(r"$\mu_{c} - \mu$")
axes[1].set_ylabel(r"second factor")
axes[1].text(7e-6, 5.5e-2, "B", color="black", fontsize=36)
plt.tight_layout()
plt.savefig("figures/factor_scaling.png", dpi=300)
