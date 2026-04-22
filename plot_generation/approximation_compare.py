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
CACHE_FILE = "plot_generation/approximation_compare.json"


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
                cp = sp.critical_points[0]
                bs = data["box_size"][:]
                phis = data["profile_eq"][:]
                zeta = data.attrs.get("zeta", 0)
                if zeta != 0:
                    raise ValueError(f"Expected zeta=0, but found zeta={zeta} in {fn}")

                N = phis.shape[-1]
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
                    del_phi = phis[i, :, 0] - phis[i, :, N // 2]
                    phi_bar = (phis[i, :, 0] + phis[i, :, N // 2]) / 2
                    mu = model.chemical_potential(phis_torch[i])[:, 0].numpy()
                    f_bulk = model._free_energy_local(phis_torch[i, :, 0]).item()
                    f_if = model._free_energy_local(torch.tensor(phi_bar)).item()
                    first_term = del_phi @ (model.kappas_r.numpy() @ del_phi)
                    second_term = f_if - f_bulk - mu @ (phi_bar - phis[i, :, 0])
                    st_approx[i] = np.sqrt(first_term * second_term)
                    st[i] = model.surface_tension(phis_torch[i]).item()
                    mu_r[i] = -model.chemical_potential(phis_torch[i])[1, 0].item()

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
                    st = st[~isnan_mask]
                    mu_r = mu_r[~isnan_mask]
                    st_approx = st_approx[~isnan_mask]
                results.append(
                    {
                        "fn": fn,
                        "st": st.tolist(),
                        "st_approx": st_approx.tolist(),
                        "mu_r": mu_r.tolist(),
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


results = load_or_compute_results()
print(f"Total valid files: {len(results)}")

# Compute mean relative error for each dataset
mean_rel_errors = []
for i, r in enumerate(results):
    st = np.array(r["st"])
    st_approx = np.array(r["st_approx"])
    mean_rel_errors.append(np.mean(np.abs(st - st_approx) / st))

idx_min = int(np.argmin(mean_rel_errors))
idx_max = int(np.argmax(mean_rel_errors))
print(
    f"Min error dataset: {results[idx_min]['fn']} (mean rel error: {mean_rel_errors[idx_min]:.4f})"
)
print(
    f"Max error dataset: {results[idx_max]['fn']} (mean rel error: {mean_rel_errors[idx_max]:.4f})"
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xticks([])
axes[0].set_xlim(2e-4, 3)
axes[0].set_ylim(1e-6, 5e-1)
axes[0].text(4.5e-5, 7e-2, "A", color="black", fontsize=36)
axes[1].set_xlim(2e-4, 3)
axes[0].set_ylabel(r"$\gamma$")
axes[1].set_xscale("log")
axes[1].set_xlabel(r"$\mu_{2,c} - \mu_2$")
axes[1].set_ylabel(r"$\epsilon_\mathrm{approx.}$ (%)")
axes[1].set_ylim(0, 100)
axes[1].text(4.5e-5, 90, "B", color="black", fontsize=36)

# Fill between min and max error datasets on the bottom panel
r_min = results[idx_min]
r_max = results[idx_max]
mu_r_min = np.array(r_min["mu_r"])
mu_r_max = np.array(r_max["mu_r"])
err_min = (
    100
    * np.abs(np.array(r_min["st"]) - np.array(r_min["st_approx"]))
    / np.array(r_min["st"])
)
err_max = (
    100
    * np.abs(np.array(r_max["st"]) - np.array(r_max["st_approx"]))
    / np.array(r_max["st"])
)

# Interpolate both onto a common mu_r grid for fill_between
all_mu = np.sort(np.unique(np.concatenate([mu_r_min, mu_r_max])))
err_min_interp = np.interp(all_mu, np.sort(mu_r_min), err_min[np.argsort(mu_r_min)])
err_max_interp = np.interp(all_mu, np.sort(mu_r_max), err_max[np.argsort(mu_r_max)])

lower = np.minimum(err_min_interp, err_max_interp)
upper = np.maximum(err_min_interp, err_max_interp)
axes[1].fill_between(all_mu, lower, upper, color="gray", alpha=0.3, zorder=0)

# Plot the first dataset (index 0) as the highlighted one
r0 = results[0]
mu_r = np.array(r0["mu_r"])
st = np.array(r0["st"])
st_approx = np.array(r0["st_approx"])

ax = axes[0]
ax.plot(
    mu_r,
    st_approx,
    "-",
    color="blue",
    zorder=1,
    linewidth=4,
    marker="o",
    markersize=10,
    markeredgecolor="black",
    label=r"$\gamma^{\mathrm{approx}}$",
)
ax.scatter(mu_r, st_approx, color="blue", edgecolor="black", s=100, zorder=2, alpha=0.7)
ax.plot(
    mu_r,
    st,
    color="red",
    zorder=1,
    linewidth=4.0,
    label=r"$\gamma$",
    marker="^",
    markersize=10,
    markeredgecolor="black",
)
ax.scatter(mu_r, st, color="red", edgecolor="black", s=100, zorder=1, marker="^")
ax.legend(loc="upper left")

ax = axes[1]
ax.plot(
    mu_r,
    100 * np.abs(st - st_approx) / st,
    "-",
    color="blue",
    zorder=1,
    linewidth=4.0,
)
ax.scatter(
    mu_r,
    100 * np.abs(st - st_approx) / st,
    color="blue",
    edgecolor="black",
    s=100,
    zorder=2,
)

plt.tight_layout()
plt.savefig("figures/approximation_compare.png", dpi=300)
# plt.show()

print(len(results))
