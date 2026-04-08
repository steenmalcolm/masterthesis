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
CACHE_FILE = "plot_generation/st_simulation_results.json"
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
                st = np.zeros(phis.shape[0])
                mu_r = np.zeros_like(st)
                for i in range(phis.shape[0]):
                    box = ComputationBox((N,), (bs[i],))
                    model = GradientExplicit(box, chis, lmd=LMD, dt=0.01)
                    model(phis_torch[i])
                    if model.error_max.item() > 1e-8:
                        raise ValueError(
                            f"Expected simulation to be converged, but found error_max={model.error_max.item()} in {fn} at index {i}"
                        )
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
                    mu_r = mu_r[~isnan_mask]
                    st = st[~isnan_mask]
                results.append(
                    {
                        "fn": fn,
                        "st": st.tolist(),
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
    results = [r for r in results if not np.isnan(r["st"][0])]
    st0_values = np.array([r["st"][0] for r in results])
    lo, hi = st0_values.min(), st0_values.max()
    targets = np.linspace(lo, hi, n)
    selected_indices = []
    for t in targets:
        dists = np.abs(st0_values - t)
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
    f"Selected representatives (st[0] values): {[results[i]['st'][0] for i in rep_indices]}"
)

fig, ax = plt.subplots(figsize=(12, 8))
colors = [f"C{i}" for i in range(N_REPRESENTATIVES)]

mu_r_first = None
for plot_idx, ri in enumerate(rep_indices):
    r = results[ri]
    mu_r = np.array(r["mu_r"])
    st = np.array(r["st"])
    c = colors[plot_idx]
    ax.scatter(mu_r, st, s=150, color=c, edgecolor="black", zorder=10)
    ax.plot(mu_r, st, "-", color=c, linewidth=3.5, zorder=9)
    if mu_r_first is None:
        mu_r_first = mu_r
        print(f"chi_DS, chi_RS, chi_DR: {r['chi_DS']}, {r['chi_RS']}, {r['chi_DR']}")

if mu_r_first is not None:
    mu_plot = np.sort(mu_r_first)
    mu_plot = np.linspace(3e-4, 1e-1, 2)
    ax.plot(
        mu_plot,
        mu_plot ** (3 / 2) * 3 * np.sqrt(8 / 9 * LMD),
        color="black",
        linestyle="--",
        linewidth=3.5,
        zorder=8,
    )
    ax.text(
        mu_plot[-1] * 0.1,
        (mu_plot[-1] * 0.1) ** (3 / 2) * 3 * np.sqrt(8 / 9 * LMD) * 1.5,
        r"$ \sim (\mu_{2,c} - \mu_2)^{\frac{3}{2}}$",
        fontsize=32,
        ha="right",
    )

ax.set_xlabel(r"$\mu_{2,c} - \mu_2$")
ax.set_ylabel(r"$\gamma$")
ax.set_xscale("log")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("figures/st_simulation_results.png", dpi=300)
