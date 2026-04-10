import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plot_params
from scipy.optimize import root_scalar

np.random.seed(42)
LMD = 1.25


def del_phi_implicit(dp, pb, pc):
    return dp - pb * np.tanh(dp / pc)


def del_phi_pl_implicit(dpl, dp, pb, pc, ps):
    exp_factor = np.exp(dpl / ps - dp / pc)
    return dpl - (1 - 2 * pb) / 2 * (1 - 1 / (1 - 2 * pb + 2 * (pb + dp) * exp_factor))


def load_symmetric(path="chapter_two_data/symmetric.h5"):
    results = []
    with h5py.File(path, "r") as f:
        for uid in f.keys():
            grp = f[uid]
            entry = {attr: float(grp.attrs[attr]) for attr in grp.attrs}
            entry["uid"] = uid
            for key in grp.keys():
                entry[key] = grp[key][:]
            results.append(entry)
    return results


def compute_all_results():
    sym_results = load_symmetric()
    results = []
    for sr in sym_results:
        chi = sr["chi"]
        eta = sr["eta"]
        phis = sr["phis"]  # shape: (num_points, 2, N)

        chi_d = -2 * eta * chi
        chi_o = chi * (1 - 2 * eta)
        phi_c = 1 / (chi_o - chi_d)
        phi_s = -1 / (chi_o + chi_d)
        mu_c = chi_d * phi_c + chi_o * phi_c + np.log(phi_c / (1 - 2 * phi_c))

        N = phis.shape[-1]
        num_points = phis.shape[0]
        del_phi = np.zeros(num_points)
        dp_exact = np.zeros(num_points)
        del_phi_pl = np.zeros(num_points)
        dpl_exact = np.zeros(num_points)
        mus = np.zeros(num_points)

        for i in range(num_points):
            phi1 = phis[i, 0, : N // 2]
            phi2 = phis[i, 1, : N // 2]
            phi0 = 1 - phi1 - phi2
            mu = chi_d * phi1[0] + chi_o * phi2[0] + np.log(phi1[0] / phi0[0])
            mus[i] = mu - mu_c
            phi_pl, phi_mn = (phi1 + phi2) / 2, (phi2 - phi1) / 2
            pb = phi_pl[0]
            sol = root_scalar(
                del_phi_implicit,
                args=(pb, phi_c),
                bracket=[1e-10, 1 / 2],
                method="bisect",
            )
            if sol.converged:
                dp = sol.root
                del_phi[i] = dp
            else:
                raise ValueError(
                    f"del_phi root finding did not converge for eta={eta:.2f} at index {i}"
                )
            sol = root_scalar(
                del_phi_pl_implicit,
                args=(dp, pb, phi_c, phi_s),
                bracket=[-0.1, 0],
                method="bisect",
            )
            if sol.converged:
                dpl = sol.root
                del_phi_pl[i] = dpl
            else:
                raise ValueError(
                    f"del_phi_pl root finding did not converge for eta={eta:.2f} at index {i}"
                )
            dp_exact[i] = phi_mn.max()
            dpl_exact[i] = phi_pl.min() - pb
        results.append(
            {
                "mus": mus,
                "del_phi": del_phi,
                "dp_exact": dp_exact,
                "del_phi_pl": del_phi_pl,
                "dpl_exact": dpl_exact,
                "ratio": eta,
            }
        )
    return results


results = compute_all_results()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].set_xscale("log")
# axes[0].set_ylim(1e-6, 5e-1)
axes[0].text(1e-6, 1.4e-2, "A", color="black", fontsize=36)
# axes[1].set_xlim(2e-4, 3)
axes[0].set_ylabel(r"$|\delta \phi_+|$")
axes[0].set_xticks([])
axes[1].set_xlabel(r"$\mu_c - \mu$")
axes[1].set_ylabel(r"$\epsilon_{\mathrm{approx.}}$ (%)")
# axes[1].set_ylim(0, 100)
axes[1].text(1e-6, 5.2, "B", color="black", fontsize=36)
axes[1].set_xscale("log")

# Fill between min and max error datasets on the bottom panel
cmap = plt.get_cmap("Blues")
ratios = np.array([r["ratio"] for r in results])
ratio_min, ratio_max = ratios.min(), ratios.max()

for r in results:
    mus = r["mus"]
    del_phi_pl = r["del_phi_pl"]
    dpl_exact = r["dpl_exact"]
    ratio = r["ratio"]
    norm = (
        (ratio - ratio_min) / (ratio_max - ratio_min) if ratio_max > ratio_min else 0.5
    )
    c = cmap(0.3 + 0.7 * norm)
    axes[0].plot(mus, np.abs(del_phi_pl), "-", color=c, linewidth=2)
    axes[0].plot(mus, np.abs(dpl_exact), "--", color=c, linewidth=2)
    rel_diff = 100 * np.abs(del_phi_pl - dpl_exact) / np.abs(dpl_exact)
    axes[1].plot(mus, rel_diff, "-", color=c, linewidth=2)

sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=plt.Normalize(vmin=ratio_min, vmax=ratio_max)
)
sm.set_array([])

plt.tight_layout()
fig.subplots_adjust(right=0.85)
cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.05)
cbar.set_label(r"$\eta$")

plt.savefig("figures/absorption.png", dpi=300)
