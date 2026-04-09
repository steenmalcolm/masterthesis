import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plot_params
import torch
from scipy.optimize import root_scalar

np.random.seed(42)
LMD = 1.25


def del_phi_implicit(dp, pb, pc):
    return dp - pb * np.tanh(dp / pc)


def del_phi_pl_implicit(dpl, dp, pb, pc, ps):
    exp_factor = np.exp(dpl / ps - dp / pc)
    return dpl - (1 - 2 * pb) / 2 * (1 - 1 / (1 - 2 * pb + 2 * (pb + dp) * exp_factor))


def compute_all_results():
    results = []
    for fn in sorted(os.listdir("chapter_two_data")):
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
                if chi_DR != chi_RS:
                    continue
                chis_r = np.array(
                    [
                        [-2 * chi_DR, chi_DS - chi_DR - chi_RS],
                        [chi_DS - chi_DR - chi_RS, -2 * chi_RS],
                    ],
                )
                chi_d, chi_o = chis_r[0, 0], chis_r[0, 1]

                bs = data["box_size"][:]
                phis = data["profile_eq"][:]
                zeta = data.attrs.get("zeta", 0)
                if zeta != 0:
                    raise ValueError(f"Expected zeta=0, but found zeta={zeta} in {fn}")

                N = phis.shape[-1]
                phis_torch = torch.tensor(phis)
                del_phi = np.zeros(phis.shape[0])
                dp_exact = np.zeros(phis.shape[0])
                del_phi_pl = np.zeros(phis.shape[0])
                dpl_exact = np.zeros(phis.shape[0])
                mus = np.zeros(phis.shape[0])
                phi_c = 1 / (chi_o - chi_d)
                phi_s = -1 / (chi_o + chi_d)

                mu_c = chi_d * phi_c + chi_o * phi_c + np.log(phi_c / (1 - 2 * phi_c))
                for i in range(phis.shape[0]):
                    dp = None
                    dpl = None
                    phi1, phi0 = phis[i, :, : N // 2]
                    phi2 = 1 - phi1 - phi0
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
                            f"del_phi root finding did not converge for {fn} at index {i}"
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
                            f"del_phi_pl root finding did not converge for {fn} at index {i}"
                        )
                    dp_exact[i] = phi_mn.max()
                    dpl_exact[i] = phi_pl.min() - pb
                results.append(
                    {
                        "fn": fn,
                        "mus": mus,
                        "del_phi": del_phi,
                        "dp_exact": dp_exact,
                        "del_phi_pl": del_phi_pl,
                        "dpl_exact": dpl_exact,
                        "ratio": chi_DR / chi_DS,
                    }
                )

        except ValueError as e:
            print(f"Error processing file {fn}: {e}")
            break
    return results


results = compute_all_results()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].set_xscale("log")
# axes[0].set_ylim(1e-6, 5e-1)
# axes[0].text(4.5e-5, 7e-2, "A", color="black", fontsize=36)
# axes[1].set_xlim(2e-4, 3)
axes[0].set_ylabel(r"$\delta \phi_+$")
axes[0].set_xticks([])
axes[1].set_xlabel(r"$\mu_c - \mu$")
axes[1].set_ylabel(r"$\epsilon_{\mathrm{approx.}}$ (%)")
# axes[1].set_ylim(0, 100)
# axes[1].text(4.5e-5, 90, "B", color="black", fontsize=36)
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
# plt.show()
