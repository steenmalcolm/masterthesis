# pylint: disable=consider-using-dict-items
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plot_params
import torch
from matplotlib import axes
from scipy.optimize import root_scalar
from surfacetension.solvers import ComputationBox, GradientExplicit

np.random.seed(42)
LMD = 1.25


def del_phi_implicit(dp, pb, pc):
    return dp - pb * np.tanh(dp / pc)


def del_phi_pl_implicit(dpl, dp, pb, pc, ps):
    exp_factor = np.exp(dpl / ps - dp / pc)
    return dpl - (1 - 2 * pb) / 2 * (1 - 1 / (1 - 2 * pb + 2 * (pb + dp) * exp_factor))


def get_mu_and_pi(dp, pb, pc, ps):
    mu = -pb / ps - dp / pc + np.log(pb + dp) - np.log(1 - 2 * pb)
    f_bulk = (
        (pb + dp) * np.log(pb + dp)
        + (pb - dp) * np.log(pb - dp)
        + (1 - 2 * pb) * np.log(1 - 2 * pb)
        - pb**2 / ps
        - dp**2 / pc
    )
    pi = mu * 2 * pb - f_bulk
    return mu, pi


def load_symmetric(path="chapter_two_data/symmetric_extended.h5"):
    results = []
    with h5py.File(path, "r") as f:
        for uid in f.keys():
            grp = f[uid]
            entry = {attr: float(grp.attrs[attr]) for attr in grp.attrs}
            entry["uid"] = uid
            for key, dataset in grp.items():
                entry[key] = dataset[:]
            results.append(entry)
    return results


def main():
    fn = "chapter_two_data/symmetric_extended.h5"

    # Try loading from cache
    if not os.path.exists(fn):
        raise FileNotFoundError(
            f"Cache file {fn} not found. Please run generate_symm_prof.py to generate the data."
        )

    results = []
    with h5py.File(fn, "r") as f:
        for uid, grp in f.items():
            results.append(
                {
                    "chi": float(grp.attrs["chi"]),
                    "eta": float(grp.attrs["eta"]),
                    "phis": grp["phis"][:],
                    "mus": grp["mus"][:],
                    "box_size": grp["box_size"][:],
                }
            )

        worst_instance = None
        worst_error = 0
        worst_data = {}
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        cmap = plt.get_cmap("Blues")
        ratios = np.array([r["eta"] for r in results])
        ratio_min, ratio_max = ratios.min() - 0.3, ratios.max()

        for r_sample in results:
            eta = r_sample["eta"]
            chi = r_sample["chi"]
            box_size = r_sample["box_size"]
            if chi < 2.5:
                continue
            is_plot = abs(eta - 0.9) < 1e-4
            phis = r_sample["phis"]
            mus = r_sample["mus"]
            N = phis.shape[-1]
            chi_d, chi_o = -2 * eta * chi, chi * (1 - 2 * eta)
            chis = np.array(
                [
                    [0, chi, eta * chi],
                    [chi, 0, eta * chi],
                    [eta * chi, eta * chi, 0],
                ]
            )

            phi_c = 1 / (chi_o - chi_d)
            phi_s = -1 / (chi_o + chi_d)
            mu_c = chi_d * phi_c + chi_o * phi_c + np.log(phi_c / (1 - 2 * phi_c))
            pb_ops = phis[:, :, 0].mean(axis=1)
            errs, errs_bad = np.zeros_like(pb_ops), np.zeros_like(pb_ops)
            mus = np.zeros_like(pb_ops)
            for i, pb in enumerate(pb_ops):

                dp = None
                dpl = None
                sol = root_scalar(
                    del_phi_implicit,
                    args=(pb, phi_c),
                    bracket=[1e-10, 1 / 2],
                    method="bisect",
                )
                if sol.converged:
                    dp = sol.root

                sol = root_scalar(
                    del_phi_pl_implicit,
                    args=(dp, pb, phi_c, phi_s),
                    bracket=[-0.1, 0],
                    method="bisect",
                )
                if sol.converged:
                    dpl = sol.root
                mu, pi = get_mu_and_pi(dp, pb, phi_c, phi_s)
                mus[i] = mu_c - mu
                f = (
                    2 * (dpl + pb) * np.log(dpl + pb)
                    + (1 - 2 * (dpl + pb)) * np.log(1 - 2 * (dpl + pb))
                    - (dpl + pb) ** 2 / phi_s
                )
                dpg = (phi_c * (f - mu * 2 * (dpl + pb) + pi)) ** 0.5
                w = dp / dpg
                w *= LMD

                bs = box_size[i]
                x = np.linspace(-bs / 4, bs / 4, N // 2)
                dx = x[1] - x[0]
                phi_mn_approx = -dp * np.tanh(x / w)
                phi_pl_approx = pb + dpl * (1 - np.tanh(x / w) ** 2)
                phi1, phi2 = (
                    phi_pl_approx + phi_mn_approx,
                    phi_pl_approx - phi_mn_approx,
                )
                phi0 = 1 - phi1 - phi2

                phi_pl_exact = (phis[i, 0, : N // 2] + phis[i, 1, : N // 2]) / 2
                phi_mn_exact = -(phis[i, 1, : N // 2] - phis[i, 0, : N // 2]) / 2
                dpl_exact = (phis[i, 0, : N // 2] + phis[i, 1, : N // 2]).min() / 2 - pb
                phis_torch_i = torch.tensor(phis[i, :, : N // 2])

                st = (
                    LMD
                    * (
                        np.gradient(phi_mn_exact, dx) ** 2 / phi_c
                        + np.gradient(phi_pl_exact, dx) ** 2 / phi_s
                    ).sum()
                    * dx
                )
                phi1_approx, phi2_approx = (
                    phi_pl_approx + phi_mn_approx,
                    phi_pl_approx - phi_mn_approx,
                )
                phis_torch_i_approx = torch.tensor(
                    np.stack([phi1_approx, phi2_approx], axis=0)
                )
                phi1, phi2 = (
                    phi_pl_approx + phi_mn_approx,
                    phi_pl_approx - phi_mn_approx,
                )
                f = (
                    phi1 * np.log(phi1)
                    + phi2 * np.log(phi2)
                    + phi0 * np.log(phi0)
                    - (phi_pl_approx) ** 2 / phi_s
                    - phi_mn_approx**2 / phi_c
                )
                mu, pi = get_mu_and_pi(dp, pb, phi_c, phi_s)
                f_if_bad = (
                    2 * pb * np.log(pb)
                    + (1 - 2 * pb) * np.log(1 - 2 * pb)
                    - pb**2 / phi_s
                )
                f_if = (
                    2 * (pb + dpl) * np.log(pb + dpl)
                    + (1 - 2 * (pb + dpl)) * np.log(1 - 2 * (pb + dpl))
                    - (pb + dpl) ** 2 / phi_s
                )

                # "Bad" approximation: square-root formula from approximation_compare.py
                # first_term = del_phi @ kappas_r @ del_phi (analytically simplified)
                # second_term = f_if - f_bulk (mu term vanishes by species swap symmetry)
                first_term = 2 * (2 * dp) ** 2 * LMD**2 * phi_c
                second_term = f_if - 2 * mu * (pb + dpl) + pi
                second_term_bad = f_if_bad - 2 * mu * pb + pi
                st_approx = np.sqrt(first_term * second_term)
                st_approx_bad = np.sqrt(first_term * second_term_bad)
                error = (st_approx - st) / st
                error_bad = (st_approx_bad - st) / st
                errs_bad[i] = error_bad
                errs[i] = error
                mus[i] = mu - mu_c
                if error > worst_error:
                    worst_error = error
                    worst_instance = np.array(
                        [x, phi_mn_approx, phi_pl_approx, phi_mn_exact, phi_pl_exact]
                    )
                    worst_data = r_sample

            ax.plot(
                mus,
                errs_bad * 100,
                "--",
                color=cmap((eta - ratio_min) / (ratio_max - ratio_min)),
                linewidth=3.5,
                label=r"$\epsilon(\bar\phi)$" if is_plot else None,
            )
            ax.plot(
                mus,
                errs * 100,
                "-",
                color=cmap((eta - ratio_min) / (ratio_max - ratio_min)),
                linewidth=3.5,
                label=r"$\epsilon(\phi_+(0))$" if is_plot else None,
            )

            is_plot = False

        ax.set_xscale("log")
        ax.set_xlabel(r"$\mu - \mu_c$")
        ax.set_ylabel(r"$\epsilon_{\mathrm{approx.}}$ (%)")
        ax.legend(loc="upper right")

        # # axes[0].set_ylim(1e-6, 5e-1)
        # axes[0].text(1e-6, 1.4e-2, "A", color="black", fontsize=36)
        # # axes[1].set_xlim(2e-4, 3)
        # axes[0].set_ylabel(r"$|\delta \phi_+|$")
        # axes[0].set_xticks([])
        # axes[1].set_xlabel(r"$\mu_c - \mu$")
        # axes[1].set_ylabel(r"$\epsilon_{\mathrm{approx.}}$ (%)")
        # # axes[1].set_ylim(0, 100)
        # axes[1].text(1e-6, 5.2, "B", color="black", fontsize=36)
        # axes[1].set_xscale("log")
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=ratio_min, vmax=ratio_max)
    )
    sm.set_array([])

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar = fig.colorbar(sm, ax=ax, pad=0.05)
    cbar.set_label(r"$\eta$")
    ax.set_ylim(0, None)
    ax.set_yticks([0, 6, 20, 40, 60, 80, 100])
    print(np.mean(errs))
    # ax.set_yscale("log")
    fig.savefig("figures/st_symmetric_approx.png", dpi=300)


if __name__ == "__main__":
    results = main()
