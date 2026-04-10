# pylint: disable=consider-using-dict-items
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
                }
            )

        worst_instance = None
        worst_error = 0
        worst_data = {}
        for r_sample in results:
            eta = r_sample["eta"]
            chi = r_sample["chi"]
            phis = r_sample["phis"]
            mus = r_sample["mus"]
            N = phis.shape[-1]
            chi_d, chi_o = -2 * eta * chi, chi * (1 - 2 * eta)
            chis = np.array(
                [
                    [0, eta * chi, eta * chi],
                    [eta * chi, 0, chi],
                    [eta * chi, chi, 0],
                ]
            )

            phi_c = 1 / (chi_o - chi_d)
            phi_s = -1 / (chi_o + chi_d)
            mu_c = chi_d * phi_c + chi_o * phi_c + np.log(phi_c / (1 - 2 * phi_c))
            pb_ops = phis[:, :, 0].mean(axis=1)
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

                x = np.linspace(-6 * w, 6 * w, N // 2)
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
                error = abs(dpl - dpl_exact) / abs(dpl_exact)
                if error > worst_error:
                    worst_error = error
                    worst_instance = np.array(
                        [x, phi_mn_approx, phi_pl_approx, phi_mn_exact, phi_pl_exact]
                    )
                    worst_data = r_sample

        print(
            f"Worst approximation error: {worst_error:.2%} for eta={worst_data['eta']:.2f} and chi={worst_data['chi']:.2f}"
        )
        x, phi_mn_approx, phi_pl_approx, phi_mn_exact, phi_pl_exact = worst_instance

        fig, axes = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)

        L = 20
        ax = axes[0]
        (l1,) = ax.plot(
            x,
            phi_mn_approx,
            linewidth=3.0,
            label=r"$\phi_-^{\mathrm{approx}}$",
            color="blue",
        )
        ax.plot(
            x,
            phi_mn_exact,
            "--",
            linewidth=2.0,
            color="red",
            label=r"$\phi_-^{\mathrm{exact}}$",
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel(r"$\phi_-(x)$")
        ax.set_xticks([])
        ax.text(
            -2 * L,
            ax.get_ylim()[1] - 0.05,
            "A",
            color="black",
            fontsize=36,
        )
        ax.set_xticks([-L, 0, L])
        ax.set_xticklabels([r"", r"", r""])
        ax.set_xlim(-L, L)

        ax = axes[1]
        (l2,) = ax.plot(
            x,
            phi_pl_approx,
            linewidth=3.0,
            label=r"$\phi_+^{\mathrm{approx}}$",
            color="blue",
        )
        ax.plot(
            x,
            phi_pl_exact,
            "--",
            linewidth=2.0,
            color="red",
            label=r"$\phi_+^{\mathrm{exact}}$",
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\phi_+(x)$")
        ax.text(
            -2 * L,
            ax.get_ylim()[1] - 0.0017,
            "B",
            color="black",
            fontsize=36,
        )
        ax.set_xticks([-L, 0, L])
        ax.set_xticklabels([r"$L/2$", r"$0$", r"$L/2$"])
        ax.set_xlim(-L, L)

        plt.show()
        # fig.savefig("figures/symm_profile_approx.png", dpi=300)


if __name__ == "__main__":
    results = main()
