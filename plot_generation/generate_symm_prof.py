import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plot_params
import torch
from scipy.optimize import root_scalar
from surfacetension.solvers import ComputationBox, GradientExplicit
from tqdm import tqdm

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
            for key in grp.keys():
                entry[key] = grp[key][:]
            results.append(entry)
    return results


def compute_all_results():
    cache_file = "chapter_two_data/symmetric_extended.h5"

    # Try loading from cache
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        results = []
        with h5py.File(cache_file, "r") as f:
            for uid in f.keys():
                grp = f[uid]
                results.append(
                    {
                        "chi": float(grp.attrs["chi"]),
                        "eta": float(grp.attrs["eta"]),
                        "phis": grp["phis"][:],
                        "mus": grp["mus"][:],
                        "box_size": grp["box_size"][:],
                    }
                )
        return results

    results = []
    N = 512
    num_points = 32
    try:
        for chi in tqdm([2.1, 2.2, 2.3, 2.4, 2.5], desc="chi"):
            etas = np.linspace(0, 1.0, 11)
            for eta in tqdm(etas, desc=f"  eta (chi={chi:.1f})", leave=False):
                uid = np.random.randint(1e9)
                chi_d, chi_o = -2 * eta * chi, chi * (1 - 2 * eta)
                chis = np.array(
                    [
                        [0, eta * chi, eta * chi],
                        [eta * chi, 0, chi],
                        [eta * chi, chi, 0],
                    ]
                )
                stability = (
                    np.linalg.eigvals(np.array([[chi_d, chi_o], [chi_o, chi_d]])) < 0
                )
                if not stability.all():
                    continue

                phi_c = 1 / (chi_o - chi_d)
                phi_s = -1 / (chi_o + chi_d)
                mu_c = chi_d * phi_c + chi_o * phi_c + np.log(phi_c / (1 - 2 * phi_c))
                pb_ops = (
                    np.exp(
                        np.linspace(
                            np.log(1e-4), np.log(0.5 - phi_c - 1e-3), num_points
                        )
                    )
                    + phi_c
                )
                phis = np.zeros((len(pb_ops), 2, N))
                mus = np.zeros((len(pb_ops),))
                box_size = np.zeros((len(pb_ops),))
                pbar = tqdm(pb_ops, desc=f"  pb (eta={eta:.2f})", leave=False)
                for i, pb in enumerate(pbar):
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
                    # phi_pl_approx = pb + dpl * (1 - np.tanh(x * dx / w * 0.7698) ** 2)
                    phi_pl_approx = pb + dpl * (1 - np.tanh(x / w) ** 2)
                    phi1, phi2 = (
                        phi_pl_approx + phi_mn_approx,
                        phi_pl_approx - phi_mn_approx,
                    )
                    phi0 = 1 - phi1 - phi2
                    phi1 = np.r_[phi1, phi1[::-1]]
                    phi0 = np.r_[phi0, phi0[::-1]]
                    x = np.linspace(-12 * w, 12 * w, N)
                    box_size[i] = x[-1] - x[0]
                    box = ComputationBox((len(x),), (x[-1] - x[0],))

                    max_iters = 200000
                    DT = 0.2
                    dt = DT
                    not_converged = True
                    while not_converged:
                        phis_t = torch.from_numpy(np.array([phi0, phi1]))
                        model = GradientExplicit(box, chis, LMD, dt=dt)
                        iters = 0
                        not_converged = False
                        past_phis = None
                        while model.error_max.item() > 1e-8:
                            phis_t = model(phis_t)
                            if iters > max_iters:
                                raise ValueError(
                                    f"Simulation did not converge after {max_iters} iterations at index {i}"
                                )
                            if iters % 10000 == 0:
                                if iters > 0:
                                    is_stagnate = np.isclose(
                                        phis_t.numpy(), past_phis.numpy(), atol=1e-10
                                    ).all()
                                else:
                                    is_stagnate = False

                                past_phis = phis_t.clone()
                                is_oscillate = (
                                    np.abs(
                                        np.diff(np.sign(np.diff(phis_t[1].numpy())))
                                    ).sum()
                                    > 100
                                )
                                if is_oscillate or is_stagnate:
                                    dt /= 2
                                    not_converged = True
                                    break

                            if torch.isnan(phis_t).any():
                                dt /= 2
                                not_converged = True
                                break

                            iters += 1

                    phi_pl_exact = (1 - phis_t[0].numpy()[: N // 2]) / 2
                    displacement = abs(
                        np.mean((phi_pl_approx - pb)) - np.mean((phi_pl_exact - pb))
                    )
                    pbar.set_postfix(d=f"{displacement:.3e}")
                    phi1, phi2 = (
                        phis_t[1].numpy(),
                        1 - phis_t[0].numpy() - phis_t[1].numpy(),
                    )
                    phis[i] = np.array([phi1, phi2])

                results.append(
                    {
                        "uid": int(uid),
                        "chi": chi,
                        "eta": eta,
                        "phis": phis,
                        "mus": mus,
                        "box_size": box_size,
                        "displacement": displacement,
                    }
                )
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Saving {len(results)} completed results to {cache_file}")
    finally:
        # Always save whatever results we have
        if results:
            with h5py.File(cache_file, "w") as f:
                for r in results:
                    grp = f.create_group(str(r["uid"]))
                    grp.attrs["chi"] = r["chi"]
                    grp.attrs["eta"] = r["eta"]
                    grp.create_dataset("phis", data=r["phis"])
                    grp.create_dataset("mus", data=r["mus"])
                    grp.create_dataset("box_size", data=r["box_size"])
            print(f"Cached {len(results)} results to {cache_file}")
    return results


if __name__ == "__main__":
    results = compute_all_results()
