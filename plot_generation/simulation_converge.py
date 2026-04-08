import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from surfacetension.solvers import ComputationBox, GradientExplicit
from tqdm import tqdm

np.random.seed(42)
chi01, chi02, chi12 = [], [], []
LMD = 1.25
debug_count = 0
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
        rescale = 1
        pbar = tqdm(range(phis.shape[0]))
        for i in pbar:
            bs[i] *= rescale
            box = ComputationBox((N,), (bs[i],))
            model = GradientExplicit(box, chis, lmd=LMD, dt=0.01)
            iterations = 0
            while model.error_max.item() > 1e-8:
                phis_torch[i] = model(phis_torch[i])
                iterations += 1
            pbar.set_postfix(iterations=iterations)
            mu = model.chemical_potential(phis_torch[i])
            rescale *= (
                np.gradient(phis[i, 0]).max()
                / np.gradient(phis_torch[i, 0].numpy()).max()
            )
        phis = phis_torch.numpy()

    with h5py.File(os.path.join("chapter_two_data", fn), "r+") as f:
        data = f[key]
        data["profile_eq"][...] = phis
        data["box_size"][...] = bs
