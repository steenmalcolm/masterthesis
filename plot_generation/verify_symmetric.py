import matplotlib.pyplot as plt
import numpy as np
import plot_params
from pyphasediagram.diagram import PhaseDiagram
from scipy.optimize import root_scalar


def root(phi2, phi1, chi12):
    return np.log(phi1 / phi2) + chi12 * (phi2 - phi1)


chi = 1.5
chi12 = 2.5
chis = np.array([[0, chi, chi], [chi, 0, chi12], [chi, chi12, 0]])
pd = PhaseDiagram(chis)
pd.build()
phic = 1 / chi12
phi1_exact = np.linspace(0.001, phic - 0.001, 10000)
phi2_exact = np.zeros_like(phi1_exact)
for i, phi1 in enumerate(phi1_exact):
    try:
        sol = root_scalar(root, args=(phi1, chi12), bracket=[phic, 1 - 1e-5])
    except ValueError:
        continue
    phi2_exact[i] = sol.root

phi0_exact = 1 - phi1_exact - phi2_exact
mask = (phi2_exact > 0) & (phi0_exact > 0) & (phi1_exact > 0)
phi1_exact = phi1_exact[mask]
phi2_exact = phi2_exact[mask]
phi0_exact = phi0_exact[mask]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# First subplot: original plot
ax1.plot(phi0_exact, phi1_exact, color="black", linestyle="--")
ax1.plot(phi0_exact, phi2_exact, color="black", linestyle="--")
s = 70

num_points = 10
scattered_phi0 = []
scattered_phi1 = []
scattered_phi2 = []
for section in pd.binodal.binodal_sections:
    for phis in section.phis:
        phi1, phi2 = phis[:, :: phis.shape[-1] // num_points]
        phi0 = 1 - phi1 - phi2
        ax1.scatter(
            phi0,
            phi2,
            marker="o",
            s=s,
            color="orange",
            edgecolor="black",
            zorder=10,
            label=r"$\phi_1^{(2)}$",
        )
        ax1.scatter(
            phi0,
            phi1,
            marker="o",
            s=s,
            color="blue",
            edgecolor="black",
            zorder=10,
            label=r"$\phi_1^{(1)}$",
        )
        # Store for residuals
        scattered_phi0.append(phi0)
        scattered_phi1.append(phi1)
        scattered_phi2.append(phi2)
        break

ax1.legend()

# Second subplot: residuals (using phi0 as x, difference as y)
scattered_phi0 = np.concatenate(scattered_phi0)
scattered_phi1 = np.concatenate(scattered_phi1)
scattered_phi2 = np.concatenate(scattered_phi2)

# Interpolate exact phi1/phi2 at scattered phi0 positions
phi1_exact_interp = np.interp(scattered_phi0, phi0_exact, phi1_exact)
phi2_exact_interp = np.interp(scattered_phi0, phi0_exact, phi2_exact)

residual_phi1 = scattered_phi1 - phi1_exact_interp
residual_phi2 = scattered_phi2 - phi2_exact_interp

ax2.scatter(
    scattered_phi0[:-1],
    residual_phi1[:-1],
    s=s,
    color="blue",
    marker="o",
    zorder=10,
    edgecolor="black",
)
ax2.scatter(
    scattered_phi0[:-1],
    residual_phi2[:-1],
    s=s,
    color="orange",
    marker="o",
    zorder=10,
    edgecolor="black",
)

ax2.plot(scattered_phi0, np.zeros_like(scattered_phi0), color="black", linestyle="--")
ax1.set_ylabel(r"$\phi_1$")
ax2.set_xlabel(r"$\phi_0$")
ax2.set_ylabel(r"$\phi_1 - \phi_{1,\mathrm{exact}}$")
ax1.set_xlim(-1e-2, 1 - 2 * phic + 1e-2)
ax2.set_xlim(-1e-2, 1 - 2 * phic + 1e-2)

plt.tight_layout()
plt.savefig("figures/verify_symmetric.png", dpi=300)
