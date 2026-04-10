import os

import matplotlib.pyplot as plt
import numpy as np
import plot_params

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L = 12.0
x0 = -0.1  # nonzero interface location
w = 1.0

phi1_left, phi1_right = 0.55, 0.30
phi2_left, phi2_right = 0.30, 0.60
phi0_left, phi0_right = 1 - phi1_left - phi2_left, 1 - phi1_right - phi2_right

x = np.linspace(-L / 2, L / 2, 1200)

# ------------------------------------------------------------
# Diffuse profiles
# Species 1 decreases, species 2 increases across the interface
# ------------------------------------------------------------
phi1 = 0.5 * (phi1_left + phi1_right) - 0.5 * (phi1_left - phi1_right) * np.tanh(
    (x - x0) / w
)
phi1 = np.concatenate((phi1, phi1[::-1]))
phi2 = 0.5 * (phi2_left + phi2_right) + 0.5 * (phi2_right - phi2_left) * np.tanh(
    (x + x0) / w
)
phi2 = np.concatenate((phi2, phi2[::-1]))
phi0 = 1 - phi1 - phi2

# Hard-interface counterparts
phi1_hard = np.where(x < 0, phi1_left, phi1_right)
phi2_hard = np.where(x < 0, phi2_left, phi2_right)
phi1_hard = np.concatenate((phi1_hard, phi1_hard[::-1]))
phi2_hard = np.concatenate((phi2_hard, phi2_hard[::-1]))
phi0_hard = 1 - phi1_hard - phi2_hard
x = np.linspace(-L / 2, L / 2, 2400)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Species 1 (blue)
(line1,) = ax.plot(x, phi1, linewidth=4.5, label=r"$\phi_1(x)$")
ax.plot(x, phi1_hard, linestyle="--", linewidth=3.5, color=line1.get_color())

# Species 2 (orange)
(line2,) = ax.plot(x, phi2, linewidth=4.5, label=r"$\phi_2(x)$")
ax.plot(x, phi2_hard, linestyle="--", linewidth=3.5, color=line2.get_color())

(line0,) = ax.plot(x, phi0, linewidth=4.5, label=r"$\phi_0(x)$")
ax.plot(x, phi0_hard, linestyle="--", linewidth=3.5, color=line0.get_color())

# ax.axvline(0, color="0.3", linestyle=":", linewidth=2.5)

# Bulk-value labels
x_left_text = -L / 2 - 1.4
x_right_text = 0

ax.text(
    x_left_text,
    phi1_left + 0.0,
    r"$\phi_1^{(1)}$",
    color=line1.get_color(),
    fontsize=36,
)
ax.text(
    x_left_text,
    phi2_left + 0.0,
    r"$\phi_0^{(1)}$",
    color=line2.get_color(),
    fontsize=36,
)
ax.text(
    x_left_text,
    phi0_left + 0.0,
    r"$\phi_2^{(1)}$",
    color=line0.get_color(),
    fontsize=36,
)

ax.text(
    x_right_text,
    phi1_right - 0.07,
    r"$\phi_1^{(2)}$",
    color=line1.get_color(),
    fontsize=36,
)
ax.text(
    x_right_text,
    phi2_right - 0.07,
    r"$\phi_0^{(2)}$",
    color=line2.get_color(),
    fontsize=36,
)
ax.text(
    x_right_text,
    phi0_right - 0.07,
    r"$\phi_2^{(2)}$",
    color=line0.get_color(),
    fontsize=36,
)

ax.set_xlim(-L / 2, L / 2)
ax.set_yticks([0.0])
ax.set_yticklabels(["0"])
ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$\phi_i(x)$")
ax.set_xticks([-L / 2, 0, L / 2])
ax.set_xticklabels([r"$-L/2$", r"$0$", r"$L/2$"])
# ax.legend(loc="center left")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/profiles_init.png", dpi=300)
