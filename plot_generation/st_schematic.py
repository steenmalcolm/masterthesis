import os

import matplotlib.pyplot as plt
import numpy as np
import plot_params

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L = 12.0
x0 = 0.7  # nonzero interface location
w = 1.0

phi1_left, phi1_right = 0.80, 0.20
phi2_left, phi2_right = 0.10, 0.40

x = np.linspace(-L / 2, L / 2, 1200)

# ------------------------------------------------------------
# Diffuse profiles
# Species 1 decreases, species 2 increases across the interface
# ------------------------------------------------------------
phi1 = 0.5 * (phi1_left + phi1_right) - 0.5 * (phi1_left - phi1_right) * np.tanh(
    (x - x0) / w
)
phi2 = 0.5 * (phi2_left + phi2_right) + 0.5 * (phi2_right - phi2_left) * np.tanh(
    (x + x0) / w
)

# Hard-interface counterparts
phi1_hard = np.where(x < 0, phi1_left, phi1_right)
phi2_hard = np.where(x < 0, phi2_left, phi2_right)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Species 1 (blue)
(line1,) = ax.plot(x, phi1, linewidth=5.0, label=r"$\phi_1(x)$")
ax.plot(x, phi1_hard, linestyle="--", linewidth=2.5, color=line1.get_color())
ax.fill_between(x, phi1, phi1_hard, color=line1.get_color(), alpha=0.25)

# Species 2 (orange)
(line2,) = ax.plot(x, phi2, linewidth=5.0, label=r"$\phi_2(x)$")
ax.plot(x, phi2_hard, linestyle="--", linewidth=2.5, color=line2.get_color())
ax.fill_between(x, phi2, phi2_hard, color=line2.get_color(), alpha=0.25)

# ax.axvline(0, color="0.3", linestyle=":", linewidth=2.5)

# Bulk-value labels
x_left_text = -L / 2 + 0.35
x_right_text = L / 2 - 1.15

ax.text(
    x_left_text,
    phi1_left + 0.04,
    r"$\phi_1^{(1)}$",
    color=line1.get_color(),
    fontsize=28,
)
ax.text(
    x_left_text,
    phi2_left + 0.04,
    r"$\phi_2^{(1)}$",
    color=line2.get_color(),
    fontsize=28,
)

ax.text(
    x_right_text,
    phi1_right + 0.04,
    r"$\phi_1^{(2)}$",
    color=line1.get_color(),
    fontsize=28,
)
ax.text(
    x_right_text,
    phi2_right + 0.04,
    r"$\phi_2^{(2)}$",
    color=line2.get_color(),
    fontsize=28,
)

ax.set_xlim(-L / 2, L / 2)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 1.0])
ax.set_yticklabels(["0", "1"])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\phi_i(x)$")
ax.set_xticks([-L / 2, 0, L / 2])
ax.set_xticklabels([r"$-L/2$", r"$0$", r"$L/2$"])

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/st_schematic.png", dpi=300)
