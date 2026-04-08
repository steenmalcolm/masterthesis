import os

import matplotlib.pyplot as plt
import numpy as np
import plot_params

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L = 12.0
x0 = -0.1  # nonzero interface location
w = 1.5

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
phi2 = 0.5 * (phi2_left + phi2_right) + 0.5 * (phi2_right - phi2_left) * np.tanh(
    (x + x0) / w
)
phi0 = 1 - phi1 - phi2


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(5, 12), constrained_layout=True)

ax = axes[0]
# Species 1 (blue)
(line1,) = ax.plot(x, phi1, linewidth=3.0, label=r"$\phi_1(x)$")

# Species 2 (orange)
(line2,) = ax.plot(x, phi2, linewidth=3.0, label=r"$\phi_2(x)$")

(line0,) = ax.plot(x, phi0, linewidth=3.0, label=r"$\phi_0(x)$")

# Bulk-value labels
x_left_text = -L / 2 + 0.2
x_right_text = 0
ax.text(x_left_text, phi1_left + 0.02, r"$\phi_2$", color=line1.get_color())
ax.text(x_left_text, phi2_left + 0.02, r"$\phi_1$", color=line2.get_color())
ax.text(x_left_text, phi0_left + 0.02, r"$\phi_0$", color=line0.get_color())
ax.text(-L / 2 - 2, 0.57, "A", color="black", fontsize=36)
# ax.text(x_left_text, phi1_left + 0.02, r"$\phi_1^{(1)}$", color=line1.get_color())
# ax.text(x_left_text, phi2_left + 0.02, r"$\phi_0^{(1)}$", color=line2.get_color())
# ax.text(x_left_text, phi0_left + 0.02, r"$\phi_2^{(1)}$", color=line0.get_color())

# ax.text(x_right_text, phi1_right - 0.05, r"$\phi_1^{(2)}$", color=line1.get_color())
# ax.text(x_right_text, phi2_right - 0.05, r"$\phi_0^{(2)}$", color=line2.get_color())
# ax.text(x_right_text, phi0_right - 0.05, r"$\phi_2^{(2)}$", color=line0.get_color())

ax.set_xlim(-L / 2, L / 2)
ax.set_yticks([0.0])
ax.set_yticklabels(["0"])
ax.set_xlabel(r"$x$")
ax.set_xticks([-L / 2, 0, L / 2])
ax.set_xticklabels([r"$-L/2$", r"$0$", r"$L/2$"])

ax = axes[1]

# Hard-interface counterparts
phi1_hard = np.where(x < 0, phi1_left, phi1_right)
n = phi1_hard.shape[0]
dx = x[1] - x[0]
w_n = int(2.2 * w / dx)
if w_n % 2 == 1:
    w_n += 1
phi1_hard[(n - w_n) // 2 : (n + w_n) // 2] = np.linspace(phi1_left, phi1_right, w_n)
phi2_hard = np.where(x < 0, phi2_left, phi2_right)
phi2_hard[(n - w_n) // 2 : (n + w_n) // 2] = np.linspace(phi2_left, phi2_right, w_n)
phi0_hard = 1 - phi1_hard - phi2_hard

ax.plot(x, phi1_hard, linewidth=2.5, color=line1.get_color())
ax.plot(x, phi2_hard, linewidth=2.5, color=line2.get_color())
ax.plot(x, phi0_hard, linewidth=2.5, color=line0.get_color())
ax.axvline(x[(n - w_n) // 2], color="black", linestyle="--")
ax.axvline(x[(n + w_n) // 2], color="black", linestyle="--")
ax.text(x_left_text, phi1_left + 0.02, r"$\phi_2$", color=line1.get_color())
ax.text(x_left_text, phi2_left + 0.02, r"$\phi_1$", color=line2.get_color())
ax.text(x_left_text, phi0_left + 0.02, r"$\phi_0$", color=line0.get_color())

# ax.axvline(0, color="0.3", linestyle=":", linewidth=2.5)


ax.set_xlim(-L / 2, L / 2)
ax.set_yticks([0.0])
ax.set_yticklabels([""])
ax.set_xlabel(r"$x$")
# ax.text(-L / 2 - 2, phi1_left + 0.03, "B", color="black")
ax.text(-L / 2 - 2, 0.57, "B", color="black", fontsize=36)

ax.set_xticks([-L / 2, x[(n - w_n) // 2], x[(n + w_n) // 2], L / 2])
ax.set_xticklabels([r"$-L/2$", r"$-w/2$", r"$w/2$", r"$L/2$"])
# os.makedirs("figures", exist_ok=True)
fig.savefig("figures/approximation_intuition.png", dpi=300)
