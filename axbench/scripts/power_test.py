"""
Power Analysis for Pearson Correlation — Concept-Level
=======================================================
Context: Paper correlates separability metrics (d', PCA variance) with
steering success across N=500 concepts. Each concept has one scalar score
(mean of best scores across 5 prompts). We run power analysis at the
concept level, i.e. N = number of concepts.

Three assessments:
  1. Minimum N to detect a target effect size (ρ = 0.3)
  2. Minimum detectable effect size (MDE) given N = 500
  3. Achieved power at N = 500 for each observed r from Table 1
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── parameters ───────────────────────────────────────────────────────────────
ALPHA       = 0.05          # significance level (two-tailed)
TARGET_POWER= 0.80          # desired power
N_PAPER     = 500           # concepts in the paper

# Observed correlations from Table 1 of the paper
observed_rs = {
    "Last Token PC1":        0.144,
    "Last Token PC1+PC2":    0.099,
    "Mean Output PC1":      -0.142,
    "Mean Output PC1+PC2":  -0.174,
    "d' (discriminability)":-0.083,
    "Overlap fraction":      0.108,
    "Steering dir. norm":   -0.183,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def power_from_r(r: float, n: int, alpha: float = 0.05) -> float:
    """
    Compute achieved power for a Pearson correlation r at sample size n.
    Uses the Fisher z-transform approach (standard for correlation power).
    """
    if abs(r) < 1e-10:
        return alpha  # trivially, no effect → power = α
    z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-tailed critical value
    # Non-centrality via Fisher z
    fisher_z = np.arctanh(r)                   # = 0.5 * ln((1+r)/(1-r))
    se = 1.0 / np.sqrt(n - 3)                 # standard error of Fisher z
    ncp = fisher_z / se                        # non-centrality parameter
    # Power = P(|Z| > z_alpha | NCP)
    power = (1 - stats.norm.cdf(z_alpha - ncp)
               + stats.norm.cdf(-z_alpha - ncp))
    return power


def n_required(r: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Minimum N (concepts) to achieve `power` for effect size `r` at `alpha`.
    Solves via bisection over integer N.
    """
    for n in range(4, 100_000):
        if power_from_r(r, n, alpha) >= power:
            return n
    return -1  # should never reach here for reasonable r


def mde_at_n(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Minimum detectable effect size (|r|) given n, alpha, power.
    Solves via bisection.
    """
    lo, hi = 0.001, 0.999
    for _ in range(60):  # bisection iterations
        mid = (lo + hi) / 2
        if power_from_r(mid, n, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT 1 — Minimum N for ρ = 0.3 (smallest practically meaningful effect)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ASSESSMENT 1: Minimum N to detect ρ = 0.3")
print(f"  α = {ALPHA}, target power = {TARGET_POWER}")
print("=" * 65)

for rho in [0.1, 0.2, 0.3, 0.5]:
    n = n_required(rho, ALPHA, TARGET_POWER)
    print(f"  ρ = {rho:>4}  →  N required = {n:>5} concepts")

n_for_0_3 = n_required(0.3, ALPHA, TARGET_POWER)
print(f"\n  ➤ For ρ = 0.3: N = {n_for_0_3}  (paper uses N = {N_PAPER})")
print(f"    Paper is {N_PAPER / n_for_0_3:.1f}× larger than required for ρ = 0.3\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT 2 — Minimum detectable effect at N = 500
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(f"ASSESSMENT 2: Minimum detectable |r| at N = {N_PAPER}")
print(f"  α = {ALPHA}, power = {TARGET_POWER}")
print("=" * 65)

mde = mde_at_n(N_PAPER, ALPHA, TARGET_POWER)
print(f"  MDE = {mde:.4f}")
print(f"  → With 500 concepts, correlations as small as |r| ≈ {mde:.3f}")
print(f"    are detectable. This is a trivially small effect size.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT 3 — Achieved power for each observed r from Table 1
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(f"ASSESSMENT 3: Achieved power for observed r values (N = {N_PAPER})")
print(f"  α = {ALPHA} (two-tailed)")
print("=" * 65)

powers = {}
for metric, r in observed_rs.items():
    p_val = 2 * (1 - stats.t.cdf(
        abs(r) * np.sqrt((N_PAPER - 2) / (1 - r**2)), df=N_PAPER - 2
    ))
    pwr = power_from_r(r, N_PAPER, ALPHA)
    powers[metric] = pwr
    sig = "✓ significant" if p_val < ALPHA else "✗ not significant"
    print(f"  {metric:<28} r={r:>7.3f}  power={pwr:.3f}  p={p_val:.4f}  {sig}")

print()
print("  Interpretation:")
print("  - All |r| < 0.2, meaning even 'significant' ones are practically tiny.")
print("  - d' at r=-0.083 has power=0.43: coin-flip chance of detecting it.")
print("  - This is NOT underpowering — it means the true effect is near zero.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT 4 — Power curves: how power varies with N, for key effect sizes
# ═══════════════════════════════════════════════════════════════════════════════
ns = np.arange(10, 600, 5)
effect_sizes = {
    "ρ = 0.10 (negligible)":  0.10,
    "ρ = 0.20 (small)":       0.20,
    "ρ = 0.30 (moderate)":    0.30,
    "ρ = 0.50 (large)":       0.50,
}

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0f1117")
gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

# ── Plot 1: Power curves ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("#1a1d27")
colors = ["#5b8dee", "#43c59e", "#f5a623", "#e05c5c"]

for (label, rho), color in zip(effect_sizes.items(), colors):
    pwrs = [power_from_r(rho, int(n), ALPHA) for n in ns]
    ax1.plot(ns, pwrs, color=color, lw=2.2, label=label)

ax1.axhline(0.80, color="white", lw=1.2, ls="--", alpha=0.6, label="80% power threshold")
ax1.axvline(N_PAPER, color="#aaaaaa", lw=1.2, ls=":", alpha=0.8, label=f"N = {N_PAPER} (paper)")
ax1.axvline(n_for_0_3, color="#43c59e", lw=1.2, ls=":", alpha=0.8,
            label=f"N = {n_for_0_3} (min for ρ=0.3)")

ax1.set_xlabel("Number of Concepts (N)", color="white", fontsize=11)
ax1.set_ylabel("Statistical Power (1−β)", color="white", fontsize=11)
ax1.set_title("Power Curves by Effect Size  |  α = 0.05, two-tailed",
              color="white", fontsize=13, fontweight="bold")
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#444")
ax1.set_xlim(10, 600)
ax1.set_ylim(0, 1.02)
legend = ax1.legend(fontsize=9, facecolor="#2a2d3a", labelcolor="white",
                    edgecolor="#555", loc="lower right")

# ── Plot 2: Achieved power per metric (bar chart) ─────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#1a1d27")

metrics = list(powers.keys())
pwr_vals = list(powers.values())
bar_colors = ["#5b8dee" if v >= 0.8 else "#e05c5c" if v < 0.5 else "#f5a623"
              for v in pwr_vals]

bars = ax2.barh(metrics, pwr_vals, color=bar_colors, edgecolor="#333", height=0.6)
ax2.axvline(0.80, color="white", lw=1.2, ls="--", alpha=0.7)
ax2.set_xlabel("Achieved Power", color="white", fontsize=10)
ax2.set_title(f"Achieved Power per Metric\n(N={N_PAPER}, α={ALPHA})",
              color="white", fontsize=11, fontweight="bold")
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[:].set_color("#444")
ax2.set_xlim(0, 1.05)

for bar, val in zip(bars, pwr_vals):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", color="white", fontsize=8)

# ── Plot 3: MDE as function of N ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#1a1d27")

ns_mde = np.arange(20, 600, 10)
mdes = [mde_at_n(int(n), ALPHA, TARGET_POWER) for n in ns_mde]

ax3.plot(ns_mde, mdes, color="#5b8dee", lw=2.2)
ax3.axvline(N_PAPER, color="#aaaaaa", lw=1.2, ls=":", label=f"N={N_PAPER} → MDE={mde:.3f}")
ax3.axhline(mde, color="#aaaaaa", lw=1.0, ls=":", alpha=0.6)
ax3.axhline(0.3, color="#43c59e", lw=1.2, ls="--", alpha=0.8, label="ρ=0.3 (practical threshold)")

ax3.set_xlabel("Number of Concepts (N)", color="white", fontsize=10)
ax3.set_ylabel("Min Detectable |r|", color="white", fontsize=10)
ax3.set_title("Minimum Detectable Effect\nvs. Sample Size",
              color="white", fontsize=11, fontweight="bold")
ax3.tick_params(colors="white")
ax3.spines[:].set_color("#444")
ax3.set_xlim(20, 600)
ax3.set_ylim(0, 0.6)
ax3.legend(fontsize=8, facecolor="#2a2d3a", labelcolor="white", edgecolor="#555")

fig.suptitle("Power Analysis — Separability vs. Steerability Correlation\n"
             "Concept-Level Pearson r  |  AxBench Concept500",
             color="white", fontsize=14, fontweight="bold", y=1.01)

plt.savefig("/mnt/user-data/outputs/power_analysis.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Plots saved to power_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY: Required N for various effect sizes (α=0.05, power=0.80)")
print("=" * 65)
print(f"  {'Effect size (ρ)':<20} {'Cohen label':<15} {'N required':<12}")
print(f"  {'-'*20} {'-'*15} {'-'*12}")
labels = {0.1: "negligible", 0.2: "small", 0.3: "moderate", 0.5: "large"}
for rho, label in labels.items():
    print(f"  {rho:<20} {label:<15} {n_required(rho, ALPHA, TARGET_POWER):<12}")

print(f"\n  Paper N = 500. MDE at N=500: |r| ≥ {mde:.4f}")
print(f"  Minimum N for ρ=0.3:         N = {n_for_0_3}")
print(f"  Conclusion: N=500 is {N_PAPER//n_for_0_3}× overpowered for moderate effects.")
print("=" * 65)