import numpy as np
from solveIK import IK

# --------------------------------------------------
# DLS Singularity Robustness Test
# --------------------------------------------------

## ==========================================================
# Search for near-singular CR3 configurations
# ==========================================================

print("\n============================================")
print(" CR3 NEAR-SINGULARITY SEARCH")
print("============================================")

best_q = None
best_sigma_min = np.inf
best_condition = None
best_singular_values = None

# Deterministic random generator for repeatable results
rng = np.random.default_rng(42)

# Search random valid joint configurations
for _ in range(5000):

    q = np.array([
    rng.uniform(-2.5, 2.5),
    rng.uniform(-2.5, 2.5),
    rng.uniform(-2.5, 2.5),
    rng.uniform(-2.5, 2.5),
    rng.uniform(-2.5, 2.5),
    rng.uniform(-2.5, 2.5)
    ])

    J = IK.calcJacobian(q)

    singular_values = np.linalg.svd(
    J,
    compute_uv=False
    )

    sigma_max = np.max(singular_values)
    sigma_min = np.min(singular_values)

    # Ignore configurations that are effectively exactly singular
    if sigma_min < 1e-6:
        continue

    condition_number = sigma_max / sigma_min

    # Find the closest nonsingular configuration to singularity
    if sigma_min < best_sigma_min:
        best_sigma_min = sigma_min
        best_q = q.copy()
        best_condition = condition_number
        best_singular_values = singular_values.copy()


print("\nBest near-singular configuration found:")
print(best_q)

print("\nSingular values:")
print(best_singular_values)

print("\nMinimum singular value:")
print(best_sigma_min)

print("\nCondition number:")
print(best_condition)

print("\n============================================")