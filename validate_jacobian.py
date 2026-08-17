import numpy as np

from solveFK import FK
from solveIK import IK


fk = FK()


def numerical_jacobian(q, eps=1e-6):
    """
    Numerically approximate the translational part of the Jacobian
    using finite differences.
    """

    J_num = np.zeros((3, 6))

    _, T0 = fk.forward(q)
    p0 = T0[0:3, 3]

    for i in range(6):
        q_test = q.copy()
        q_test[i] += eps

        _, T1 = fk.forward(q_test)
        p1 = T1[0:3, 3]

        J_num[:, i] = (p1 - p0) / eps

    return J_num


# Test configuration
q = np.array([
0.2,
-0.4,
0.3,
0.5,
-0.2,
0.4
])

J_analytic = IK.calcJacobian(q)

J_numeric = numerical_jacobian(q)

print("\nAnalytical translational Jacobian:")
print(J_analytic[0:3, :])

print("\nNumerical translational Jacobian:")
print(J_numeric)

print("\nDifference:")
print(J_analytic[0:3, :] - J_numeric)

error = np.linalg.norm(
J_analytic[0:3, :] - J_numeric
)

print("\nJacobian error norm:")
print(error)

# --------------------------------------------------
# Validate rotational part of Jacobian
# --------------------------------------------------

eps = 1e-7

# Full analytical Jacobian
J_analytical = IK.calcJacobian(q)

# Numerical rotational Jacobian
Jw_numerical = np.zeros((3, 6))

_, T0 = IK.fk.forward(q)
R0 = T0[0:3, 0:3]

for i in range(6):

    q_eps = q.copy()
    q_eps[i] += eps

    _, T_eps = IK.fk.forward(q_eps)
    R_eps = T_eps[0:3, 0:3]

    # Relative infinitesimal rotation in world frame
    R_dot = (R_eps - R0) / eps
    omega_hat = R_dot @ R0.T

    Jw_numerical[:, i] = np.array([
        omega_hat[2, 1],
        omega_hat[0, 2],
        omega_hat[1, 0]
    ])

print("\nAnalytical rotational Jacobian:")
print(J_analytical[3:6, :])

print("\nNumerical rotational Jacobian:")
print(Jw_numerical)

difference_rot = J_analytical[3:6, :] - Jw_numerical

print("\nRotational difference:")
print(difference_rot)

print("\nRotational Jacobian error norm:")
print(np.linalg.norm(difference_rot))
