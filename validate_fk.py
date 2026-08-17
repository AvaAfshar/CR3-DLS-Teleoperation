import numpy as np
from solveFK import FK
# --------------------------------------------------
# Basic rotation matrices
# --------------------------------------------------

def Rx(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def Ry(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def Rz(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


# --------------------------------------------------
# Create a homogeneous transformation matrix
# --------------------------------------------------

def make_transform(R, p):
    T = np.eye(4)

    T[0:3, 0:3] = R
    T[0:3, 3] = p

    return T


# --------------------------------------------------
# URDF origin transformation
#
# URDF uses:
# R = Rz(yaw) Ry(pitch) Rx(roll)
# --------------------------------------------------

def urdf_origin(xyz, rpy):

    x, y, z = xyz
    roll, pitch, yaw = rpy

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    p = np.array([x, y, z])

    return make_transform(R, p)


# --------------------------------------------------
# Forward kinematics directly from CR3 URDF data
# --------------------------------------------------

def urdf_fk(q):

    joint_data = [

        # xyz                   rpy
        ([0, 0, 0.1348],        [0, 0, 0]),

        ([0, 0, 0],             [np.pi/2, np.pi/2, 0]),

        ([-0.2740, 0, 0],       [0, 0, 0]),

        ([-0.2300, 0, 0.1283],  [0, 0, -np.pi/2]),

        ([0, -0.1160, 0],       [np.pi/2, 0, 0]),

        ([0, 0.1050, 0],        [-np.pi/2, 0, 0])
    ]

    T = np.eye(4)

    for i in range(6):

        xyz, rpy = joint_data[i]

        # Fixed URDF joint origin
        T_origin = urdf_origin(xyz, rpy)

        # Revolute joint rotation around local z-axis
        T_joint = make_transform(
            Rz(q[i]),
            np.array([0, 0, 0])
        )

        T = T @ T_origin @ T_joint

    return T


# --------------------------------------------------
# Standard DH transformation
# --------------------------------------------------

def dh_transform(a, d, alpha, theta):

    cth = np.cos(theta)
    sth = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([

        [cth, -sth*ca,  sth*sa, a*cth],

        [sth,  cth*ca, -cth*sa, a*sth],

        [0,         sa,      ca,     d],

        [0,          0,       0,     1]
    ])


# --------------------------------------------------
# Forward kinematics using our DH table
# --------------------------------------------------

def dh_fk(q):

    a = np.array([
        0,
        -0.2740,
        -0.2300,
        0,
        0,
        0
    ])

    d = np.array([
        0.1348,
        0,
        0,
        0.1283,
        0.1160,
        0.1050
    ])

    alpha = np.array([
        np.pi/2,
        0,
        0,
        np.pi/2,
        -np.pi/2,
        0
    ])

    theta_offset = np.array([
        0,
        -np.pi/2,
        0,
        -np.pi/2,
        0,
        0
    ])

    theta = q + theta_offset

    T = np.eye(4)

    for i in range(6):

        A = dh_transform(
            a[i],
            d[i],
            alpha[i],
            theta[i]
        )

        T = T @ A

    return T


# --------------------------------------------------
# Validation
# --------------------------------------------------
fk=FK()
test_configurations = [

    np.array([0, 0, 0, 0, 0, 0]),

    np.array([
        0.2,
        -0.3,
        0.4,
        0.1,
        -0.2,
        0.3
    ]),

    np.array([
        -0.4,
        0.2,
        -0.5,
        0.3,
        0.4,
        -0.2
    ])
]


for number, q in enumerate(test_configurations, start=1):

    T_urdf = urdf_fk(q)

    joint_positions,T_dh=fk.forward(q)

    # Position vectors
    p_urdf = T_urdf[0:3, 3]
    p_dh = T_dh[0:3, 3]

    # Position error
    position_error = np.linalg.norm(p_urdf - p_dh)

    # Overall matrix error
    matrix_error = np.max(np.abs(T_urdf - T_dh))

    print("\n----------------------------------------")
    print("Test configuration:", number)
    print("----------------------------------------")

    print("\nJoint angles:")
    print(q)

    print("\nURDF transformation:")
    print(np.round(T_urdf, 6))

    print("\nDH transformation:")
    print(np.round(T_dh, 6))

    print("\nPosition error:")
    print(position_error, "m")

    print("\nMaximum transformation matrix error:")
    print(matrix_error)


print("\nValidation complete.")