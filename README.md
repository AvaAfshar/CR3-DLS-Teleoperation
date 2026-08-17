# CR3-DLS-Teleoperation

Kinematic modelling and Damped Least Squares (DLS) inverse kinematics for teleoperation of the DOBOT CR3 collaborative robot.

## Overview

This repository contains the Python implementation developed as part of an MSc Robotics dissertation project at the University of Birmingham.

The work focuses on the kinematic modelling of the DOBOT CR3 robot and the implementation of a Damped Least Squares inverse kinematics method for teleoperation. The DLS approach is used to provide improved numerical stability when the robot operates near kinematic singularities.

## Implementation

The repository contains the following Python scripts:

- `solveFK.py` — Forward kinematics implementation for the DOBOT CR3.
- `solveIK.py` — Damped Least Squares inverse kinematics implementation.
- `validate_DH.py` — Validation of the adopted Denavit-Hartenberg kinematic model.
- `validate_fk.py` — Numerical validation of the forward kinematics implementation.
- `validate_jacobian.py` — Numerical validation of the analytical Jacobian using finite differences.

## Validation

The kinematic implementation was validated numerically before integration into the teleoperation framework.

Forward kinematics and Jacobian validation scripts are included to allow the numerical results reported in the dissertation to be reproduced.

## Requirements

The implementation uses Python 3 and NumPy.

```bash
pip install numpy
