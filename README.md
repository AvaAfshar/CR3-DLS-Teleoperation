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
-  `validate_ik.py` — Multi-pose numerical validation of the DLS in inverse kinematics solver.
-  `test_singularity.py` — SVD-based search for a near-singular CR3 configuration for evaluating DLS robustness.
- `cr3_ik_sim.py` — ROS 2 simulation of the CR3 DLS inverse kinematics controller, including Cartesian target generation, IK solution and joint trajectory execution in RViz.

## Validation

The kinematic implementation was validated numerically before integration into the teleoperation framework.

DH model,Forward kinematics,Jacobian, and inverse kinemtics validation scripts are included to allow the numerical results reported in the dissertation to be reproduced.
The DLS solver was subsequently evaluated in ROS 2/RViz using both a nominal Cartesian displacement test and a near-singular configuration identified through singular value decomposition (SVD). The simulation scripts reproduce the experiments and results presented in the dissertation.

## Requirements

The implementation uses Python 3 and NumPy.

```bash
pip install numpy
