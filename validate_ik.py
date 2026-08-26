#!/usr/bin/env python

import numpy as np

from solveFK import FK
from solveIK import IK


def rotation_error_deg(R_target, R_actual):
    R_error = R_target @ R_actual.T

    value = (np.trace(R_error) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    angle = np.arccos(value)

    return np.rad2deg(angle)


def validate_single_pose(fk, ik, q_target_deg, q_initial_deg, test_number):

    q_target = np.deg2rad(q_target_deg)
    q_initial = np.deg2rad(q_initial_deg)

    # Generate target pose using validated FK
    _, T_target = fk.forward(q_target)

    # Run IK
    q_history, success = ik.inverse(
    T_target,
    q_initial
    )

    if len(q_history) == 0:
        return {
        "test": test_number,
        "success": False,
        "iterations": 0,
        "position_error_mm": np.nan,
        "orientation_error_deg": np.nan,
        "joint_limits": False,
        "q_solution": None
        }

    q_solution = np.array(q_history[-1])

    # Forward kinematics of IK solution
    _, T_solution = fk.forward(q_solution)

    # Position error
    p_target = T_target[0:3, 3]
    p_solution = T_solution[0:3, 3]

    position_error = np.linalg.norm(
    p_target - p_solution
    )

    # Orientation error
    R_target = T_target[0:3, 0:3]
    R_solution = T_solution[0:3, 0:3]

    orientation_error = rotation_error_deg(
    R_target,
    R_solution
    )

    # Joint-limit check
    joint_limits_valid = ik.check_joint_constraints(
    q_solution,
    T_target
    )

    return {
    "test": test_number,
    "success": success,
    "iterations": len(q_history),
    "position_error_mm": position_error * 1000.0,
    "orientation_error_deg": orientation_error,
    "joint_limits": joint_limits_valid,
    "q_solution": np.rad2deg(q_solution)
    }


def validate_ik():

    fk = FK()
    ik = IK()

    print("\n==============================================================")
    print(" CR3 IK MULTI-POSE VALIDATION")
    print("==============================================================\n")

    # --------------------------------------------------
    # Test configurations
    # --------------------------------------------------

    test_poses = [

        [20.0, -30.0, 40.0, 25.0, -20.0, 30.0],

        [-30.0, -20.0, 30.0, -40.0, 25.0, -20.0],

        [45.0, -45.0, 60.0, 30.0, -30.0, 45.0],

        [-60.0, 30.0, -40.0, 50.0, 20.0, -35.0],

        [90.0, -60.0, 70.0, -45.0, 35.0, 60.0]
    ]

    # Same initial guess for all tests
    initial_guesses = [
        [0.0, -10.0, 10.0, 0.0, 0.0, 0.0],
        [-10.0, -10.0, 10.0, -10.0, 10.0,-10.0],
        [20.0,10.0,-20.0,20.0,10.0,-20.0],
        [-30.0,10.0,-20.0,20.0,10.0,-20.0],
        [50.0,-30.0,30.0,-20.0,20.0,30.0]

    ]

    results = []

    # --------------------------------------------------
    # Run tests
    # --------------------------------------------------

    for i, q_target in enumerate(test_poses, start=1):

        print(f"Running Test {i}...")

        result = validate_single_pose(
        fk,
        ik,
        q_target,
        initial_guesses[i - 1],
        i
        )

        results.append(result)

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n==============================================================")
    print(" RESULTS")
    print("==============================================================")

    print(
        f"{'Test':<6}"
        f"{'Iterations':<12}"
        f"{'Pos Error (mm)':<18}"
        f"{'Ori Error (deg)':<18}"
        f"{'Limits':<10}"
        f"{'Result':<10}"
    )

    print("-" * 74)

    for r in results:

        passed = (
            r["success"]
         and r["joint_limits"]
            and r["position_error_mm"] < 5.0
            and r["orientation_error_deg"] < 0.5
        )

        result_text = "PASS" if passed else "FAIL"

        print(
            f"{r['test']:<6}"
            f"{r['iterations']:<12}"
            f"{r['position_error_mm']:<18.4f}"
            f"{r['orientation_error_deg']:<18.6f}"
            f"{str(r['joint_limits']):<10}"
            f"{result_text:<10}"
        )

    # --------------------------------------------------
    # Detailed joint solutions
    # --------------------------------------------------

    print("\n==============================================================")
    print(" IK JOINT SOLUTIONS")
    print("==============================================================")

    for r in results:

        print(f"\nTest {r['test']}:")

        if r["q_solution"] is not None:
            print(
                np.round(
                    r["q_solution"],
                    4
                )
            )
        else:
            print("No solution")

    # --------------------------------------------------
    # Overall validation
    # --------------------------------------------------

    all_passed = True

    for r in results:

        passed = (
            r["success"]
            and r["joint_limits"]
            and r["position_error_mm"] < 5.0
            and r["orientation_error_deg"] < 0.5
        )

        if not passed:
            all_passed = False

    print("\n==============================================================")

    if all_passed:
        print("OVERALL IK VALIDATION: PASSED")
    else:
        print("OVERALL IK VALIDATION: SOME TESTS FAILED")

    print("==============================================================\n")


if __name__ == "__main__":
    validate_ik()
