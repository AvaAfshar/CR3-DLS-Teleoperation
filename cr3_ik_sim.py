
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from solveFK import FK
from solveIK import IK


class CR3IKSimulation(Node):

    def __init__(self):
        super().__init__('cr3_dls_ik_simulation')

        self.fk = FK()
        self.ik = IK()

        self.joint_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6'
        ]

        self.current_q = None

        # Receive current CR3 joint configuration
        self.joint_state_subscriber = self.create_subscription(
        JointState,
        '/joint_states',
        self.joint_state_callback,
        10
        )

        # Send trajectory commands to the CR3 controller
        self.trajectory_publisher = self.create_publisher(
        JointTrajectory,
        '/cr3_group_controller/joint_trajectory',
        10
        )

        self.get_logger().info(
        'CR3 DLS IK simulation node started. Waiting for joint states...'
        )

    def joint_state_callback(self, msg):

        # Reorder incoming joint states into:
        # joint1, joint2, ..., joint6
        joint_map = dict(zip(msg.name, msg.position))

        if all(name in joint_map for name in self.joint_names):
            self.current_q = np.array(
                [joint_map[name] for name in self.joint_names],
                dtype=float
        )

    def run_test(self):

        # Wait until the current simulated joint configuration is available
        while rclpy.ok() and self.current_q is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_q is None:
            self.get_logger().error('Could not obtain current joint state.')
            return

        print("\nCurrent CR3 joint configuration:")
        print(self.current_q)

        # ==========================================================
        # NEAR-SINGULARITY DLS TEST
        # ==========================================================

        # Singular CR3 configuration identified from SVD analysis
        q_singular = np.array([
            0.11711924,
            2.03598687,
            1.8899544,
            1.56253885,
            0.079499448,
            -1.24144679
        ])

        print("\nSingular starting configuration:")
        print(q_singular)

        # Cartesian pose corresponding to the singular configuration
        _,target_pose = self.fk.forward(q_singular)
        target_pose = target_pose.copy()

        # Request a small 10 mm Cartesian displacement
        target_pose[0, 3] += 0.01

        print("\nTarget Cartesian transformation:")
        print(target_pose)

        # Solve IK starting deliberately from the singular configuration
        q_history, success = self.ik.inverse(
            target_pose,
            q_singular.copy()
        )








        
        if not success or len(q_history) == 0:
            self.get_logger().error('DLS inverse kinematics failed.')
            return

        q_solution = np.array(q_history[-1])

        print("\nDLS IK solution:")
        print(q_solution)

        print("\nNumber of IK iterations:")
        print(len(q_history))

        # Verify achieved Cartesian pose
        _, achieved_pose = self.fk.forward(q_solution)

        position_error = np.linalg.norm(
        target_pose[0:3, 3] - achieved_pose[0:3, 3]
        )

        print("\nPosition error (mm):")
        print(position_error * 1000.0)

        # ----------------------------------------------------------
        # SEND IK RESULT TO CR3 SIMULATION
        # ----------------------------------------------------------
        trajectory = JointTrajectory()

        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = q_solution.tolist()

        # Give the simulated robot 3 seconds to reach the configuration
        point.time_from_start.sec = 3
        point.time_from_start.nanosec = 0

        trajectory.points.append(point)

        self.trajectory_publisher.publish(trajectory)

        self.get_logger().info(
        'DLS IK solution published to CR3 simulation.'
        )


def main(args=None):

    rclpy.init(args=args)

    node = CR3IKSimulation()

    # Allow ROS publishers/subscribers to initialise
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.run_test()

    # Keep the node alive briefly so the trajectory is delivered
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
