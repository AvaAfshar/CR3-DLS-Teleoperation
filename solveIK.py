#!/usr/bin/env python
import os
import sys
import numpy as np
from solveFK import FK

class IK:

    # JOINT LIMITS
    lower = np.array([-6.28,-6.28,-2.7039,-6.28,-6.28,-6.28])
    upper = np.array([6.28,6.28,2.7039,6.28,6.28,6.28])

    fk = FK()

    def __init__(self):
        pass

    @staticmethod
    def calcJacobian(q):
        """
        Calculate the Jacobian of the end effector in a given configuration.
        INPUT:
        q - 1 x 6 vector of joint angles [q1, q2, q3, q4, q5, q6]
        OUTPUT:
        J - the Jacobian matrix 
        """

        J = []       
        # YOUR CODE STARTS HERE
        J=np.zeros((6,6)) #create empty jacobian matrix
        jointPositions,T0e=IK.fk.forward(q) #get the T0e from forward kinematic part
        p_e=T0e[0:3,3] #get end-effector position
        T=np.eye(4) #transformation for base frame
        for i in range(6): #loop over seven joints
            p_i=T[0:3,3]#get the origin of the previous frame
            z_i=T[0:3,2] #get rotation axis of the previous frame
            J[0:3,i]=np.cross(z_i,(p_e-p_i)) #linear velocity
            J[3:6,i]=z_i #angular velocity
            a,alpha,d=IK.fk.dh_params[i]
            theta_i=q[i]+IK.fk.joint_offsets[i]
            T_i=IK.fk.build_dh_transform(a,alpha,d,theta_i) #build transformation matrix
            T=T@T_i #update transformation matrix for next joint
        # YOUR CODE ENDS HERE
        return J

    @staticmethod
    def cal_target_transform_vec(target, current):
        """
        Calculate the displacement vector and axis of rotation from 
        the current frame to the target frame

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
                 end effector to world

        current - 4x4 numpy array representing the current transformation from
                  end effector to world

        OUTPUTS:
        translate_vec - a 3-element numpy array containing the target translation vector from
                        the current frame to the target frame, expressed in the world frame

        rotate_vec - a 3-element numpy array containing the target rotation vector from
                     the current frame to the end effector frame
        """

        translate_vec = []
        rotate_vec = []
        # YOUR CODE STARTS HERE
        current_position=current[0:3,3] #get the position from current matrix
        target_position=target[0:3,3] #get the position from target matrix
        translate_vec= target_position- current_position # calculate position displacement

        current_rotation=current[0:3,0:3]#get the rotation from current matrix
        target_rotation=target[0:3,0:3]#get the rotation from target matrix
        displacement_rotation=(target_rotation)@(current_rotation.T) #calculate displacement in rotation
        omega_ss=0.5*(displacement_rotation- displacement_rotation.T) #compute skew symmetric for turning matrix to vector
        rotate_vec=np.array([omega_ss[2,1],
                             omega_ss[0,2],
                             omega_ss[1,0]
                            ])
        ## YOUR CODE ENDS HERE

        return translate_vec, rotate_vec

    def check_joint_constraints(self,q,target):
        """
        Check if the given candidate solution respects the joint limits.

        INPUTS
        q - the given solution (joint angles)

        target - 4x4 numpy array representing the desired transformation from
                 end effector to world

        OUTPUTS:
        success - True if some predefined certain conditions are met. Otherwise False
        """

        success = True

        # YOUR CODE STARTS HERE
        for i in range(6): #loop over joints for checking each joint
            if q[i]<IK.lower[i] or q[i]>IK.upper[i]:
                success=False
                break

     
        # YOUR CODE ENDS HERE

        return success


    @staticmethod
    def solve_ik(q,target):
        """
        Uses the method you prefer to calculate the joint velocity 

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final solution

        target - a 4x4 numpy array containing the target end effector pose

        OUTPUTS:
        dq - a desired joint velocity
        Note: Make sure that it will smoothly decay to zero magnitude as the task is achieved.
        """

        dq = []
        # YOUR CODE STARTS HERE
        jointPosition,T0e=IK.fk.forward(q)
        translate_vec, rotate_vec=IK.cal_target_transform_vec(target,T0e) #get displacements from function
        displacement=np.hstack((translate_vec, rotate_vec)) #put translation and rotation displacement in one vector
        J=IK.calcJacobian(q) #calculate Jacobian
        lambda_dls=0.03 # scale factor
        A=J@J.T+(lambda_dls **2) * np.eye(6)
        dq=J.T @ np.linalg.solve(A,displacement)

      

        return dq

    def inverse(self, target, initial_guess):
        """
        Solve the inverse kinematics of the robot arm

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        initial_guess - 1x6 vector of joint angles [q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with the solution process (has set up for you)

        OUTPUTS:
        q - 1x6 vector of joint angles [q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.

        success - True if IK is successfully solved. Otherwise False
        """

        q = initial_guess
        success = False

        # YOUR CODE STARTS HERE
        q_curr=np.array(initial_guess)
        q=[]
        success=False
        iteration_number=2000 #maximum iterations
        terminating_position=0.005  #termination condition for accuracy of results for position
        terminating_rotation=np.deg2rad(0.5) #termination condition for accuracy of results for angles
        gain=0.05 #small gain for converging algorithm
        for iteration in range(iteration_number):
            jointPosition,T0e=IK.fk.forward(q_curr)
            translate_vec, rotate_vec=IK.cal_target_transform_vec(target,T0e) #get displacements from function
            position_error=np.linalg.norm(translate_vec) #calculate error for position
            rotation_error=np.linalg.norm(rotate_vec) #calculate error for roation
            q.append(q_curr.copy())
            if (position_error< terminating_position )and (rotation_error<terminating_rotation): #check for termination
                success=True
                break

            dq=IK.solve_ik(q_curr,target) #compute dq
            q_curr=q_curr+gain*dq #update joint

            if not self.check_joint_constraints(q_curr,target): #check for joint constraints
                success=False
                break
        
        # YOUR CODE ENDS HERE

        return q, success

if __name__ == "__main__":
    pass
