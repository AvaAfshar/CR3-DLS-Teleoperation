#!/usr/bin/env python
import numpy as np
class FK(): 

    def __init__(self):
        # Define geometric parameters for computing the forward kinematics. 
        self.dh_params = self.init_dh_params()
        self.joint_offsets = self.init_joint_offsets()

    def init_dh_params(self):
        """
        Initialize dh parameters from all intermediate frames in the form [a, alpha, d]
        """

        dh_params = [[    0.0      ,   np.pi/2    ,    0.1348  ],  #for joint 1
                     [   -0.2740   ,   0.0        ,    0.0     ],  #for joint 2
                     [   -0.2300   ,   0.0        ,    0.0     ],  #for joint 3
                     [    0.0      ,   np.pi/2    ,    0.1283  ],  #for joint 4
                     [    0.0      ,  -np.pi/2    ,    0.1160  ],  #for joint 5
                     [    0.0      ,   0.0        ,    0.1050  ],  #for joint 6
         ] 
                    
        return dh_params

    def init_joint_offsets(self):
        """
        Initialize joint position offsets
        relative to intermediate frames defined using
        DH conventions 
        """

        joint_offsets = np.array([
                                   0.0,     #for joint 1
                                 -np.pi/2,  #for joint 2
                                   0.0,     #for joint 3
                                 -np.pi/2,  #for joint 4
                                   0.0,     #for joint 5
                                   0.0,     #for joint 6
                         ]) 
        return joint_offsets

    def build_dh_transform(self, a, alpha, d, theta):
        """
        Construct transformation matrix T,
        using DH parameters and conventions
        """
        
        T = []
        cos_alpha=np.cos(alpha)
        sin_alpha=np.sin(alpha)
        cos_theta=np.cos(theta)
        sin_theta=np.sin(theta)
        T=np.array([[cos_theta,  -sin_theta*cos_alpha,    sin_theta*sin_alpha,   a*cos_theta  ],
                    [sin_theta,   cos_theta*cos_alpha,   -cos_theta*sin_alpha,   a*sin_theta  ],
                    [   0.0   ,          sin_alpha   ,         cos_alpha     ,        d       ],
                    [   0.0   ,              0.0     ,             0.0       ,        1.0     ]
                    ])
        return T

    def forward(self, q):
        """
        INPUT:
        q - 1x6 vector of joint angles [q0, q1, q2, q3, q4, q5]

        OUTPUTS:
        jointPositions - 6x 3 matrix, where each row corresponds to a rotational joint of the robot
                         Each row contains the [x,y,z] coordinates in the world frame of the respective 
                         joint's center in meters. The base of the robot is located at [0,0,0].

        T0e - a homogeneous transformation matrix,
              representing the end effector frame expressed in the world frame
        """

        jointPositions = []
        T0e = []
    
        jointPositions = np.zeros((6,3)) #7rows for each joint.3 columns for x,y,z
        T=np.eye(4) #base frame
        for i in range(6): #loop over joints
            theta_i=q[i] + self.joint_offsets[i] #get theta for each joint
            a,alpha,d=self.dh_params[i] #read DH parameters for each joint
            T_i=self.build_dh_transform(a,alpha,d,theta_i) #build DH tranform matrix for each joint
            T=T@T_i #kinematics chain
            jointPositions[i,:]=T[:3,3] #store joint positions (only x,y,z)

        T0e=T #end-effector transform
        return jointPositions, T0e

if __name__ == "__main__":
    pass