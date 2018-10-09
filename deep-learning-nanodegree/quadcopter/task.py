import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.max_velocity_z = 5.0
        self.min_velocity_z = -5.0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self, steps=0):
        """Uses current pose of sim to return reward."""
        x, y, z = self.sim.pose[0], self.sim.pose[1], self.sim.pose[2]
        vx, vy, vz = self.sim.v
        #reward = 1. - .03 *(abs(self.sim.pose[:3] - self.target_pos)).sum() - 0.03 * (vx +  vy) + 0.05 * min(self.max_velocity_z, vz) - 0.005
        #reward = 1. - .005 *(abs(self.sim.pose[:3] - self.target_pos)).sum() + 0.001 * min(self.max_velocity_z, abs(vz)) + 0.002 * z 
        #reward = 1. - .005 *(abs(z - self.target_pos[2])).sum() + 0.001 * vz 
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        reward = 1. - .005 *(abs(self.sim.pose[:3] - self.target_pos)).sum() + 0.01 * min(self.max_velocity_z, vz) + 0.001 * z 
        return np.tanh(reward)

    def step(self, rotor_speeds, steps=0):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(steps) 
            pose_all.append(self.sim.pose)
            if done :
                reward += 1
                #break
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state