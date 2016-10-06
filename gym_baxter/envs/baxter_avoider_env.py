import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

from .baxter_env import BaxterEnv
import baxter_utils as bu

class BaxterAvoiderEnv(BaxterEnv):

    """
    An environment to test training the Baxter to reach a given location
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, timesteps, sphere, control=bu.POSITION, limbs=bu.BOTH_LIMBS):
        super(BaxterAvoiderEnv, self).__init__(timesteps, sphere, control=control, limbs=limbs)

        # left(x, y, z), right(x, y, z)
        # x is forward, y is left, z is up
        if (self.limbs == bu.BOTH_LIMBS):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0,
                                  np.random.random(), np.random.random(), 1.0])
        elif (self.limbs == bu.LEFT_LIMB):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0])
        elif (self.limbs == bu.RIGHT_LIMB):
            self.goal = np.array([np.random.random(), np.random.random(), 1.0])

        self.sphere = sphere
        self.step_in_sphere = 0
        self.t = 0


    @property
    def action_space(self):
        # this will be the joint space
        return Box(-np.inf, np.inf, (self.joint_space,))


    @property
    def observation_space(self):
        # [baxter joint angles; goal pos]
        return Box(-np.inf, np.inf, (self.joint_space + self.goal.size,))


    def _step(self, action):
        if (self.limbs == bu.BOTH_LIMBS):
            laction, raction = self.get_joint_action_dict(action)
            assert(len(laction)) == len(self.llimb.joint_angles())
            assert(len(raction)) == len(self.rlimb.joint_angles())
            if (self.control == bu.VELOCITY):
                self.llimb.set_joint_velocities(laction)
                self.rlimb.set_joint_velocities(raction)
            elif (self.control == bu.TORQUE):
                self.llimb.set_joint_torques(laction)
                self.rlimb.set_joint_torques(raction)
            elif (self.control == bu.POSITION):
                self.llimb.set_joint_positions(laction)
                self.rlimb.set_joint_positions(raction)
        elif (self.limbs == bu.LEFT_LIMB):
            laction = self.get_joint_action_dict(action)
            assert(len(laction)) == len(self.llimb.joint_angles())
            if (self.control == bu.VELOCITY):
                self.llimb.set_joint_velocities(laction)
            elif (self.control == bu.TORQUE):
                self.llimb.set_joint_torques(laction)
            elif (self.control == bu.POSITION):
                self.llimb.set_joint_positions(laction)
        elif (self.limbs == bu.RIGHT_LIMB):
            raction = self.get_joint_action_dict(action)
            assert(len(raction)) == len(self.rlimb.joint_angles())
            if (self.control == bu.VELOCITY):
                self.rlimb.set_joint_velocities(raction)
            elif (self.control == bu.TORQUE):
                self.rlimb.set_joint_torques(raction)
            elif (self.control == bu.POSITION):
                self.rlimb.set_joint_positions(raction)

        self.control_rate.sleep()
        self.state = np.hstack((self.get_joint_angles(), self.goal))

        # check if in area
        if (self.limbs == bu.BOTH_LIMBS):
            if (bu.limb_in_sphere(self.llimb, self.sphere) or
                    bu.limb_in_sphere(self.rlimb, self.sphere)):
                self.step_in_sphere += 1
        elif (self.limbs == bu.LEFT_LIMB):
            if (bu.limb_in_sphere(self.llimb, self.sphere)):
                self.step_in_sphere += 1
        elif (self.limbs == bu.RIGHT_LIMB):
            if (bu.limb_in_sphere(self.rlimb, self.sphere)):
                self.step_in_sphere += 1


        # only consider reward at end of task
        # if we are finished, the reward is weighted combination of the distance
        # from the goal and the number of steps in the space
        done = True if self.t == self.timesteps-1 else False
        if done:
            dist = np.linalg.norm(self.goal-self.get_endeff_position())
            dist_reward = np.exp(-dist)
            space_reward = np.exp(-float(self.step_in_sphere) / self.timesteps)
            print ("num steps in sphere: ", self.step_in_sphere, dist_reward, space_reward)
            reward = .5*dist_reward+.5*space_reward # if at goal reward = 1, else asymptopes to 0
        else:
            reward = 0.0
        self.t += 1

        return self.state, reward, done, {}


    def _reset(self):
        self.step_in_sphere = 0
        self.t = 0
        if (self.limbs == bu.BOTH_LIMBS):
            self.llimb.move_to_neutral()
            self.rlimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))
        elif (self.limbs == bu.LEFT_LIMB):
            self.llimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))
        if (self.limbs == bu.RIGHT_LIMB):
            self.rlimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))


    def _render(self, mode='human', close=False):
        pass
