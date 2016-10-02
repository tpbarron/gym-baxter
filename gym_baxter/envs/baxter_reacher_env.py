import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .baxter_env import BaxterEnv

class BaxterReacherEnv(gym.Env):
    """
    An environment to test training the Baxter to reach a given location
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, timesteps, control=BaxterEnv.POSITION, limbs=BaxterEnv.BOTH_LIMBS,):
        super(BaxterReacherEnv, self).__init__(timesteps, control=control, limbs=limbs)

        # left(x, y, z), right(x, y, z)
        # x is forward, y is left, z is up
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0,
                                  np.random.random(), np.random.random(), 1.0])
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            self.goal = np.array([np.random.random(), -np.random.random(), 1.0])
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            self.goal = np.array([np.random.random(), np.random.random(), 1.0])

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
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            laction, raction = self.get_joint_action_dict(action)
            assert(len(laction)) == len(self.llimb.joint_angles())
            assert(len(raction)) == len(self.rlimb.joint_angles())
            if (self.control == BaxterEnv.VELOCITY):
                self.llimb.set_joint_velocities(laction)
                self.rlimb.set_joint_velocities(raction)
            elif (self.control == BaxterEnv.TORQUE):
                self.llimb.set_joint_torques(laction)
                self.rlimb.set_joint_torques(raction)
            elif (self.control == BaxterEnv.POSITION):
                self.llimb.set_joint_positions(laction)
                self.rlimb.set_joint_positions(raction)
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            laction = self.get_joint_action_dict(action)
            assert(len(laction)) == len(self.llimb.joint_angles())
            if (self.control == BaxterEnv.VELOCITY):
                self.llimb.set_joint_velocities(laction)
            elif (self.control == BaxterEnv.TORQUE):
                self.llimb.set_joint_torques(laction)
            elif (self.control == BaxterEnv.POSITION):
                self.llimb.set_joint_positions(laction)
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            raction = self.get_joint_action_dict(action)
            assert(len(raction)) == len(self.rlimb.joint_angles())
            if (self.control == BaxterEnv.VELOCITY):
                self.rlimb.set_joint_velocities(raction)
            elif (self.control == BaxterEnv.TORQUE):
                self.rlimb.set_joint_torques(raction)
            elif (self.control == BaxterEnv.POSITION):
                self.rlimb.set_joint_positions(raction)

        self.control_rate.sleep()
        self.state = np.hstack((self.get_joint_angles(), self.goal))

        # only consider reward at end of task
        # if we are finished, the reward is the distance from the goal, else 0
        done = True if self.t == self.timesteps-1 else False
        if done:
            dist = np.linalg.norm(self.goal-self.get_endeff_position())
            reward = np.exp(-dist) # if at goal reward = 1, else asymptopes to 0
            #reward = -dist # closer go goal, smaller the value
        else:
            reward = 0.0
        self.t += 1

        return self.state, reward, done, {}


    def _reset(self):
        if (self.limbs == BaxterEnv.BOTH_LIMBS):
            self.llimb.move_to_neutral()
            self.rlimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))
        elif (self.limbs == BaxterEnv.LEFT_LIMB):
            self.llimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))
        elif (self.limbs == BaxterEnv.RIGHT_LIMB):
            self.rlimb.move_to_neutral()
            self.state = self.get_joint_angles()
            return np.hstack((self.state, self.goal))
        self.t = 0

    def _render(self, mode='human', close=False):
        pass
