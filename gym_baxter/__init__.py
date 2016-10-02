from gym.envs.registration import register

register(
    id='BaxterReacher-v0',
    entry_point='gym_baxter.envs:BaxterReacherEnv',
    timestep_limit=10000,
)
register(
    id='BaxterAvoider-v0',
    entry_point='gym_baxter.envs:BaxterAvoiderEnv',
    timestep_limit=10000,
)
