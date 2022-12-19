from gym.envs.registration import register

register(
    id='molecule-circus',
    entry_point='env_gym:LigEnv',
)