from gymnasium.envs.registration import register

register(
    id="XArm6Reach-v0",
    entry_point="xarm_rl.envs.reach_env:XArm6ReachEnv",
    max_episode_steps=100,
)

register(
    id="XArm6PickPlace-v0",
    entry_point="xarm_rl.envs.pick_place_env:XArm6PickPlaceEnv",
    max_episode_steps=150,
)
