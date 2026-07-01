from gymnasium.envs.registration import register

register(
    id="XArm6Reach-v0",
    entry_point="xarm_rl.envs.reach_env:XArm6ReachEnv",
    max_episode_steps=200,
)

# Per-camera variants for experimenting with each viewpoint (same env, different
# render camera). The policy is state-based so the camera only affects the
# d3il image channel; swap freely without retraining the expert.
for _suffix, _cam in [("front", "front"), ("obB", "ob_b"), ("obC", "ob_c"),
                      ("obD", "ob_d"), ("topdown", "topdown"), ("topzoom", "topzoom")]:
    register(
        id=f"XArm6Reach-{_suffix}-v0",
        entry_point="xarm_rl.envs.reach_env:XArm6ReachEnv",
        max_episode_steps=200,
        kwargs={"render_camera": _cam},
    )

register(
    id="XArm6Pusher-v0",
    entry_point="xarm_rl.envs.pusher_env:XArm6PusherEnv",
    max_episode_steps=150,
)

register(
    id="XArm6PickPlace-v0",
    entry_point="xarm_rl.envs.pick_place_env:XArm6PickPlaceEnv",
    max_episode_steps=150,
)
