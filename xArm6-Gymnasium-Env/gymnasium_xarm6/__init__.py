from gymnasium.envs.registration import registry, register, make, spec

kwargs = {
    'reward_type': 'sparse',
    'distraction': True,
}
register(
    id='xArm6Reach{}-v1'.format('SparseDist'),
    entry_point='gymnasium_xarm6.envs:xArm6ReachEnv',
    kwargs=kwargs,
    max_episode_steps=100,
)

kwargs = {
    'reward_type': 'sparse',
    'distraction': False,
}
register(
    id='xArm6Reach{}-v1'.format('SparseNoDist'),
    entry_point='gymnasium_xarm6.envs:xArm6ReachEnv',
    kwargs=kwargs,
    max_episode_steps=100,
)

kwargs = {
    'reward_type': 'dense',
    'distraction': True,
}
register(
    id='xArm6Reach{}-v1'.format('DenseDist'),
    entry_point='gymnasium_xarm6.envs:xArm6ReachEnv',
    kwargs=kwargs,
    max_episode_steps=100,
)

kwargs = {
    'reward_type': 'dense',
    'distraction': False,
}
register(
    id='xArm6Reach{}-v1'.format('DenseNoDist'),
    entry_point='gymnasium_xarm6.envs:xArm6ReachEnv',
    kwargs=kwargs,
    max_episode_steps=100,
)

kwargs = {
    'reward_type': 'dense',
    'distraction': True,
    'viewpoint': True
}
register(
    id='xArm6Reach{}-v1'.format('DenseDistView'),
    entry_point='gymnasium_xarm6.envs:xArm6ReachEnv',
    kwargs=kwargs,
    max_episode_steps=100,
)


# for reward_type in ['sparse', 'dense']:
#     r_suffix = 'Dense' if reward_type == 'dense' else 'Sparse'
#     kwargs['reward_type'] = reward_type
    
#     for distraction in [True, False]:
#         d_suffix = 'Dist' if distraction == True else 'NoDist'
#         suffix = r_suffix + d_suffix

#         kwargs['distraction'] = distraction
#         print(kwargs)

# register(
#     id='xArm6Reach{}-v1'.format(suffix),
#     entry_point='gymnasium_xarm6.envs:xArm6ReachEnv',
#     kwargs=kwargs,
#     max_episode_steps=100,
# )