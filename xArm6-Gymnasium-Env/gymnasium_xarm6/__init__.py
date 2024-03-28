from gymnasium.envs.registration import registry, register, make, spec

def _merge(a, b):
    a.update(b);
    return a;


for reward_type in ['sparse', 'dense']:
    r_suffix = 'Dense' if reward_type == 'dense' else 'Sparse'
    kwargs = {
        'reward_type': reward_type,
    }
    for distraction in [True, False]:
        if distraction == True:
            suffix = r_suffix + 'Dist' 
        else:
            suffix = r_suffix + 'NoDist' 
        kwargs['distraction'] = distraction
        register(
            id='xArm6Reach{}-v1'.format(suffix),
            entry_point='gymnasium_xarm6.envs:xArm6ReachEnv',
            kwargs=kwargs,
            max_episode_steps=100,
        )