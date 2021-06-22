import gym
from spinup import ddpg_pytorch
from spinup.utils.test_policy import load_policy_and_env, run_policy

if __name__ == '__main__':
    # env_name = 'MountainCarContinuous-v0'
    # ddpg env initial value fluctuate
    # ddpg-1 env inital value and set point fluctuate without output
    # ddpg-2 env inital value and set point fluctuate with output
    # ddpg-3 env inital value and set point fluctuate with output greatly
    env_name = 'gym_level:jacket-v0'
    exp_name = 'ddpg-6'
    seed = 10

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name, 0)

    try:
        ddpg_pytorch(
            lambda : gym.make(env_name),
            gamma=0.99, seed=seed, epochs=50,
            logger_kwargs=logger_kwargs
        )
    except KeyboardInterrupt:
        pass

    path = logger_kwargs['output_dir']
    env, get_action = load_policy_and_env(path, 'last', True)
    run_policy(env, get_action, 50)
