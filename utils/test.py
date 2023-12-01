import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C,PPO,DDPG
from sb3_contrib import TRPO
from gymnasium import spaces

def test_model(env, sb3_algo, path_to_model):
    
    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case 'TRPO':
            model = TRPO.load(path_to_model, env=env)
        case 'DDPG':
            model = DDPG.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _states = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break
