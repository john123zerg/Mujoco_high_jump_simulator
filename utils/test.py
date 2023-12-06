import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C,PPO,DDPG
from sb3_contrib import TRPO
import re
from utils import parse_pattern 
import time
pattern = re.compile(r'<([^<>]*)>')
# We have 6 algorithms and this test.py will find its path for each model
def test_model(env, sb3_algo,record,path_to_model):
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
    modified_file_name = re.search(pattern, str(env))
    # e.g. /~~/~~/~~/Ant-v4 -> Ant
    modified_env = modified_file_name.group(1)[:-3]
    # reset the environment
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    max_height=0
    max_velocity=0
    max_x_vel=0
    num_steps = 0
    match modified_env.lower():
        #Get the states representing z_coordinates,x_velocity,z_velocity
        case 'halfcheetah':
            list_=[0]
        case 'walker2d':
            list_=[0,8,9]
        case 'humanoid':
            list_=[0,22,24]
        case 'ant':
            list_=[0,13,15]
        case 'hopper':
            list_=[0,5,6]
    start_time = time.time()
    elapsed_time = 0
    total_height = 0
    total_velocity = 0
    while True:
        action, _states = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if obs[list_[0]] > max_height:
            max_height = obs[list_[0]]
        if obs[list_[1]] > max_x_vel:
            max_x_vel = obs[list_[1]]
        # Since HalfCheetah doesn't have x velocity or z velocity in the state space
        if modified_env.lower() != 'halfcheetah' and obs[list_[2]] > max_velocity:
            max_velocity = obs[list_[2]]

        # Update variables for calculating averages
        total_height += obs[list_[0]]
        total_velocity += obs[list_[2]]
        num_steps += 1

        # If you added -r 1 in the command line, you will not end the test in 10 seconds
        # If you did nothing in the command line, the test will end in 10 seconds
        if record==0:
            elapsed_time = time.time() - start_time
        else:
            elapsed_time=0
        
        if elapsed_time > 10:
            print(f'Maximum Height: {max_height}')
            print(f'Maximum Velocity: {max_velocity}')
            print(f'Maximum X Velocity: {max_x_vel}')
            print(f'Average Height: {total_height / num_steps}')
            print(f'Average Velocity: {total_velocity / num_steps}')
            print(f"Agent stayed at x position for more than 10 seconds. Terminating.")
            break
        if done:
            extra_steps -= 1

            if extra_steps < 0:
                print(f'Maximum Height: {max_height}')
                print(f'Maximum Velocity: {max_velocity}')
                print(f'Maximum X Velocity: {max_x_vel}')
                print(f'Average Height: {total_height / num_steps}')
                print(f'Average Velocity: {total_velocity / num_steps}')
                break
