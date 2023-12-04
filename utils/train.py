from sb3_contrib import TRPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC, TD3, A2C,PPO,DDPG
import numpy as np
import torch
import os
import torch.nn as nn
import re
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.envs.mujoco import ant_v4, half_cheetah_v4, humanoid_v4, hopper_v4, walker2d_v4
algo_off_list=['SAC','TD3','DDPG']
algo_on_list=['A2C','PPO','TRPO']
pattern = re.compile(r'<([^<>]*)>')
# Create directories to hold models and logs
model_dir = "./models"

os.makedirs(model_dir, exist_ok=True)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) ->torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs_CNN = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
def train_model(env, sb3_algo,policy,critic_size,wall,wall_size,reward_function):
    #policy_='MlpPolicy','CnnPolicy',MultiInputPolicy',
    # The noise objects for DDPG
    modified_file_name = re.search(pattern, str(env))
    modified_env = modified_file_name.group(1)[:-2]
    log_dir = f"./logs/{modified_env}/{sb3_algo}/{policy}/{wall}/{wall_size}/reward_function_{reward_function}"
    if critic_size!=32:
        log_dir = f"./logs/{modified_env}/{sb3_algo}/{policy}/{critic_size}/{wall}/{wall_size}/reward_function_{reward_function}"
    os.makedirs(log_dir, exist_ok=True)
    print(env)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    print(sb3_algo)
    if sb3_algo in algo_on_list:
        policy_kwargs_dnn = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[32, 32]))
        if critic_size!=32:
            policy_kwargs_dnn = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[int(critic_size), int(critic_size)]))
    else:
        policy_kwargs_dnn = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], qf=[32, 32]))
        if critic_size!=32:
            policy_kwargs_dnn = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], qf=[int(critic_size), int(critic_size)]))




    if policy=='CNN':
        policy_kwargs_=policy_kwargs_CNN
        policy_='CnnPolicy'
    elif policy=='DNN':
        policy_kwargs_=policy_kwargs_dnn
        policy_='MlpPolicy'
    else:
        policy_kwargs_=None
        policy_='MlpPolicy'
    print(policy_kwargs_)
    match sb3_algo:
        
        case 'SAC':
            model = SAC(policy_, env,policy_kwargs=policy_kwargs_, action_noise=action_noise,verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3(policy_, env,policy_kwargs=policy_kwargs_,action_noise=action_noise, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C(policy_, env,policy_kwargs=policy_kwargs_,verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TRPO':
            model = TRPO(policy_, env,policy_kwargs=policy_kwargs_,verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'PPO':
            model = PPO(policy_, env,policy_kwargs=policy_kwargs_, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'DDPG':
            model = DDPG(policy_, env,policy_kwargs=policy_kwargs_, action_noise=action_noise,verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 25000
    iters = 0
    MAX_TIMESTEPS = 1000000
    total_timesteps = 0
    while total_timesteps < MAX_TIMESTEPS:
            iters += 1
           

            if modified_file_name:
                inner_content = modified_file_name.group(1)[:-2]
                print(inner_content)

                # Create a directory for gymenv if it doesn't exist
                gymenv_dir = os.path.join(model_dir, inner_content)
                os.makedirs(gymenv_dir, exist_ok=True)

                # Create a directory for sb3_algo if it doesn't exist
                sb3_algo_dir = os.path.join(gymenv_dir, sb3_algo)
                os.makedirs(sb3_algo_dir, exist_ok=True)
                if policy != 'None':
                    sb3_algo_dir = os.path.join(sb3_algo_dir, policy)
                    os.makedirs(sb3_algo_dir, exist_ok=True)
                else:
                    sb3_algo_dir = os.path.join(sb3_algo_dir, 'MlpPolicy')
                    os.makedirs(sb3_algo_dir, exist_ok=True)

            # Check if training exceeds the maximum timesteps
            if total_timesteps + TIMESTEPS > MAX_TIMESTEPS:
                remaining_timesteps = MAX_TIMESTEPS - total_timesteps
                model.learn(total_timesteps=remaining_timesteps, reset_num_timesteps=False)
                break
            else:
                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
                total_timesteps += TIMESTEPS

            file_name = f"./{sb3_algo_dir}/{wall}/{wall_size}/reward_function_{reward_function}/{inner_content}_{sb3_algo}_{policy}_{total_timesteps:010d}"
            model.save(file_name)
