import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C,PPO,DDPG
import os
import argparse
import re
import numpy as np
from sb3_contrib import TRPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from gymnasium import spaces

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
        # 정규 표현식을 사용하여 가장 안쪽에 있는 <와 > 사이의 모든 문자를 찾아 제거
pattern = re.compile(r'<([^<>]*)>')

algo_off_list=['SAC','TD3','DDPG']
algo_on_list=['A2C','PPO','TRPO']



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
def train(env, sb3_algo,policy):
    #policy_='MlpPolicy','CnnPolicy',MultiInputPolicy',
    # The noise objects for DDPG
    print(env)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    print(sb3_algo)
    if sb3_algo in algo_on_list:
        policy_kwargs_dnn = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[32, 32]))
    else:
        policy_kwargs_dnn = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], qf=[32, 32]))




    if policy=='CNN':
        policy_kwargs_=policy_kwargs_CNN
        policy_='CnnPolicy'
    elif policy=='DNN':
        policy_kwargs_=policy_kwargs_dnn
        policy_='MlpPolicy'
    else:
        policy_kwargs_='None'
        policy_='MlpPolicy'
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
    while True:
        iters += 1
        modified_file_name = re.search(pattern, str(env))

        if modified_file_name:
            inner_content = modified_file_name.group(1)[:-2]
            print(inner_content)

            # Create a directory for gymenv if it doesn't exist
            gymenv_dir = os.path.join(model_dir, inner_content)
            os.makedirs(gymenv_dir, exist_ok=True)

            # Create a directory for sb3_algo if it doesn't exist
            sb3_algo_dir = os.path.join(gymenv_dir, sb3_algo)
            os.makedirs(sb3_algo_dir, exist_ok=True)
            if policy!='None':
                sb3_algo_dir = os.path.join(sb3_algo_dir, policy)
                os.makedirs(sb3_algo_dir, exist_ok=True)

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        file_name = f"{sb3_algo_dir}/{inner_content}_{sb3_algo}_{policy}_{TIMESTEPS*iters:010d}"
        model.save(file_name)
def search_file(path_to_model,policy):
#'Ant-A2C'
    
    prefix, suffix = path_to_model.rsplit("-", 1)
    env_folder=prefix+'-/'
    algorithm_folder=suffix+'/'
    policy_folder=policy+'/'
    path_='./models/'+env_folder+algorithm_folder+policy_folder
    files = os.listdir(path_)
    files = [file for file in files if os.path.isfile(os.path.join(path_, file))]
    print(files)
    # 파일을 최신 수정 순으로 정렬
    path_model=sorted(files)
    print(path_model)
    
    final_path_model=path_model[-1]
    print(final_path_model)
    return path_+final_path_model
def test(env, sb3_algo, path_to_model):
    
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


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-p','--policy',default='MlpPolicy')
    args = parser.parse_args()


    if args.train:
        gymenv = gym.make(args.gymenv+'-v4', render_mode='None')
        train(gymenv, args.sb3_algo,args.policy)

    if(args.test):
        test_file=search_file(args.gymenv+'-'+args.sb3_algo,args.policy)
        print(test_file)
        if os.path.isfile(test_file):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, args.sb3_algo, path_to_model=test_file)
        else:
            print(f'{args.test} not found.')
