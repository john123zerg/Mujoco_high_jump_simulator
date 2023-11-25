import gymnasium as gym
import mujoco_py
import numpy as np
from stable_baselines3 import SAC, TD3, A2C,PPO,DDPG
import os
import argparse
import re
import numpy as np
from sb3_contrib import TRPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# 환경 설정
env = gym.make('Hopper-v4', render_mode='human')
# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
# 보상 함수 수정
def custom_reward(state, action):
    # 여기에서 원하는 보상 로직을 구현합니다.
    # 이 예제에서는 위로 높게 점프하고, 떨어지지 않는 것을 장려하는 로직입니다.

    # 1. 에이전트의 높이에 대한 보상
    height_reward = state[1]  # state[1]은 에이전트의 y 위치 (높이)입니다.

    # 2. 에이전트가 높이를 유지하면서 앞으로 나아가는 행동에 대한 추가 보상
    forward_reward = state[0]  # state[0]은 에이전트의 x 위치입니다.

    # 3. 에이전트가 점프 행동을 선택한 경우에 대한 보상
    jump_reward = 10.0 if action[3] > 0.0 else 0.0

    # 4. 위에서 계산한 보상들을 조합하여 최종 보상 계산
    total_reward = height_reward + forward_reward + jump_reward

    return total_reward

# 보상 함수 설정
env.unwrapped.reward_func = custom_reward

def train(env, policy):
    # 알고리즘 선택 (SAC 사용)
    model = SAC(policy, env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        file_name = f"{model_dir}/sac_{policy}_{TIMESTEPS*iters}"
        model.save(file_name)

def test(env, path_to_model):
    # 학습된 모델 불러오기
    model = SAC.load(path_to_model, env=env)

    obs = env.reset()
    total_reward = 0.0
    for _ in range(100000000):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break

    print("Total Reward:", total_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Hopper-v4')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-p', '--policy', default='MlpPolicy')
    args = parser.parse_args()

    if args.train:
        train(env, args.policy)

    if args.test:
        if os.path.isfile(args.test):
            test(env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
