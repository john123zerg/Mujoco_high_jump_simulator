import gymnasium as gym
import os
import argparse
from utils import parse_pattern,train,test,enable_xml_wall


# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
        # 정규 표현식을 사용하여 가장 안쪽에 있는 <와 > 사이의 모든 문자를 찾아 제거

algo_off_list=['SAC','TD3','DDPG']
algo_on_list=['A2C','PPO','TRPO']



if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-p','--policy',default='MlpPolicy')
    parser.add_argument('-w','--wall',default=0)
    parser.add_argument('-ws','--wall_size',default=None)

    args = parser.parse_args()
    enable_xml_wall.modify_xml_file(args.gymenv.lower(),args.wall,args.wall_size)
    print(args.gymenv)
    if args.train:
        gymenv = gym.make(args.gymenv+'-v4', render_mode='None')
        print(f'__{args.policy}')
        train.train_model(gymenv, args.sb3_algo,args.policy)

    if args.test:
        test_file=parse_pattern.search_file(args.gymenv+'-'+args.sb3_algo,args.policy)
        print(test_file)
        if os.path.isfile(test_file):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test.test_model(gymenv, args.sb3_algo, path_to_model=test_file)
        else:
            print(f'{args.test} not found.')
