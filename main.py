import gymnasium as gym
import os
import argparse
from utils import parse_pattern,train,test,enable_xml_wall,override_env_2
import time

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
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
    parser.add_argument('-cs','--critic_size',default=32)
    parser.add_argument('-w','--wall',default=0)
    parser.add_argument('-ws','--wall_size',default=None)
    parser.add_argument('-tw','--test_wall',default=0)
    parser.add_argument('-z','--reward_function',default=0)
    parser.add_argument('-tws','--test_wall_size',default=None)
    parser.add_argument('-f','--file_number',default=0)
    parser.add_argument('-r','--record',default=0)
    args = parser.parse_args()
    if args.test_wall!=0 or args.wall!=0:
        enable_xml_wall.modify_xml_file(args.gymenv.lower(),args.wall,args.test_wall_size)
    else:
        enable_xml_wall.modify_xml_file(args.gymenv.lower(),args.wall,args.wall_size)
    print(f'reward_{args.reward_function}')
    if args.reward_function=='1':
        print('modifying environment')
        override_env_2.modify_env(args.gymenv)
        print('override_2')
    else:
        print('deleting environment')
        override_env_2.delete_reward(args.gymenv)
    if args.train:
        gymenv = gym.make(args.gymenv+'-v4', render_mode='None')
        train.train_model(gymenv, args.sb3_algo,args.policy,args.critic_size,args.wall,args.wall_size,args.reward_function)

    if args.test:
        test_file=parse_pattern.search_file(args.gymenv+'-'+args.sb3_algo,args.policy,args.wall,args.wall_size,args.test_wall,args.reward_function,args.file_number)
        if os.path.isfile(test_file):
            gymenv = gym.make(args.gymenv+'-v4', render_mode='human')
            test.test_model(gymenv, args.sb3_algo,args.record, path_to_model=test_file)
        else:
            print(f'{args.test} not found.')
