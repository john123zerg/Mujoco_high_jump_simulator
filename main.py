import gymnasium as gym
import os
import argparse
from utils import parse_pattern,train,test,enable_xml_wall,override_env
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
    parser.add_argument('-t', '--train', action='store_true', help='Train model')
    parser.add_argument('-s', '--test', metavar='path_to_model',help='Test model')
    parser.add_argument('-p','--policy',default='MlpPolicy',help='MlpPolicy,CNN,DNN')
    parser.add_argument('-cs','--critic_size',default=32,help='critic_size is the size of the critic network in the training phase')
    parser.add_argument('-w','--wall',default=0,help='In order to train without a wall, you just do not enter any -w -ws -tw -tws for convenience. However if you test with a wall, you need to enter -tw and -tws. If you train with a wall, you need to enter -w -ws -tws.')
    parser.add_argument('-ws','--wall_size',default=None,help='wall_size is the size of the wall in the training phase -> The name you are using for saving the model.')
    parser.add_argument('-z','--reward_function',default=0,help='0 is the default reward function, 1 is the modified reward function')
    parser.add_argument('-tw','--test_wall',default=0,help='If you have tw==0 and have w==1, it means you trained a wall but you want to test without a wall.')
    parser.add_argument('-tws','--transform_wall_size',default=None,help='transform wall size -> This will change the wall existence, size.')
    parser.add_argument('-f','--file_number',default=0,help='0 is the latest==most trained model, 1 is the second latest model, etc.')
    parser.add_argument('-r','--replay',default=0,help='If -r==0, The test end in 10 seconds, if not, the test ends when the terminate condition is met.')
    args = parser.parse_args()
    if args.test_wall!=0 or args.wall!=0:
        enable_xml_wall.modify_xml_file(args.gymenv.lower(),args.wall,args.transform_wall_size)
    else:
        enable_xml_wall.modify_xml_file(args.gymenv.lower(),args.wall,args.wall_size)
    print(f'reward_{args.reward_function}')
    if args.reward_function=='1':
        print('modifying environment')
        override_env.modify_env(args.gymenv)
        print('override environment')
    else:
        print('deleting environment')
        override_env.delete_reward(args.gymenv)
    if args.train:
        gymenv = gym.make(args.gymenv+'-v4', render_mode='None')
        train.train_model(gymenv, args.sb3_algo,args.policy,args.critic_size,args.wall,args.wall_size,args.reward_function)

    if args.test:
        test_file=parse_pattern.search_file(args.gymenv+'-'+args.sb3_algo,args.policy,args.critic_size,args.wall,args.wall_size,args.test_wall,args.reward_function,args.file_number)
        if os.path.isfile(test_file):
            gymenv = gym.make(args.gymenv+'-v4', render_mode='human')
            test.test_model(gymenv, args.sb3_algo,args.replay, path_to_model=test_file)
        else:
            print(f'{args.test} not found.')
