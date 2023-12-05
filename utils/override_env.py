import numpy as np

from gymnasium.envs import mujoco

def reward_function(env):
    match env:
        case 'half_cheetah':
            list_=[1,0]
            height_=0.7
        case 'walker2d':
            list_=[1,9,10,0]
            height_=1.25
        case 'humanoid':
            list_=[2,24,26,0]
            height_=1.4
        case 'ant':
            list_=[2,15,17,0]
            height_=0.75
        case 'hopper':
            list_=[1,6,7,0]
            height_=1.25

    function_code=function_reward_list(list_,height_)
    return function_code,height_


def function_reward_list(list_,height_):
    if len(list_)==2:
             
            function_code= [f"        height_reward =  100*(self.state_vector()[{list_[0]}]/{height_}) if self.state_vector()[{list_[0]}]/{height_} > 1.0 else 0.0 \n",
    f"        height_reward_2 = 10*(self.state_vector()[{list_[0]}]/{height_}) if self.state_vector()[{list_[0]}]/{height_} > 0.5 and self.state_vector()[{list_[0]}]/{height_} < 1.0 else 0.0\n",
    f"        height_reward_3 = -10*abs((self.state_vector()[{list_[0]}]/{height_})) if self.state_vector()[{list_[0]}]/{height_} < 0.5 else 0.0\n",                 
    f"        x_reward = 100 * self.state_vector()[{list_[-1]}] if self.state_vector()[{list_[-1]}] > 1.0 else 0.0\n",                       
    f"        x_reward_2 = 10 * self.state_vector()[{list_[-1]}] if self.state_vector()[{list_[-1]}] > 0.5 and self.state_vector()[{list_[3]}] < 1.0 else 0.0\n",
    f"        x_reward_3 = -10 * self.state_vector()[{list_[-1]}] if self.state_vector()[{list_[-1]}] < 0.5 else 0.0\n",
    f"        total_reward=height_reward + height_reward_2 + height_reward_3 + x_reward + x_reward_2 + x_reward_3\n",
    "        reward = forward_reward  + total_reward - ctrl_cost\n"]
    elif len(list_)==4:
            function_code= [f"        height_reward =  100*(self.state_vector()[{list_[0]}]/{height_}) if self.state_vector()[{list_[0]}]/{height_} > 1.0 else 0.0 \n",
    f"        height_reward_2 = 10*(self.state_vector()[{list_[0]}]/{height_}) if self.state_vector()[{list_[0]}]/{height_} > 0.5 and self.state_vector()[{list_[0]}]/{height_} < 1.0 else 0.0\n",
    f"        height_reward_3 = -10*abs((self.state_vector()[{list_[0]}]/{height_})) if self.state_vector()[{list_[0]}]/{height_} < 0.5 else 0.0\n",                       
    f"        x_reward = 100 * self.state_vector()[{list_[-1]}] if self.state_vector()[{list_[-1]}] > 3.0 else 0.0\n",                       
    f"        x_reward_2 = 10 * self.state_vector()[{list_[-1]}] if self.state_vector()[{list_[-1]}] > 0.5 and self.state_vector()[{list_[3]}] < 3.0 else 0.0\n",
    f"        x_reward_3 = -10 * abs(self.state_vector()[{list_[-1]}]) if self.state_vector()[{list_[-1]}] < 0.5 else 0.0\n",
    f"        x_velocity_reward = 10*self.state_vector()[{list_[1]}] if self.state_vector()[{list_[1]}] > 1.0 else  0.0\n",  
    f"        x_velocity_reward_2 = 5*self.state_vector()[{list_[1]}] if self.state_vector()[{list_[1]}] < 1.0 and self.state_vector()[{list_[1]}] > 0.5 else 0.0\n",
    f"        z_velocity_reward = 10*self.state_vector()[{list_[2]}] if self.state_vector()[{list_[2]}] > 1.0 else 0.0\n", 
    f"        z_velocity_reward_2 = 5*self.state_vector()[{list_[2]}] if self.state_vector()[{list_[2]}] < 1.0 and self.state_vector()[{list_[2]}] > 0.5 else 0.0\n",
    f"        total_reward=height_reward + height_reward_2 + height_reward_3 + x_reward + x_reward_2 + x_reward_3 + x_velocity_reward + x_velocity_reward_2 + z_velocity_reward + z_velocity_reward_2\n",
    "        rewards = forward_reward  + total_reward + healthy_reward\n"]
    return function_code


def modify_env(env):


    if env.lower()=='halfcheetah':
        env='half_cheetah'
    print(env)
    mujoco_folder = mujoco.__file__
    env_py=(mujoco_folder[:-11]+env.lower()+'_v4.py')
    print(env_py)
    with open(env_py, 'r') as file:
        existing_lines = file.readlines()
    
    step_function_start = None
    for i, line in enumerate(existing_lines):
        if line.strip().startswith("def step("):
            step_function_start = i
            break
    for i, line in enumerate(existing_lines[step_function_start:]):
        if line.strip().startswith("height_reward"):
            return
        if line.strip().startswith("reward"):
            step_function_reward = i
            break
    if step_function_start is not None:
        print(f"Found 'step' function in {env_py} starting at line {step_function_start + 1}.")
    else:
        print("Could not find 'step' function in the code.")
    modified_code = existing_lines[:step_function_start+step_function_reward] +reward_function(env.lower())[0]+existing_lines[step_function_reward+step_function_start+1:]

    with open(env_py, 'w') as file:
        file.write("".join(modified_code))
def delete_reward(env):
    
    if env.lower()=='halfcheetah':
        env='half_cheetah'
    mujoco_folder = mujoco.__file__
    print(env.lower())   
    env_py=(mujoco_folder[:-11]+env.lower()+'_v4.py')
    print(env_py)
    with open(env_py, 'r') as file:
        existing_lines = file.readlines()
    step_function_start = None
    no_reward_function=0
    for i, line in enumerate(existing_lines):

        if line.strip().startswith(reward_function(env.lower())[0][0][8:8+5]):
            step_function_start = i
            no_reward_function=1
            break
    if no_reward_function==0:
        return
    print(f'_safsfas_{reward_function(env.lower())[0][0][8:13]}')
    if env.lower()=='half_cheetah':
        modified_code=existing_lines[:step_function_start]+['        reward = forward_reward - ctrl_cost\n']+existing_lines[step_function_start+len(reward_function(env.lower())[0]):]
    else:
        modified_code = existing_lines[:step_function_start]+['        rewards = forward_reward + healthy_reward\n']+existing_lines[step_function_start+len(reward_function(env.lower())[0]):]
    with open(env_py, 'w') as file:
        file.write("".join(modified_code))
