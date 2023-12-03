import numpy as np

from gymnasium.envs import mujoco


def modify_env(env, wall, wall_size):


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
    modified_code = existing_lines[:step_function_start+step_function_reward] +reward_function(env.lower(),wall,wall_size)+existing_lines[step_function_reward+step_function_start+1:]

    with open(env_py, 'w') as file:
        file.write("".join(modified_code))
def reward_function(env,wall,wall_size):
    match env:
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
    function_code=function_reward_list(wall,wall_size,list_)


#Halfcheetah 0
#Walker2d 0  8 9
#Humanoid 0 22 24
#Ant 0 ,13 15
#Hopper 0 5 6
    return function_code
def delete_reward(env,wall,wall_size):
    
    
    mujoco_folder = mujoco.__file__
   
    env_py=(mujoco_folder[:-11]+env.lower()+'_v4.py')
    print(env_py)
    with open(env_py, 'r') as file:
        existing_lines = file.readlines()
    step_function_start = None
    no_reward_function=0
    for i, line in enumerate(existing_lines):

        if line.strip().startswith(reward_function(env.lower(),wall,wall_size)[0][8:-1]):
            step_function_start = i
            no_reward_function=1
            break
    if no_reward_function==0:
        return
    modified_code = existing_lines[:step_function_start]+['        rewards = forward_reward + healthy_reward\n']+existing_lines[step_function_start+len(reward_function(env.lower(),wall,wall_size)):]

    # 수정된 코드를 파일에 쓰기
    with open(env_py, 'w') as file:
        file.write("".join(modified_code))

def function_reward_list(wall,wall_size,list_):
    if wall==1:
        if len(list_)==1:
            function_code= [f"        height_reward = 100.0 * self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}] > 1.0 else 0.0\n",
    f"        height_reward_2 = 10.0 * self.state_vector()[{list_[0]}] if 0.5 < self.state_vector()[{list_[0]}] < 1.0 else 0.0\n",
    f"        height_reward_3 =-10.0*self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}]<0.5 else 0.0\n",
    f"        height_reward = height_reward+ height_reward_2+height_reward_3\n",
    f"        distance_to_wall = np.sqrt([xy_position_after[0],self.state_vector()[{list_[0]}]],[3,{wall_size*2}])\n",
    "        collision_penalty = -1 / (distance_to_wall + 1e-6) * 0.1\n",
    "        rewards = forward_reward + healthy_reward + height_reward + collision_penalty\n"]
        elif len(list_)==2:
            function_code= [f"        height_reward = 100.0 * self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}] > 1.0 else 0.0\n",
    f"        height_reward_2 = 10.0 * self.state_vector()[{list_[0]}] if 0.5 < self.state_vector()[{list_[0]}] < 1.0 else 0.0\n",
    f"        height_reward_3 =-10.0*self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}]<0.5 else 0.0\n",
    f"        height_reward = height_reward+ height_reward_2+height_reward_3\n",
    f"        z_velocity_reward = 100.0 * self.state_vector()[{list_[1]}] if self.state_vector()[{list_[1]}] > 1.0 else 0.0\n",
    f"        z_velocity_reward_2 = 5.0 * self.state_vector()[{list_[1]}] if 0.5 < self.state_vector()[{list_[1]}] < 1.0 else 0.0\n",
    f"        total_z_velocity_reward = z_velocity_reward + z_velocity_reward_2\n",
    f"        distance_to_wall = np.sqrt([xy_position_after[0],self.state_vector()[{list_[0]}]],[3,{wall_size*2}])\n",
    "        collision_penalty = -1 / (distance_to_wall + 1e-6) * 0.1\n",
    "        rewards = forward_reward + healthy_reward + height_reward + total_z_velocity_reward + collision_penalty\n"]
        elif len(list_)==3:
            function_code= [f"        height_reward = 100.0 * self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}] > 1.0 else 0.0\n",
    f"        height_reward_2 = 10.0 * self.state_vector()[{list_[0]}] if 0.5 < self.state_vector()[{list_[0]}] < 1.0 else 0.0\n",
    f"        height_reward_3 =-10.0*self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}]<0.5 else 0.0\n",
    f"        height_reward = height_reward+ height_reward_2+height_reward_3\n",
    f"        x_velocity_reward = 100.0 * self.state_vector()[{list_[1]}] if self.state_vector()[{list_[1]}] > 1.0 else 0.0\n",
    f"        x_velocity_reward_2 = 5.0 * self.state_vector()[{list_[1]}] if 0.5 < self.state_vector()[{list_[1]}] < 1.0 else 0.0\n",
    f"        total_x_velocity_reward = x_velocity_reward + x_velocity_reward_2\n",
    f"        z_velocity_reward = 100.0 * self.state_vector()[{list_[2]}] if self.state_vector()[{list_[2]}] > 1.0 else 0.0\n",
    f"        z_velocity_reward_2 = 5.0 * self.state_vector()[{list_[2]}] if 0.5 < self.state_vector()[{list_[2]}] < 1.0 else 0.0\n",
    f"        total_z_velocity_reward = z_velocity_reward + z_velocity_reward_2\n",
    f"        distance_to_wall = np.sqrt([xy_position_after[0],self.state_vector()[{list_[0]}]] - [3,{wall_size*2}])\n",
    "        collision_penalty = -1 / (distance_to_wall + 1e-6) * 0.1\n",
    "        rewards = forward_reward + healthy_reward + height_reward +total_x_velocity_reward+ total_z_velocity_reward + collision_penalty\n"]
    else:
        if len(list_)==1:
            function_code= [f"        height_reward = 100.0 * self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}] > 1.0 else 0.0\n",
    f"        height_reward_2 = 10.0 * self.state_vector()[{list_[0]}] if 0.5 < self.state_vector()[{list_[0]}] < 1.0 else 0.0\n",
    f"        height_reward_3 =-10.0*self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}]<0.5 else 0.0\n",
    f"        height_reward = height_reward+ height_reward_2+height_reward_3\n",
    "        rewards = forward_reward + healthy_reward + height_reward\n"]
        elif len(list_)==2:
            function_code= [f"        height_reward = 100.0 * self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}] > 1.0 else 0.0\n",
    f"        height_reward_2 = 10.0 * self.state_vector()[{list_[0]}] if 0.5 < self.state_vector()[{list_[0]}] < 1.0 else 0.0\n",
    f"        height_reward_3 =-10.0*self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}]<0.5 else 0.0\n",
    f"        height_reward = height_reward+ height_reward_2+height_reward_3\n",
    f"        z_velocity_reward = 100.0 * self.state_vector()[{list_[1]}] if self.state_vector()[{list_[1]}] > 1.0 else 0.0\n",
    f"        z_velocity_reward_2 = 5.0 * self.state_vector()[{list_[1]}] if 0.5 < self.state_vector()[{list_[1]}] < 1.0 else 0.0\n",
    f"        total_z_velocity_reward = z_velocity_reward + z_velocity_reward_2\n",
    "        rewards = forward_reward + healthy_reward + height_reward + total_z_velocity_reward\n"]
        elif len(list_)==3:
            function_code= [f"        height_reward = 100.0 * self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}] > 1.0 else 0.0\n",
    f"        height_reward_2 = 10.0 * self.state_vector()[{list_[0]}] if 0.5 < self.state_vector()[{list_[0]}] < 1.0 else 0.0\n",
    f"        height_reward_3 =-10.0*self.state_vector()[{list_[0]}] if self.state_vector()[{list_[0]}]<0.5 else 0.0\n",
    f"        height_reward = height_reward+ height_reward_2+height_reward_3\n",
    f"        x_velocity_reward = 100.0 * self.state_vector()[{list_[1]}] if self.state_vector()[{list_[1]}] > 1.0 else 0.0\n",
    f"        x_velocity_reward_2 = 5.0 * self.state_vector()[{list_[1]}] if 0.5 < self.state_vector()[{list_[1]}] < 1.0 else 0.0\n",
    f"        total_x_velocity_reward = x_velocity_reward + x_velocity_reward_2\n",
    f"        z_velocity_reward = 100.0 * self.state_vector()[{list_[2]}] if self.state_vector()[{list_[2]}] > 1.0 else 0.0\n",
    f"        z_velocity_reward_2 = 5.0 * self.state_vector()[{list_[2]}] if 0.5 < self.state_vector()[{list_[2]}] < 1.0 else 0.0\n",
    f"        total_z_velocity_reward = z_velocity_reward + z_velocity_reward_2\n",
    "        rewards = forward_reward + healthy_reward + height_reward +total_x_velocity_reward+ total_z_velocity_reward\n"]
    return function_code
    

#Hopper 0 5 6
#Ant 0 ,13 15
#Halfcheetah 0
#Walker2d 0  8 9
#Humanoid 0 22 24



