# Mujoco_high_jump_simulator

Our goal is to simulate a Mujoco high jump since the simulator usually only goes forward.



1. git clone
    ```bash
    git clone https://github.com/john123zerg/RL.git
    conda create -n mujoco python==3.11.0 -y
    conda activate mujoco
    pip install -r requirements.txt
    pip install install patchelf
    
## How to Train 
<details>
    <summary>Train the model</summary>

['Walker2d','Hopper','HalfCheetah','Humanoid','Ant']
['SAC','A2C','PPO','TRPO','DDPG','TD3']
    
1. Train
   if wall 0 -> don't need to write -w
    It will train until 1 million.
   ```bash
    python main.py Walker2d SAC -t -w 1 -ws 0.2 -z 1
    #The parameters -t : train -w : wall existence -ws : wall_size -z : changing_the_reward_function_to_high_jump_reward
</details>


## How to Test 
<details>
    <summary>Test the model</summary>
    
1. Download the model files from the models folder
   
    https://drive.google.com/file/d/1e43kluy7EnhDWN1LeRMWMxoJAoGZUYT-/view?usp=sharing

2. Test
   If you want to test with a wall when you didn't train with a wall,
   ```bash
    python main.py Humanoid SAC -s . -tw 0 -w 1 -ws 0.2 -z 1
    #The parameters -tw : test_wall -> tells the path_parser to find whether a wall_trained model or not
    #For -w and -ws, it's changing the XML so it deletes, creates, or modifies the wall
</details>



We modify the XML files and the env files.
