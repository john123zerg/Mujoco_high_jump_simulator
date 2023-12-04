# Mujoco_high_jump_simulator

Our goal is to simulate a Mujoco high jump since the Mujoco usually only aims to go forward.


## Environment setup

CPU: Intel i7-13700

GPU: RTX-4090

RAM: 64GB

OS : Ubuntu 22.04.03 LTS

Conda environment : Python 3.11.0

Mujoco model : mujoco210

Train settings
(All of them are v4)
| Humanoid    | Walker2d    | Hopper      | HalfCheetah | Ant         |  
|-------------|-------------|-------------|-------------|-------------|
|![image](https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/fc70570c-2ec5-464c-8606-426fd8fdcbb2)|![image](https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/c0ee6381-8ac0-4d22-94da-0af25f07f350)|![image](https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/96c4388e-d6e5-4d61-b821-6ff59e9089e8)|![image](https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/c5a109ca-cdae-412a-a242-92f865303660)|![image](https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/d4f31a02-61b5-4caa-8076-3760c9df5c7a)|



| Wall X   | Wall 0.4 | 
|----------|----------|
|<img width="220" alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/ab4c8c17-ab73-4373-bf51-b913e38ffae7">|<img width="220" alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/9a8878ff-6bbb-42e7-b933-53122d6e17d7">|

| Normal reward | Jump reward  | 
|---------------|--------------|

| A2C | SAC | DDPG | TD3 | PPO | TRPO |
|-----|-----|------|-----|-----|------|

(For the jump reward, we added z-coordinates, z-velocity, distance_to_wall_pentaly to the total_reward)

To make these operations a whole pipeline, we made codes that would enable editing the environments efficiently.


# 1. Python setups
1. Creating the conda environment
    ```bash
    git clone https://github.com/john123zerg/RL.git
    conda create -n mujoco python==3.11.0 -y
    conda activate mujoco
    pip install -r requirements.txt
    pip install install patchelf
# 2. Make sure you have mujoco
The following platforms are currently supported:

- Linux with Python 3.6+. See [the `Dockerfile`](Dockerfile) for the canonical list of system dependencies.
- OS X with Python 3.6+.

The following platforms are DEPRECATED and unsupported:

- Windows support has been DEPRECATED and removed in [2.0.2.0](https://github.com/openai/mujoco-py/releases/tag/v2.0.2.0a1). One known good past version is [1.50.1.68](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/README.md#requirements).
- Python 2 has been DEPRECATED and removed in [1.50.1.0](https://github.com/openai/mujoco-py/releases/tag/1.50.1.0). Python 2 users can stay on the [`0.5` branch](https://github.com/openai/mujoco-py/tree/0.5). The latest release there is [`0.5.7`](https://github.com/openai/mujoco-py/releases/tag/0.5.7) which can be installed with `pip install mujoco-py==0.5.7`.

### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
1. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

If you want to specify a nonstandard location for the package,
use the env variable `MUJOCO_PY_MUJOCO_PATH`.

[reference : https://github.com/openai/mujoco-py/blob/master/README.md?plain=1 ]

    
# 3. How to Train 


['Walker2d','Hopper','HalfCheetah','Humanoid','Ant']
['SAC','A2C','PPO','TRPO','DDPG','TD3']
    
1. Train
   if wall 0 -> don't need to write -w
    It will train until 1 million.
   ```bash
    python main.py Walker2d SAC -t -w 1 -ws 0.2 -z 1
    #The parameters -t : train, -w : wall existence (Bool), -ws : wall_size (Float), -z : changing_the_reward_function_to_high_jump_reward (Bool)


# 4. How to Test 
    
1. Download the model files from the models folder if you want the full files (The final models are already in the repository)
   
    https://drive.google.com/file/d/1e43kluy7EnhDWN1LeRMWMxoJAoGZUYT-/view?usp=sharing

2. Test
   If you want to test with a wall when you didn't train with a wall,
   ```bash
    python main.py Humanoid SAC -s . -tw 0 -w 1 -ws 0.2 -z 1
    #The parameters -tw : test_wall -> tells the path_parser to find whether a wall_trained model or not (Bool)
    #For -w and -ws, it's changing the XML so it deletes, creates, or modifies the wall

# 5. Using the Tensorboard

1. Tensorboard commands (Your train code needs to be running)
   ```bash
    tensorboard --logdir ./logs
   

