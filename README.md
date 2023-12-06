

First, we sincerely appreciate https://github.com/DLR-RM/stable-baselines3 for supplying the base code.

Our goal is to simulate a Mujoco high jump since the Mujoco environment usually only aims to go forward.


## Environment setup

CPU : Intel i7 - 13700

GPU : RTX - 4090 

RAM : 64 GB

OS : Ubuntu 22.04.03 LTS 

Conda environment : Python 3.11.0 

PyTorch version : 2.1.0

MuJoCo version : MuJoCo210

MuJoCo Env version : v4 (Gymnasium not Open AI Gym)

Mujoco model : mujoco210

Base code : Stable baselines3, Stable baselines3-contrib

Train settings
(All of them are v4)
| Humanoid    | Walker2d    | Hopper      | HalfCheetah | Ant         |  
|-------------|-------------|-------------|-------------|-------------|
|<img width="100" height='100' alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/59cb2ede-b1ac-4125-a551-854013d5aba0">|<img width="100" height='100' alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/23f15ebb-3645-44c7-b7c5-8c328e563c58">|<img width="100" height='100' alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/901594cb-545b-40f4-bffc-3729a20f8f0f">|<img width="100" height='100' alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/70c4903a-f966-4129-89a6-5501d8d58008">|<img width="100" height='100' alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/03fe9194-22be-4bf9-a0a8-d0764cb19c03">|




| No Wall  | Wall 0.4 | 
|----------|----------|
|<img width="220" alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/ab4c8c17-ab73-4373-bf51-b913e38ffae7">|<img width="220" alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/9a8878ff-6bbb-42e7-b933-53122d6e17d7">|

| Normal reward | Jump reward  | 
|---------------|--------------|
|<img width="220" alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/77b1a63c-4e75-4fbc-9521-d3841a17fc75">|<img width="220" alt="image" src="https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/136e57b4-0a35-4c66-879f-ed73eefa4e9d">|

| A2C | SAC | DDPG | TD3 | PPO | TRPO |
|-----|-----|------|-----|-----|------|

(For the jump reward, we added weighted z-coordinates and z-velocity to the total_reward)

To make these operations a whole pipeline, we made codes that would enable editing the environments efficiently.


# 1. Python setups
## ※ Beware that our environment is Ubuntu 22.04.03LTS; it may not work in Windows Subsystem for Linux, Virtual Machine, or Mac OSX.
1. Creating the conda environment
    ```bash
    git clone https://github.com/john123zerg/Mujoco_high_jump_simulator.git
    conda create -n mujoco python==3.11.0 -y
    conda activate mujoco
    cd Mujoco_high_jump_simulator
    pip install gymnasium
    pip install sb3_contrib
    pip install gymnasium[mujoco]
    pip install tensorboard
    pip install install patchelf
    python init.py
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
    python main.py Walker2d SAC -t -z 1 -w 1 -ws 0.2 -tw 1 -tws 0.2 -z 1

The parameters 

-t : train 

-z : changing_the_reward_function_to_high_jump_reward (Bool)

-w : wall existence for path (Bool)

-ws : wall_size for path (Float)

-tw : train_wall (Bool) 

-tws : modify the wall size (Float)


# 4. How to Test 
    
1. Download the model files from the models folder if you want the full files (The final models are already in the repository)
   
    https://drive.google.com/file/d/1e43kluy7EnhDWN1LeRMWMxoJAoGZUYT-/view?usp=sharing

2. Test
   If you want to test with a wall when you didn't train with a wall,
   ```bash
    python main.py Humanoid SAC -s . -w 1 -ws 0.2 -tw 1 -tws 0.2 -z 1 -r 1 -f 0
The parameters 

-s : Enables entering test mode

-tw : train_wall -> tells the path_parser whether to find a wall_trained model or not (Bool)

-w -ws -tws it's changing the XML so it deletes, creates, or modifies the wall

-r : replay file - 1 enables the test to last forever if not, it will end after 10 seconds (Bool), -f : file rank number ranking - 0 is default (Int)


# 5. Using the Tensorboard

1. Tensorboard commands (Your train code needs to be running)
   ```bash
    tensorboard --logdir ./logs
2. Episode reward mean results (e.g., Humanoid without walls and with normal rewards)
   ![image](https://github.com/john123zerg/Mujoco_high_jump_simulator/assets/63462803/a791591f-bfdb-4b70-9315-b306e4d4d5aa)

   

