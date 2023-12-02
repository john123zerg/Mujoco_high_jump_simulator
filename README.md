# RL
1. git clone
    ```bash
    git clone https://github.com/john123zerg/RL.git
2. create conda env
    ```bash
    conda create -n mujoco python==3.11.0 -y
3. activate mujoco

    ```bash
    conda activate mujoco
4. install requiremnets.txt

    ```bash
    pip install -r requirements.txt
## How to Train 
<details>
    <summary>Train the model</summary>

['Walker2d','Hopper','HalfCheetah','Humanoid','Ant']
['SAC','A2C','PPO','TRPO','DDPG','TD3']
    
1. Train
   if wall 0 -> don't need to write -w
    It will train until 1 million.
   ```bash
    python main.py Walker2d SAC -t -w 1 -ws 0.2
    #The parameters -t : train -w : wall existence -ws : wall_size
</details>


## How to Test 
<details>
    <summary>Test the model</summary>
    
1. Download the model files from the models folder
   
    https://drive.google.com/file/d/1e43kluy7EnhDWN1LeRMWMxoJAoGZUYT-/view?usp=sharing

2. Test
   If you want to test with a wall when you didn't train with a wall,
   ```bash
    python main.py Humanoid SAC -s . -tw 0 -w 1 -ws 0.2
    #The parameters -tw : test_wall -> tells the path_parser to find whether a wall_trained model or not
    #For -w and -ws, it's changing the xml so it deletes, creates or modifies the wall
</details>



Need to modify the XML files, modify the env files
