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
   ```bash
    python train.py Walker2d SAC -t 
    python train.py Walker2d A2C -t 
    python train.py Walker2d PPO -t 
    python train.py Walker2d TRPO -t 
    python train.py Walker2d TD3 -t 
    python train.py Walker2d DDPG -t 
</details>


## How to Test 
<details>
    <summary>Test the model</summary>
    
1. Download the model files from the models folder
   
    https://drive.google.com/file/d/1e43kluy7EnhDWN1LeRMWMxoJAoGZUYT-/view?usp=sharing

2. Test
   ```bash
    python train.py Humanoid SAC -s .
</details>



Need to modify the XML files, modify the env files
