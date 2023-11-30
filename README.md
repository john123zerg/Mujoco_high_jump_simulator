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
    
1. Train
   ```bash
    python train.py Walker2d-v4 SAC -t -p MlpPolicy
    python train.py Walker2d-v4 A2C -t -p MlpPolicy
    python train.py Walker2d-v4 PPO -t -p MlpPolicy
    python train.py Walker2d-v4 TRPO -t -p MlpPolicy
    python train.py Walker2d-v4 TD3 -t -p MlpPolicy
    python train.py Walker2d-v4 DDPG -t -p MlpPolicy
</details>


## How to Test 
<details>
    <summary>Test the model</summary>
    
1. Test
   ```bash
    python train.py Humanoid-v4 SAC -s ./models/Walker2d-v4_A2C_MlpPolicy_50000.zip
</details>



Need to modify the XML files, modify the env files

In the XML files, a red wall has been added.

In the env files, adjusted the reward to jump high and go further.
