import os

def search_file(path_to_model,policy,critic_size,wall,wall_size,test_wall,reward_function,file_number):
    prefix, suffix = path_to_model.rsplit("-", 1)
    env_folder=prefix+'-/'
    algorithm_folder=suffix+'/'
    policy_folder=policy+'/'
    if policy!='MlpPolicy':
        policy_folder=policy_folder+f'critic_size_{str(critic_size)}/'
    if int(test_wall)==0 and int(wall)==1:
        wall=0
        wall_size=None
    wall_folder=str(wall)+'/'
    wall_size_folder=str(wall_size)+'/'
    reward_function_folder=f'reward_function_{reward_function}/'
    path_='./models/'+env_folder+algorithm_folder+policy_folder+wall_folder+wall_size_folder+reward_function_folder
    print(path_)
    files = os.listdir(path_)
    files = [file for file in files if os.path.isfile(os.path.join(path_, file))]
    print(files)
    path_model=sorted(files)
    print(path_model)
    
    final_path_model=path_model[-1-int(file_number)]
    print(final_path_model)
    return path_+final_path_model
