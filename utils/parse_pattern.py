import re
import os

def search_file(path_to_model,policy,wall,wall_size,test_wall,reward_function):
#'Ant-A2C'
    
    prefix, suffix = path_to_model.rsplit("-", 1)
    env_folder=prefix+'-/'
    algorithm_folder=suffix+'/'
    policy_folder=policy+'/'
    if test_wall==0:
        path_='./models/'+env_folder+algorithm_folder+policy_folder
    else:
        wall_folder=str(wall)+'/'
        wall_size_folder=str(wall_size)+'/'
        reward_function_folder=str(f'reward_function_{reward_function}')+'/'
        path_='./models/'+env_folder+algorithm_folder+policy_folder+wall_folder+wall_size_folder+reward_function_folder
    files = os.listdir(path_)
    files = [file for file in files if os.path.isfile(os.path.join(path_, file))]
    print(files)
    # 파일을 최신 수정 순으로 정렬
    path_model=sorted(files)
    print(path_model)
    
    final_path_model=path_model[-1]
    print(final_path_model)
    return path_+final_path_model
