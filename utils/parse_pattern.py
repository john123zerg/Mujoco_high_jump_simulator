import re
import os

def search_file(path_to_model,policy):
#'Ant-A2C'
    
    prefix, suffix = path_to_model.rsplit("-", 1)
    env_folder=prefix+'-/'
    algorithm_folder=suffix+'/'
    policy_folder=policy+'/'
    path_='./models/'+env_folder+algorithm_folder+policy_folder
    files = os.listdir(path_)
    files = [file for file in files if os.path.isfile(os.path.join(path_, file))]
    print(files)
    # 파일을 최신 수정 순으로 정렬
    path_model=sorted(files)
    print(path_model)
    
    final_path_model=path_model[-1]
    print(final_path_model)
    return path_+final_path_model