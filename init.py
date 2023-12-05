import numpy as np

from gymnasium.envs.mujoco import mujoco_rendering


mujoco_folder = mujoco_rendering.__file__
print(f'{mujoco_folder}')
line_to_edit=f'            bottomleft, "Solver iterations", str(self.data.solver_iter + 1)'
edited_line=f'            bottomleft, "Solver iterations", str(self.data.solver_niter + 1)\n'
with open(mujoco_folder, 'r') as file:
        existing_lines = file.readlines()
print(line_to_edit)
step_function_start = None
for i, line in enumerate(existing_lines):
    if line.strip().startswith(line_to_edit[12:-5]):
        step_function_start = i
        break
if step_function_start is not None:
    print(f"Found 'step' function in {mujoco_folder} starting at line {step_function_start + 1}.")
else:
    print("Could not find 'step' function in the code.")
modified_code = existing_lines[:step_function_start] +[edited_line]+existing_lines[step_function_start+1:]
with open(mujoco_folder, 'w') as file:
    file.write("".join(modified_code))
