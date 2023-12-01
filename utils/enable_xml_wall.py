import xml.etree.ElementTree as ET
from gymnasium.envs.mujoco import ant
def modify_xml_file(env, wall, wall_size):
    mujoco_folder = ant.__file__[:-6]
    xml_file_path = mujoco_folder + f'assets/{env}.xml'
    print(xml_file_path)
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    target_geom_element =root.find(".//worldbody")
    
    modify_or_remove_box(root, target_geom_element, wall, wall_size)

    tree.write(xml_file_path)

def modify_or_remove_box(root,target_geom_element, wall, wall_size):
    geom_elements = root.findall(".//geom[@type='box']")
    print('geom_elements')
    print(wall)
    print(len(geom_elements))
    if int(wall) == 1 and len(geom_elements)>0:
        print('modify_box_element')
        # If wall is 1 and box exists, modify box properties
        for geom_element in geom_elements:
            modify_box_element(geom_element, wall_size)
    elif int(wall) == 1 and len(geom_elements)==0:
        print('create_box_element')
        # If wall is 1 and box doesn't exist, create it

        create_box_element(target_geom_element, wall_size)
    elif int(wall) == 0 and len(geom_elements)>0:
        print('remove_box_element')

        remove_box_element(target_geom_element,geom_elements)

def create_box_element(target_geom_element, wall_size):
    # <geom> 태그 생성
    geom_attributes = {
        'type': 'box',
        'size': f'0.1 3 {wall_size}',
        'pos': f'3 0 {wall_size}',
        'rgba': '1 0 0 1'
    }
    
    geom = ET.Element('geom', geom_attributes)
    target_geom_element.append(geom)
   
def modify_box_element(geom_element, wall_size):
    # Modify the existing box element properties
    geom_element.set("size", f"0.1 3 {wall_size}")
    geom_element.set("pos", f"3 0 {wall_size}")

def remove_box_element(target_geom_element,geom_element):
    # Remove the existing box elements
    print(type(geom_element[0]))
    print(geom_element[0].tag, geom_element[0].attrib, geom_element[0].text)

    target_geom_element.remove(geom_element[0])