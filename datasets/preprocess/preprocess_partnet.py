import os
import json
import argparse
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='mesh/glb_texture')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    assert os.path.exists(input_path), f'{input_path} does not exist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for mesh_name in tqdm(os.listdir(input_path)):
        mesh_path = os.path.join(input_path, mesh_name)
        
        # Use textured GLB for rendering
        textured_glb_path = os.path.join('mesh/glb_texture', mesh_name)
        # Use parts GLB for NPY generation  
        parts_glb_path = os.path.join('mesh/glb_part_voxel', mesh_name)
        
        # 1. Sample points from mesh surface (using glb_part_voxel)
        os.system(f"python datasets/preprocess/mesh_to_point.py --input {parts_glb_path} --output {output_path}")
        # 2. Render images (using glb_texture)
        os.system(f"python datasets/preprocess/render.py --input {textured_glb_path} --output {output_path}")
        # 3. Remove background for rendered images and resize to 90%
        export_mesh_folder = os.path.join(output_path, mesh_name.replace('.glb', ''))
        export_rendering_path = os.path.join(export_mesh_folder, 'rendering.png')
        os.system(f"python datasets/preprocess/rmbg.py --input {export_rendering_path} --output {output_path}")
        # 4. Skip IoU calculation 
        time.sleep(1)
    
    # generate configs
    configs = []
    for mesh_name in tqdm(os.listdir(input_path)):
        mesh_path = os.path.join(output_path, mesh_name.replace('.glb', ''))
        num_parts_path = os.path.join(mesh_path, 'num_parts.json')
        surface_path = os.path.join(mesh_path, 'points.npy')
        image_path = os.path.join(mesh_path, 'rendering_rmbg.png')
        
        parts_glb_path = os.path.join('mesh/glb_part_voxel', mesh_name)
        
        config = {
            "file": mesh_name,
            "num_parts": 0,
            "valid": False,
            "mesh_path": parts_glb_path,  # Use glb_part_voxel for mesh_path
            "surface_path": None,
            "image_path": None,
            "iou_mean": 0.0,  # No IoU calculation
            "iou_max": 0.0    # No IoU calculation
        }
        try:
            config["num_parts"] = json.load(open(num_parts_path))['num_parts']
            # Skip IoU loading since we don't calculate it
            assert os.path.exists(surface_path)
            config['surface_path'] = surface_path
            assert os.path.exists(image_path)
            config['image_path'] = image_path
            config['valid'] = True
            configs.append(config)
        except:
            continue
    
    configs_path = os.path.join(output_path, 'object_part_configs.json')
    json.dump(configs, open(configs_path, 'w'), indent=4)