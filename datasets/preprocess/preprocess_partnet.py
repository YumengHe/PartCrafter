import os
import json
import argparse
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='mesh/partnet_glb', help='Input directory containing subfolders with part.glb and texture.glb')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    assert os.path.exists(input_path), f'{input_path} does not exist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get all subfolders (all have same format: part.glb + texture.glb)
    subfolders = [item for item in os.listdir(input_path) 
                  if os.path.isdir(os.path.join(input_path, item))]

    print(f"Found {len(subfolders)} subfolders")

    for object_id in tqdm(subfolders):
        subfolder_path = os.path.join(input_path, object_id)
        
        # Use texture.glb for rendering
        textured_glb_path = os.path.join(subfolder_path, 'texture.glb')
        
        # Check if voxel.glb exists, use it instead of part.glb
        voxel_glb_path = os.path.join(subfolder_path, 'voxel.glb')
        part_glb_path = os.path.join(subfolder_path, 'part.glb')
        
        if os.path.exists(voxel_glb_path):
            parts_glb_path = voxel_glb_path
            mesh_type = "voxel"
        else:
            parts_glb_path = part_glb_path
            mesh_type = "part"
        
        print(f"Processing {object_id}: using {mesh_type}.glb for point sampling")
        
        # 1. Sample points from mesh surface (using voxel.glb or part.glb)
        os.system(f"python datasets/preprocess/mesh_to_point.py --input {parts_glb_path} --output {output_path} --name {object_id}")
        # 2. Render images (using texture.glb)
        os.system(f"python datasets/preprocess/render.py --input {textured_glb_path} --output {output_path} --name {object_id}")
        # 3. Skip background removal and resizing - keep original rendering.png
        # 4. Skip IoU calculation 
        time.sleep(1)
    
    # generate configs
    configs = []
    for object_id in tqdm(subfolders):
        # Fixed structure: preprocessed_data/object_id/
        mesh_path = os.path.join(output_path, object_id)
        num_parts_path = os.path.join(mesh_path, 'num_parts.json')
        surface_path = os.path.join(mesh_path, 'points.npy')
        image_path = os.path.join(mesh_path, 'rendering.png')
        
        # Use voxel.glb if available, otherwise part.glb for training
        voxel_glb_path = os.path.join(input_path, object_id, 'voxel.glb')
        part_glb_path = os.path.join(input_path, object_id, 'part.glb')
        
        if os.path.exists(voxel_glb_path):
            parts_glb_path = voxel_glb_path
        else:
            parts_glb_path = part_glb_path
        
        config = {
            "file": object_id,
            "num_parts": 0,
            "valid": False,
            "mesh_path": parts_glb_path,  # Use part.glb for mesh_path
            "surface_path": None,
            "image_path": None,
            "iou_mean": 0.0,  # No IoU calculation
            "iou_max": 0.0    # No IoU calculation
        }
        try:
            config["num_parts"] = json.load(open(num_parts_path))['num_parts']
            assert os.path.exists(surface_path)
            config['surface_path'] = surface_path
            assert os.path.exists(image_path)
            config['image_path'] = image_path
            config['valid'] = True
            configs.append(config)
        except Exception as e:
            print(f"Error processing {object_id}: {e}")
            continue
    
    configs_path = os.path.join(output_path, 'object_part_configs.json')
    json.dump(configs, open(configs_path, 'w'), indent=4)