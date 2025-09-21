#!/usr/bin/env python3
"""
PartNet to GLB Converter with texture
"""

import os
import sys
import json
import shutil
from pathlib import Path
import argparse
import tempfile
from typing import List, Dict, Optional

try:
    import trimesh
    import numpy as np
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install trimesh[easy] numpy")
    sys.exit(1)


def read_mtl_file(mtl_path: Path) -> Dict[str, Dict]:
    """Parse MTL file and extract material properties."""
    materials = {}
    current_material = None
    
    if not mtl_path.exists():
        return materials
    
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            command = parts[0]
            
            if command == 'newmtl':
                current_material = parts[1]
                materials[current_material] = {}
            elif current_material and len(parts) >= 2:
                if command == 'Kd':  # Diffuse color
                    materials[current_material]['diffuse'] = [float(x) for x in parts[1:4]]
                elif command == 'Ks':  # Specular color
                    materials[current_material]['specular'] = [float(x) for x in parts[1:4]]
                elif command == 'map_Kd':  # Diffuse texture
                    materials[current_material]['diffuse_texture'] = parts[1]
                elif command == 'Ns':  # Specular exponent
                    materials[current_material]['shininess'] = float(parts[1])
                elif command == 'd' or command == 'Tr':  # Transparency
                    materials[current_material]['transparency'] = float(parts[1])
    
    return materials


def load_obj_with_materials(obj_path: Path, images_dir: Path) -> Optional[trimesh.Scene]:
    """Load OBJ file with materials and textures."""
    try:
        # Load the mesh
        mesh = trimesh.load(str(obj_path))
        
        # If it's a single mesh, convert to scene
        if isinstance(mesh, trimesh.Trimesh):
            scene = trimesh.Scene([mesh])
        elif isinstance(mesh, trimesh.Scene):
            scene = mesh
        else:
            print(f"Warning: Unexpected mesh type for {obj_path}")
            return None
        
        # Load materials from MTL file
        mtl_path = obj_path.with_suffix('.mtl')
        materials = read_mtl_file(mtl_path)
        
        # Apply materials to meshes in the scene
        for geometry_name, geometry in scene.geometry.items():
            if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                # Try to find corresponding material
                material_name = None
                if hasattr(geometry.visual.material, 'name'):
                    material_name = geometry.visual.material.name
                
                if material_name and material_name in materials:
                    material_props = materials[material_name]
                    
                    # Set diffuse color
                    if 'diffuse' in material_props:
                        diffuse = material_props['diffuse']
                        if len(diffuse) >= 3:
                            # Convert to RGBA (add alpha = 1.0)
                            rgba = diffuse + [1.0] if len(diffuse) == 3 else diffuse
                            geometry.visual.material.diffuse = rgba
                    
                    # Handle texture
                    if 'diffuse_texture' in material_props:
                        texture_path = material_props['diffuse_texture']
                        # Handle relative paths
                        if texture_path.startswith('../images/'):
                            texture_filename = texture_path.replace('../images/', '')
                            full_texture_path = images_dir / texture_filename
                            if full_texture_path.exists():
                                try:
                                    # Load texture image
                                    from PIL import Image
                                    image = Image.open(full_texture_path)
                                    geometry.visual.material.image = image
                                except Exception as e:
                                    print(f"Warning: Could not load texture {full_texture_path}: {e}")
        
        return scene
    
    except Exception as e:
        print(f"Error loading {obj_path}: {e}")
        return None


def merge_partnet_to_glb_trimesh(partnet_dir: Path, output_dir: Path) -> bool:
    """
    Process a single PartNet object directory and create a merged GLB file using trimesh.
    
    Args:
        partnet_dir (Path): Path to the PartNet object directory
        output_dir (Path): Output directory for GLB files
    
    Returns:
        bool: True if successful, False otherwise
    """
    partnet_dir = Path(partnet_dir)
    output_dir = Path(output_dir)
    
    # Check if textured_objs directory exists
    textured_objs_dir = partnet_dir / "textured_objs"
    if not textured_objs_dir.exists():
        print(f"No textured_objs directory found in {partnet_dir}")
        return False
    
    images_dir = partnet_dir / "images"
    object_id = partnet_dir.name
    
    print(f"Processing object {object_id}...")
    
    # Get all OBJ files
    obj_files = list(textured_objs_dir.glob("*.obj"))
    if not obj_files:
        print(f"No OBJ files found in {textured_objs_dir}")
        return False
    
    print(f"Found {len(obj_files)} OBJ files")
    
    # Create a combined scene
    combined_scene = trimesh.Scene()
    
    # Load and merge all OBJ files
    for i, obj_file in enumerate(obj_files):
        print(f"Loading {obj_file.name}...")
        
        scene = load_obj_with_materials(obj_file, images_dir)
        if scene is None:
            print(f"Warning: Failed to load {obj_file}")
            continue
        
        # Add all geometries from this scene to the combined scene
        for geom_name, geometry in scene.geometry.items():
            # Create unique name to avoid conflicts
            unique_name = f"{obj_file.stem}_{geom_name}"
            combined_scene.add_geometry(geometry, geom_name=unique_name)
    
    if len(combined_scene.geometry) == 0:
        print(f"No geometries were successfully loaded for {object_id}")
        return False
    
    print(f"Combined {len(combined_scene.geometry)} geometries")
    
    # Export as GLB
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / f"{object_id}.glb"
    
    try:
        # Export the combined scene
        combined_scene.export(str(glb_path))
        print(f"Successfully exported {glb_path}")
        return True
    except Exception as e:
        print(f"Error exporting GLB for {object_id}: {e}")
        return False


def process_partnet_dataset_trimesh(input_dir: str, output_dir: str):
    """
    Process entire PartNet dataset directory using trimesh.
    
    Args:
        input_dir (str): Path to PartNet dataset directory
        output_dir (str): Output directory for GLB files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return
    
    # Find all object directories (numeric names)
    object_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not object_dirs:
        print(f"No object directories found in {input_path}")
        return
    
    print(f"Found {len(object_dirs)} objects to process")
    
    successful = 0
    failed = 0
    
    for obj_dir in sorted(object_dirs):
        try:
            if merge_partnet_to_glb_trimesh(obj_dir, output_path):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {obj_dir.name}: {e}")
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description='Convert PartNet objects to GLB format using trimesh')
    parser.add_argument('input_dir', help='Path to PartNet dataset directory')
    parser.add_argument('output_dir', help='Output directory for GLB files')
    parser.add_argument('--single', help='Process single object directory instead of entire dataset')
    
    args = parser.parse_args()
    
    if args.single:
        # Process single object
        single_obj_dir = Path(args.input_dir) / args.single
        if single_obj_dir.exists():
            merge_partnet_to_glb_trimesh(single_obj_dir, args.output_dir)
        else:
            print(f"Error: Object directory {single_obj_dir} does not exist")
    else:
        # Process entire dataset
        process_partnet_dataset_trimesh(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()