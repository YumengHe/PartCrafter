#!/usr/bin/env python3
"""
Merge OBJ files to GLB format with two outputs:
1. part.glb - Merged parts without textures
2. texture.glb - Preserved original parts with textures

Usage:
    python merge_to_glb.py mesh/partnet mesh/partnet_glb --single 35059
    python merge_to_glb.py mesh/partnet mesh/partnet_glb --range 35059 35100
"""

import os
import sys
import argparse
import trimesh
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import shutil
from typing import Optional, List, Tuple, Dict

def parse_urdf_links(urdf_path: Path) -> Dict[str, List[str]]:
    """
    Parse URDF file and extract link to mesh file mappings.
    
    Args:
        urdf_path (Path): Path to URDF file
    
    Returns:
        dict: Dictionary mapping link names to list of mesh filenames
    """
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        link_to_meshes = defaultdict(list)
        
        for link in root.findall('link'):
            link_name = link.get('name')
            if link_name == 'base':  # Skip base link
                continue
                
            # Find all visual elements in this link
            for visual in link.findall('visual'):
                mesh_elem = visual.find('.//mesh')
                if mesh_elem is not None:
                    filename = mesh_elem.get('filename')
                    if filename:
                        # Extract just the filename from path like "textured_objs/original-50.obj"
                        mesh_file = os.path.basename(filename)
                        link_to_meshes[link_name].append(mesh_file)
        
        return dict(link_to_meshes)
    
    except Exception as e:
        print(f"    Warning: Could not parse URDF file: {e}")
        return {}

def merge_parts(input_dir: Path, object_id: str) -> Optional[trimesh.Scene]:
    """Merge OBJ parts grouped by URDF links (for part.glb)"""
    object_dir = input_dir / object_id
    textured_objs_dir = object_dir / "textured_objs"
    urdf_path = object_dir / "mobility.urdf"
    
    if not textured_objs_dir.exists():
        print(f"Warning: {textured_objs_dir} does not exist")
        return None
    
    if not urdf_path.exists():
        print(f"Warning: No URDF file found in {object_dir}")
        return None
    
    # Parse URDF to get link-to-mesh mappings
    link_to_meshes = parse_urdf_links(urdf_path)
    if not link_to_meshes:
        print(f"Warning: No links found in URDF for {object_id}")
        return None
    
    print(f"Found {len(link_to_meshes)} links in URDF")
    
    # Create a scene to combine all meshes grouped by links
    scene = trimesh.Scene()
    parts_created = 0
    
    # Process each link as a separate part
    for link_name, mesh_files in link_to_meshes.items():
        if not mesh_files:
            continue
            
        print(f"  Processing link: {link_name} ({len(mesh_files)} meshes)")
        
        # Collect meshes for this link (geometry only, no materials)
        link_meshes = []
        
        for mesh_file in mesh_files:
            obj_path = textured_objs_dir / mesh_file
            if not obj_path.exists():
                print(f"    Warning: {mesh_file} not found, skipping")
                continue
                
            try:
                # Load only the geometry, ignore MTL files completely
                mesh = trimesh.load_mesh(str(obj_path))
                
                # Extract geometry from loaded mesh
                if isinstance(mesh, trimesh.Trimesh):
                    # Create a new mesh with only vertices and faces (no materials)
                    clean_mesh = trimesh.Trimesh(
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        process=False
                    )
                    # Set a basic gray color
                    clean_mesh.visual.face_colors = [128, 128, 128, 255]
                    link_meshes.append(clean_mesh)
                    
                elif isinstance(mesh, trimesh.Scene):
                    # Extract all geometries from scene
                    for geometry in mesh.geometry.values():
                        if isinstance(geometry, trimesh.Trimesh):
                            clean_mesh = trimesh.Trimesh(
                                vertices=geometry.vertices,
                                faces=geometry.faces,
                                process=False
                            )
                            clean_mesh.visual.face_colors = [128, 128, 128, 255]
                            link_meshes.append(clean_mesh)
            
            except Exception as e:
                print(f"    Error loading {mesh_file}: {e}")
                continue
        
        # Combine all meshes for this link
        if link_meshes:
            try:
                if len(link_meshes) == 1:
                    combined_mesh = link_meshes[0]
                else:
                    # Combine multiple meshes manually to ensure clean geometry
                    combined_vertices = []
                    combined_faces = []
                    vertex_offset = 0
                    
                    for mesh in link_meshes:
                        combined_vertices.append(mesh.vertices)
                        faces_with_offset = mesh.faces + vertex_offset
                        combined_faces.append(faces_with_offset)
                        vertex_offset += len(mesh.vertices)
                    
                    if combined_vertices and combined_faces:
                        all_vertices = np.vstack(combined_vertices)
                        all_faces = np.vstack(combined_faces)
                        combined_mesh = trimesh.Trimesh(
                            vertices=all_vertices, 
                            faces=all_faces,
                            process=False
                        )
                        combined_mesh.visual.face_colors = [128, 128, 128, 255]
                    else:
                        print(f"    Warning: No valid geometry for {link_name}")
                        continue
                
                # Add to scene with correct link name
                scene.add_geometry(combined_mesh, node_name=link_name)
                parts_created += 1
                print(f"    Created part: {link_name} with {len(link_meshes)} meshes")
                
            except Exception as e:
                print(f"    Error combining meshes for {link_name}: {e}")
    
    # Check if we have any geometry to export
    if len(scene.geometry) == 0:
        print(f"Warning: No valid geometry found for {object_id}")
        return None
    
    print(f"  Total parts (links) created: {parts_created}")
    return scene

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

def preserve_parts_with_textures(input_dir: Path, object_id: str) -> Optional[trimesh.Scene]:
    """Preserve original parts with textures using merge_to_glb_texture.py approach"""
    object_dir = input_dir / object_id
    textured_objs_dir = object_dir / "textured_objs"
    images_dir = object_dir / "images"
    
    if not textured_objs_dir.exists():
        print(f"Warning: {textured_objs_dir} does not exist")
        return None
    
    # Get all OBJ files
    obj_files = list(textured_objs_dir.glob("*.obj"))
    if not obj_files:
        print(f"Warning: No OBJ files found in {textured_objs_dir}")
        return None
    
    print(f"Found {len(obj_files)} OBJ files")
    
    # Create a combined scene
    combined_scene = trimesh.Scene()
    
    # Load and merge all OBJ files
    for i, obj_file in enumerate(obj_files):
        print(f"  Loading {obj_file.name}...")
        
        scene = load_obj_with_materials(obj_file, images_dir)
        if scene is None:
            print(f"  Warning: Failed to load {obj_file}")
            continue
        
        # Add all geometries from this scene to the combined scene
        for geom_name, geometry in scene.geometry.items():
            # Create unique name to avoid conflicts
            unique_name = f"{obj_file.stem}_{geom_name}"
            combined_scene.add_geometry(geometry, geom_name=unique_name)
    
    if len(combined_scene.geometry) == 0:
        print(f"Warning: No geometries were successfully loaded for {object_id}")
        return None
    
    print(f"  Combined {len(combined_scene.geometry)} geometries")
    return combined_scene

def process_object(input_dir: Path, output_dir: Path, object_id: str) -> bool:
    """Process a single object to create both part.glb and texture.glb"""
    print(f"\nProcessing object: {object_id}")
    
    # Create output directory for this object
    object_output_dir = output_dir / object_id
    object_output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Copy mobility.urdf file
    print("Copying mobility.urdf...")
    urdf_input_path = input_dir / object_id / "mobility.urdf"
    if urdf_input_path.exists():
        urdf_output_path = object_output_dir / "mobility.urdf"
        try:
            shutil.copy2(str(urdf_input_path), str(urdf_output_path))
            print(f"✓ Copied: {urdf_output_path}")
        except Exception as e:
            print(f"✗ Failed to copy mobility.urdf: {e}")
    else:
        print("✗ mobility.urdf not found")
    
    # Generate part.glb (merged parts)
    print("Creating part.glb (merged parts)...")
    merged_scene = merge_parts(input_dir, object_id)
    if merged_scene is not None:
        part_output_path = object_output_dir / "part.glb"
        try:
            merged_scene.export(str(part_output_path))
            print(f"✓ Saved: {part_output_path}")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to save part.glb: {e}")
    else:
        print("✗ Failed to create merged scene")
    
    # Generate texture.glb (original parts with textures)
    print("Creating texture.glb (parts with textures)...")
    textured_scene = preserve_parts_with_textures(input_dir, object_id)
    if textured_scene is not None:
        texture_output_path = object_output_dir / "texture.glb"
        try:
            textured_scene.export(str(texture_output_path))
            print(f"✓ Saved: {texture_output_path}")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to save texture.glb: {e}")
    else:
        print("✗ Failed to create textured scene")
    
    return success_count == 2

def main():
    parser = argparse.ArgumentParser(description="Merge OBJ files to GLB format with two variants")
    parser.add_argument("input_dir", type=str, help="Input directory containing object folders")
    parser.add_argument("output_dir", type=str, help="Output directory for GLB files")
    parser.add_argument("--single", type=str, help="Process single object ID")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), 
                       help="Process range of object IDs")
    parser.add_argument("--list", type=str, help="Text file containing list of object IDs")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which objects to process
    object_ids = []
    
    if args.single:
        object_ids = [args.single]
    elif args.range:
        start, end = args.range
        object_ids = [str(i) for i in range(start, end + 1)]
    elif args.list:
        list_file = Path(args.list)
        if list_file.exists():
            with open(list_file, 'r') as f:
                object_ids = [line.strip() for line in f if line.strip()]
        else:
            print(f"Error: List file {list_file} does not exist")
            sys.exit(1)
    else:
        # Process all subdirectories in input_dir
        object_ids = [d.name for d in input_dir.iterdir() if d.is_dir()]
    
    print(f"Processing {len(object_ids)} objects...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    success_count = 0
    total_count = len(object_ids)
    
    for i, object_id in enumerate(object_ids, 1):
        print(f"\n[{i}/{total_count}] Processing {object_id}")
        
        # Check if input object directory exists
        object_input_dir = input_dir / object_id
        if not object_input_dir.exists():
            print(f"Warning: Input directory {object_input_dir} does not exist, skipping")
            continue
        
        if process_object(input_dir, output_dir, object_id):
            success_count += 1
            print(f"✓ Successfully processed {object_id}")
        else:
            print(f"✗ Failed to process {object_id}")
    
    print(f"\n" + "=" * 60)
    print(f"Processing complete: {success_count}/{total_count} objects processed successfully")
    
    if success_count == total_count:
        print("✓ All objects processed successfully!")
    else:
        print(f"⚠ {total_count - success_count} objects failed to process")

if __name__ == "__main__":
    main()