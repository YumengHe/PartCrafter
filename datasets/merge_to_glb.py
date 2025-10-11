#!/usr/bin/env python3

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional, List, Dict

try:
    import trimesh
    import numpy as np
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install trimesh[easy] numpy")
    sys.exit(1)

import xml.etree.ElementTree as ET
from collections import defaultdict

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

# Removed material loading functions since we're copying original folders directly

# Removed preserve_parts_with_textures function since we're copying folders directly

def process_object(input_dir: Path, output_dir: Path, object_id: str) -> bool:
    """Process a single object to create part.glb and copy original data folders"""
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
            success_count += 1
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
    
    # Copy images folder
    print("Copying images folder...")
    images_input_path = input_dir / object_id / "images"
    if images_input_path.exists():
        images_output_path = object_output_dir / "images"
        try:
            shutil.copytree(str(images_input_path), str(images_output_path), dirs_exist_ok=True)
            print(f"✓ Copied: {images_output_path}")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to copy images folder: {e}")
    else:
        print("✗ images folder not found")
    
    # Copy textured_objs folder
    print("Copying textured_objs folder...")
    textured_objs_input_path = input_dir / object_id / "textured_objs"
    if textured_objs_input_path.exists():
        textured_objs_output_path = object_output_dir / "textured_objs"
        try:
            shutil.copytree(str(textured_objs_input_path), str(textured_objs_output_path), dirs_exist_ok=True)
            print(f"✓ Copied: {textured_objs_output_path}")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to copy textured_objs folder: {e}")
    else:
        print("✗ textured_objs folder not found")
    
    return success_count >= 3  # urdf + part.glb + at least one folder copied

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