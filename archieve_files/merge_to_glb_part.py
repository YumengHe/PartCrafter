#!/usr/bin/env python3
"""
Script to combine OBJ files from partnet folder into GLB files.
Groups OBJ files by URDF links to create proper segmentation without materials.
Each folder in partnet represents one object with multiple OBJ files.
Output GLB files are named after the folder name and saved to output folder.
"""

import os
import sys
import argparse
import trimesh
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_urdf_links(urdf_path):
    """
    Parse URDF file and extract link to mesh file mappings.
    
    Args:
        urdf_path (str): Path to URDF file
    
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


def combine_obj_files_to_glb(dataset_folder, output_folder):
    """
    Combine OBJ files in a dataset folder into a single GLB file, grouped by URDF links.
    No materials or textures are processed - only pure geometry.
    
    Args:
        dataset_folder (str): Path to the dataset folder containing OBJ files
        output_folder (str): Path to output folder for GLB files
    """
    folder_name = os.path.basename(dataset_folder)
    print(f"Processing folder: {folder_name}")
    
    # Look for URDF file
    urdf_path = os.path.join(dataset_folder, "mobility.urdf")
    if not os.path.exists(urdf_path):
        print(f"  No URDF file found in {folder_name}, skipping")
        return
    
    # Parse URDF to get link-to-mesh mappings
    link_to_meshes = parse_urdf_links(urdf_path)
    if not link_to_meshes:
        print(f"  No links found in URDF for {folder_name}, skipping")
        return
    
    print(f"  Found {len(link_to_meshes)} links in URDF")
    
    # Find textured_objs folder
    textured_objs_folder = os.path.join(dataset_folder, "textured_objs")
    if not os.path.exists(textured_objs_folder):
        print(f"  No textured_objs folder found in {folder_name}, skipping")
        return
    
    # Create a scene to combine all meshes
    scene = trimesh.Scene()
    parts_created = 0
    
    # Process each link as a separate part
    for link_name, mesh_files in link_to_meshes.items():
        if not mesh_files:
            continue
            
        print(f"    Processing link: {link_name} ({len(mesh_files)} meshes)")
        
        # Collect meshes for this link (geometry only, no materials)
        link_meshes = []
        
        for mesh_file in mesh_files:
            obj_path = os.path.join(textured_objs_folder, mesh_file)
            if not os.path.exists(obj_path):
                print(f"      Warning: {mesh_file} not found, skipping")
                continue
                
            try:
                # Load only the geometry, ignore MTL files completely
                mesh = trimesh.load_mesh(obj_path)
                
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
                print(f"      Error loading {mesh_file}: {e}")
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
                        print(f"      Warning: No valid geometry for {link_name}")
                        continue
                
                # Add to scene with correct link name
                scene.add_geometry(combined_mesh, node_name=link_name)
                parts_created += 1
                print(f"      Created part: {link_name} with {len(link_meshes)} meshes")
                
            except Exception as e:
                print(f"      Error combining meshes for {link_name}: {e}")
    
    # Check if we have any geometry to export
    if len(scene.geometry) == 0:
        print(f"  No valid geometry found in {folder_name}")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Export as GLB
    output_path = os.path.join(output_folder, f"{folder_name}.glb")
    
    try:
        # Export the scene as GLB (geometry only)
        scene.export(output_path)
        print(f"  Successfully exported: {output_path}")
        print(f"  Total parts (links) created: {parts_created}")
        
    except Exception as e:
        print(f"  Error exporting {output_path}: {e}")


def main():
    """Main function to process PartNet dataset folders."""
    parser = argparse.ArgumentParser(description='Convert PartNet objects to GLB format without materials')
    parser.add_argument('input_dir', help='Path to PartNet dataset directory')
    parser.add_argument('output_dir', help='Output directory for GLB files')
    parser.add_argument('--single', help='Process single object ID instead of entire dataset')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    if args.single:
        # Process single object
        single_obj_dir = input_dir / args.single
        if single_obj_dir.exists():
            print(f"Processing single object: {args.single}")
            combine_obj_files_to_glb(str(single_obj_dir), str(output_dir))
        else:
            print(f"Error: Object directory {single_obj_dir} does not exist")
        return
    
    # Get all subdirectories in input folder (numeric object IDs)
    dataset_folders = [
        input_dir / item.name
        for item in input_dir.iterdir()
        if item.is_dir() and item.name.isdigit()
    ]
    
    if not dataset_folders:
        print("No object folders found!")
        return
    
    print(f"Found {len(dataset_folders)} object folders to process")
    print()
    
    # Process each dataset folder
    successful_exports = 0
    for dataset_folder in sorted(dataset_folders):
        try:
            combine_obj_files_to_glb(str(dataset_folder), str(output_dir))
            successful_exports += 1
        except Exception as e:
            print(f"Error processing {dataset_folder.name}: {e}")
        print()
    
    print(f"Processing complete!")
    print(f"Successfully processed: {successful_exports}/{len(dataset_folders)} folders")


if __name__ == "__main__":
    main()