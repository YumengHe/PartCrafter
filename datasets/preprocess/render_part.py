#!/usr/bin/env python3
"""
Script to render each URDF link as separate images.
Parses URDF file to identify links and their corresponding mesh parts,
then renders each link separately using the same camera view as render.py.
"""

import os
import trimesh
import numpy as np
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from PIL import Image

from src.utils.data_utils import normalize_mesh
from src.utils.render_utils import render_single_view

RADIUS = 4
IMAGE_SIZE = (2048, 2048)
LIGHT_INTENSITY = 2.0
NUM_ENV_LIGHTS = 36

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
        print(f"Warning: Could not parse URDF file: {e}")
        return {}


def load_link_meshes(dataset_folder, link_name, mesh_files):
    """
    Load and combine all mesh files for a specific link.
    
    Args:
        dataset_folder (str): Path to the dataset folder
        link_name (str): Name of the link
        mesh_files (list): List of mesh files for this link
        
    Returns:
        trimesh.Scene or None: Combined mesh scene for the link
    """
    textured_objs_folder = os.path.join(dataset_folder, "textured_objs")
    if not os.path.exists(textured_objs_folder):
        print(f"No textured_objs folder found")
        return None

    # Collect meshes for this link
    link_meshes = []

    for mesh_file in mesh_files:
        obj_path = os.path.join(textured_objs_folder, mesh_file)
        if not os.path.exists(obj_path):
            print(f"Warning: {mesh_file} not found for link {link_name}")
            continue
            
        try:
            # Load mesh with textures if available
            mesh = trimesh.load(obj_path, process=False)
            
            if isinstance(mesh, trimesh.Trimesh):
                link_meshes.append(mesh)
            elif isinstance(mesh, trimesh.Scene):
                # Extract all geometries from scene
                for geometry in mesh.geometry.values():
                    if isinstance(geometry, trimesh.Trimesh):
                        link_meshes.append(geometry)
        
        except Exception as e:
            print(f"Error loading {mesh_file}: {e}")
            continue
    
    if not link_meshes:
        print(f"No valid meshes found for link {link_name}")
        return None
    
    # Create scene with all meshes for this link
    if len(link_meshes) == 1:
        scene = trimesh.Scene(link_meshes[0])
    else:
        scene = trimesh.Scene()
        for i, mesh in enumerate(link_meshes):
            scene.add_geometry(mesh, node_name=f"{link_name}_part_{i}")
    
    return scene


def render_link_parts(input_path, output_path, object_name):
    """
    Render each URDF link as separate images.
    
    Args:
        input_path (str): Path to folder containing URDF, textured_objs and images
        output_path (str): Output directory for rendered images
        object_name (str): Name for the output folder
    """
    dataset_folder = Path(input_path)
    if object_name is None:
        object_name = dataset_folder.name
    
    # Look for URDF file
    urdf_path = dataset_folder / "mobility.urdf"
    if not urdf_path.exists():
        print(f"No URDF file found in {dataset_folder}")
        return
    
    # Parse URDF to get link-to-mesh mappings
    link_to_meshes = parse_urdf_links(urdf_path)
    if not link_to_meshes:
        print(f"No links found in URDF")
        return
    
    print(f"Found {len(link_to_meshes)} links in URDF")
    
    # Create output directory
    output_dir = Path(output_path) / object_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render each link separately
    for link_name, mesh_files in link_to_meshes.items():
        print(f"Rendering link: {link_name} ({len(mesh_files)} meshes)")
        
        # Load meshes for this link
        link_scene = load_link_meshes(dataset_folder, link_name, mesh_files)
        if link_scene is None:
            continue
        
        try:
            # Normalize mesh (same as in render.py)
            normalized_scene = normalize_mesh(link_scene)
            geometry = normalized_scene.to_geometry()
            
            # Render with same parameters as render.py
            image = render_single_view(
                geometry,
                radius=RADIUS,
                image_size=IMAGE_SIZE,
                light_intensity=LIGHT_INTENSITY,
                num_env_lights=NUM_ENV_LIGHTS,
                return_type='pil'
            )
            
            # Save image with link name
            output_image_path = output_dir / f'{link_name}.png'
            image.save(str(output_image_path))
            print(f"Saved: {output_image_path}")
            
        except Exception as e:
            print(f"Error rendering link {link_name}: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render URDF links as separate images')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input folder containing URDF, textured_objs and images')
    parser.add_argument('--output', type=str, default='rendered_parts',
                        help='Output directory for rendered images')
    parser.add_argument('--name', type=str, default=None, 
                        help='Custom output folder name')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    object_name = args.name

    assert os.path.exists(input_path), f'{input_path} does not exist'

    render_link_parts(input_path, output_path, object_name)