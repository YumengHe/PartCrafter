#!/usr/bin/env python3
"""
Extract watertight surface mesh using voxelization and marching cubes.
Processes each part separately and normalizes the final result.
"""

import argparse
import sys
import trimesh
import numpy as np
from pathlib import Path
import os


def process_single_part(mesh, name, voxel_resolution):
    """Process a single mesh part with voxelization."""
    print(f"  Processing part: {name}")
    
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"  Skipping {name} (not a mesh)")
        return mesh
    
    print(f"  Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Watertight: {mesh.is_watertight}")
    
    # If already watertight, skip voxelization
    if mesh.is_watertight:
        print(f"  {name} is already watertight, skipping voxelization")
        return mesh
    
    # Calculate voxel pitch (size)
    if max(mesh.extents) == 0:
        print(f"  {name} has zero extents, skipping")
        return mesh
        
    pitch = max(mesh.extents) / voxel_resolution
    
    # Estimate memory usage
    estimated_voxels = (mesh.extents / pitch).prod()
    estimated_memory_mb = estimated_voxels * 1 / (1024 * 1024)
    
    if estimated_memory_mb > 500:
        print(f"  High memory usage predicted ({estimated_memory_mb:.1f} MB), reducing resolution...")
        target_memory_mb = 200
        scale_factor = (target_memory_mb / estimated_memory_mb) ** (1/3)
        voxel_resolution = max(10, int(voxel_resolution * scale_factor))
        pitch = max(mesh.extents) / voxel_resolution
        print(f"  Adjusted resolution: {voxel_resolution}")
    
    try:
        # Store original mesh properties
        original_bounds = mesh.bounds
        original_center = mesh.bounds.mean(axis=0)
        original_extents = mesh.extents
        
        # Voxelize the mesh
        vox = mesh.voxelized(pitch=pitch)
        
        # Extract isosurface using marching cubes
        shell = vox.marching_cubes
        
        # Transform shell back to original position and scale
        voxel_bounds = shell.bounds
        voxel_center = shell.bounds.mean(axis=0)
        voxel_extents = shell.extents
        
        # Calculate scale factor to match original extents
        scale_factors = original_extents / voxel_extents
        # Use minimum scale factor to preserve aspect ratio
        scale_factor = np.min(scale_factors[scale_factors > 0])
        
        # Apply transformation: scale first, then translate
        shell.vertices = shell.vertices * scale_factor
        shell_center_after_scale = shell.bounds.mean(axis=0)
        translation = original_center - shell_center_after_scale
        shell.vertices = shell.vertices + translation
        
        # Clean up
        shell.update_faces(shell.nondegenerate_faces())
        shell.remove_unreferenced_vertices()
        
        print(f"  Result: {len(shell.vertices)} vertices, {len(shell.faces)} faces")
        print(f"  Watertight: {shell.is_watertight}")
        print(f"  Scale factor: {scale_factor:.4f}")
        
        return shell
        
    except (MemoryError, Exception) as e:
        print(f"  Error voxelizing {name}: {e}")
        print(f"  Returning original mesh")
        return mesh



def process_file(input_path, output_path, voxel_resolution=100):
    """Process a single GLB file and save to output path."""
    try:
        print(f"Loading GLB file: {input_path}")
        
        # Load the GLB file as a scene
        scene = trimesh.load(input_path, force='scene')
        
        if isinstance(scene, trimesh.Scene):
            print(f"Found {len(scene.geometry)} parts")
            
            # Process each part separately
            processed_geometries = {}
            
            for name, mesh in scene.geometry.items():
                processed_mesh = process_single_part(mesh, name, voxel_resolution)
                processed_geometries[name] = processed_mesh
            
            # Create new scene
            new_scene = trimesh.Scene(geometry=processed_geometries)
            
        else:
            print("Single mesh loaded")
            new_scene = process_single_part(scene, "main_mesh", voxel_resolution)
        
        # Clean up meshes using trimesh's built-in methods
        print("\nCleaning up meshes...")
        if isinstance(new_scene, trimesh.Scene):
            for name, mesh in new_scene.geometry.items():
                if isinstance(mesh, trimesh.Trimesh):
                    original_faces = len(mesh.faces)
                    original_vertices = len(mesh.vertices)
                    
                    # Apply trimesh cleanup methods (using new API)
                    if hasattr(mesh, 'unique_faces'):
                        mesh.update_faces(mesh.unique_faces())
                    else:
                        mesh.remove_duplicate_faces()
                    after_dup_faces = len(mesh.faces)
                    
                    # Remove degenerate faces (zero-area faces)
                    if hasattr(mesh, 'nondegenerate_faces'):
                        mesh.update_faces(mesh.nondegenerate_faces())
                    else:
                        mesh.remove_degenerate_faces()
                    after_degen_faces = len(mesh.faces)
                    
                    mesh.remove_unreferenced_vertices()
                    after_vertices = len(mesh.vertices)
                    
                    dup_removed = original_faces - after_dup_faces
                    degen_removed = after_dup_faces - after_degen_faces
                    vert_removed = original_vertices - after_vertices
                    
                    print(f"  {name}:")
                    if dup_removed > 0:
                        print(f"    Removed {dup_removed} duplicate faces")
                    if degen_removed > 0:
                        print(f"    Removed {degen_removed} degenerate faces")
                    if vert_removed > 0:
                        print(f"    Removed {vert_removed} unreferenced vertices")
                    if dup_removed == 0 and degen_removed == 0 and vert_removed == 0:
                        print(f"    No cleanup needed")
        else:
            if isinstance(new_scene, trimesh.Trimesh):
                original_faces = len(new_scene.faces)
                original_vertices = len(new_scene.vertices)
                
                # Apply trimesh cleanup methods (using new API)
                if hasattr(new_scene, 'unique_faces'):
                    new_scene.update_faces(new_scene.unique_faces())
                else:
                    new_scene.remove_duplicate_faces()
                after_dup_faces = len(new_scene.faces)
                
                # Remove degenerate faces (zero-area faces)
                if hasattr(new_scene, 'nondegenerate_faces'):
                    new_scene.update_faces(new_scene.nondegenerate_faces())
                else:
                    new_scene.remove_degenerate_faces()
                after_degen_faces = len(new_scene.faces)
                
                new_scene.remove_unreferenced_vertices()
                after_vertices = len(new_scene.vertices)
                
                dup_removed = original_faces - after_dup_faces
                degen_removed = after_dup_faces - after_degen_faces
                vert_removed = original_vertices - after_vertices
                
                if dup_removed > 0:
                    print(f"  Removed {dup_removed} duplicate faces")
                if degen_removed > 0:
                    print(f"  Removed {degen_removed} degenerate faces")
                if vert_removed > 0:
                    print(f"  Removed {vert_removed} unreferenced vertices")
                if dup_removed == 0 and degen_removed == 0 and vert_removed == 0:
                    print("  No cleanup needed")
        
        final_scene = new_scene
        
        # Export the result
        final_scene.export(output_path)
        print(f"Watertight surface saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        return False


def process_folder(input_folder, voxel_resolution=100):
    """Process all GLB files in a folder."""
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder {input_path} does not exist")
        return False
    
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory")
        return False
    
    # Create output folder with '_voxel' suffix
    output_folder = input_path.parent / f"{input_path.name}_voxel"
    output_folder.mkdir(exist_ok=True)
    
    # Find all GLB files
    glb_files = list(input_path.glob("*.glb")) + list(input_path.glob("*.gltf"))
    
    if not glb_files:
        print(f"No GLB/GLTF files found in {input_path}")
        return False
    
    print(f"Found {len(glb_files)} GLB/GLTF files in {input_path}")
    print(f"Output folder: {output_folder}")
    print("-" * 60)
    
    success_count = 0
    total_count = len(glb_files)
    
    for i, glb_file in enumerate(glb_files, 1):
        print(f"\n[{i}/{total_count}] Processing: {glb_file.name}")
        
        # Preserve original filename
        output_file = output_folder / glb_file.name
        
        if process_file(glb_file, output_file, voxel_resolution):
            success_count += 1
        
        print("-" * 40)
    
    print(f"\nProcessing complete: {success_count}/{total_count} files processed successfully")
    return success_count == total_count


def extract_voxel_surface_per_part(input_path, voxel_resolution=100):
    """Extract watertight surface for each part separately (legacy function for single file)."""
    output_path = input_path.parent / f"{input_path.stem}_voxel_surface_{voxel_resolution}.glb"
    return process_file(input_path, output_path, voxel_resolution)


def main():
    parser = argparse.ArgumentParser(description="Extract watertight surface using voxelization")
    parser.add_argument("input_path", help="Path to input GLB file or folder containing GLB files")
    parser.add_argument("-r", "--resolution", type=int, default=100, 
                       help="Voxel resolution (default: 100, higher = more detail)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)
    
    print(f"Input: {input_path}")
    print(f"Voxel resolution: {args.resolution}")
    print("-" * 60)
    
    success = False
    
    if input_path.is_dir():
        # Process folder
        success = process_folder(input_path, args.resolution)
    elif input_path.is_file():
        # Process single file
        if not input_path.suffix.lower() in ['.glb', '.gltf']:
            print(f"Warning: File {input_path} may not be a GLB/GLTF file")
        
        output_path = input_path.parent / f"{input_path.stem}_voxel_surface_{args.resolution}.glb"
        print(f"Output: {output_path}")
        success = extract_voxel_surface_per_part(input_path, args.resolution)
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)
    
    if success:
        print("\n✓ Successfully extracted watertight surface!")
    else:
        print("\n✗ Failed to extract surface")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()