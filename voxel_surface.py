
#!/usr/bin/env python3
"""
voxel_surface.py

Extract per-part watertight surface mesh via voxelization + marching cubes,
with **robust scaling for thin parts**, **filled occupancy** for watertightness,
and **aggressive post-cleaning** to remove zero faces/duplicates and shrink size.

Key fixes vs original:
- Anisotropic axis-wise rescale to match each original extent (prevents thin parts from shrinking).
- Guarantee at least N voxels across the thinnest dimension (configurable) while respecting memory limits.
- Fill the voxel grid before marching cubes to enforce closed solids.
- Stronger cleanup: weld/merge-vertices with tolerance, repeated degenerate-face removal, biggest component keep, optional decimation.

Usage:
  python voxel_surface.py <input_glb_or_folder> -r 100 --min-thickness-voxels 3 --weld-tol 1e-6 --target-faces 0

Notes:
- Requires trimesh >= 4.x and numpy.
- target-faces=0 disables decimation; set to e.g. 50k to compress large meshes.
"""

import argparse
import sys
import trimesh
import numpy as np
from pathlib import Path
import os
import math

# --------------------------- helpers ---------------------------------

def pretty_grid_shape(extents, pitch):
    grid = np.maximum(1, np.ceil(extents / max(pitch, 1e-12)).astype(int))
    return tuple(int(x) for x in grid)

def estimate_voxel_memory_mb(extents, pitch, bytes_per_voxel=1.0):
    # Approximate memory as grid_size product bytes (bool ~1B)
    grid = np.maximum(1, np.ceil(extents / max(pitch, 1e-12)).astype(float))
    voxels = float(np.prod(grid))
    return voxels * bytes_per_voxel / (1024.0 * 1024.0)

def robust_cleanup_mesh(mesh: trimesh.Trimesh, weld_tol: float, loop_passes: int = 2):
    """
    A stronger cleanup: weld vertices, remove duplicates/degenerates repeatedly,
    drop tiny connected components, and fix normals. Returns a new mesh object.
    """
    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
        return mesh

    m = mesh.copy()

    # Weld close vertices first to kill near-zero triangles
    try:
        m.merge_vertices(radius=max(weld_tol, 0.0))
    except Exception:
        pass

    for _ in range(max(1, loop_passes)):
        try:
            if hasattr(m, 'unique_faces'):
                m.update_faces(m.unique_faces())
            else:
                m.remove_duplicate_faces()
        except Exception:
            pass

        try:
            if hasattr(m, 'nondegenerate_faces'):
                m.update_faces(m.nondegenerate_faces())
            else:
                m.remove_degenerate_faces()
        except Exception:
            pass

        try:
            m.remove_unreferenced_vertices()
        except Exception:
            pass

        try:
            trimesh.repair.fix_normals(m)
        except Exception:
            pass

    # Keep the biggest connected component to avoid floating junk
    try:
        parts = m.split(only_watertight=False)
        if len(parts) >= 2:
            parts = sorted(parts, key=lambda p: (len(p.faces), p.volume if p.is_volume else 0.0), reverse=True)
            m = parts[0]
    except Exception:
        pass

    return m

def ensure_axiswise_rescale(shell: trimesh.Trimesh, original_center, original_extents):
    """
    Apply **anisotropic** rescale so that the marching-cubes shell matches
    the original extents on each axis. This prevents thin parts from shrinking.
    """
    if not isinstance(shell, trimesh.Trimesh):
        return shell

    shell = shell.copy()
    se = shell.extents
    # Avoid division by zero if shell had degenerate axis
    scale = np.divide(original_extents, np.maximum(se, 1e-12))

    # Build 4x4 scale matrix (axis-wise)
    S = np.eye(4)
    S[:3, :3] = np.diag(scale)

    # Translate to origin, scale, then translate to original_center
    c = shell.bounds.mean(axis=0)
    T1 = np.eye(4); T1[:3, 3] = -c
    T2 = np.eye(4); T2[:3, 3] = original_center

    M = T2 @ S @ T1
    shell.apply_transform(M)
    return shell

def compute_pitch_with_thickness_guard(extents, voxel_resolution, min_thickness_voxels, mem_limit_mb):
    """
    Start from user resolution on the largest extent, but **guard** the thinnest axis so it still
    has >= min_thickness_voxels (helps doors/sheets). If memory explodes, relax gracefully.
    """
    extents = np.maximum(extents, 1e-9)
    # Candidate 1: standard pitch so that largest axis ~= voxel_resolution
    pitch_by_res = float(np.max(extents)) / max(1, voxel_resolution)
    # Candidate 2: ensure thinnest axis has enough voxels
    pitch_by_thin = float(np.min(extents)) / max(1, min_thickness_voxels)

    # Use smaller pitch (higher resolution) to satisfy both goals
    pitch = min(pitch_by_res, pitch_by_thin)

    # Memory check; if too high, increase pitch uniformly
    mem_mb = estimate_voxel_memory_mb(extents, pitch)
    if mem_mb > mem_limit_mb:
        scale = (mem_limit_mb / max(mem_mb, 1e-9)) ** (1.0/3.0)
        pitch /= max(scale, 1e-6)  # increase pitch (coarser) => fewer voxels
    return pitch

# --------------------------- core pipeline ----------------------------

def process_single_part(mesh, name, voxel_resolution, min_thickness_voxels, mem_limit_mb, weld_tol, target_faces):
    print(f"  Processing part: {name}")

    if not isinstance(mesh, trimesh.Trimesh):
        print(f"  Skipping {name} (not a mesh)")
        return mesh

    print(f"  Original: {len(mesh.vertices)} v, {len(mesh.faces)} f")
    print(f"  Watertight: {mesh.is_watertight}")
    original_bounds = mesh.bounds
    original_center = original_bounds.mean(axis=0)
    original_extents = mesh.extents

    # Quick early exit: already watertight and clean-ish
    if mesh.is_watertight and len(mesh.faces) > 0:
        m = robust_cleanup_mesh(mesh, weld_tol=weld_tol, loop_passes=1)
        print(f"  {name} already watertight; minor cleanup only -> {len(m.vertices)} v, {len(m.faces)} f")
        return m

    if np.max(original_extents) <= 0:
        print(f"  {name} has zero extents, skipping")
        return mesh

    # Compute a pitch that respects both user resolution and thin-part guard
    pitch = compute_pitch_with_thickness_guard(
        original_extents, voxel_resolution, min_thickness_voxels, mem_limit_mb
    )
    grid_shape = pretty_grid_shape(original_extents, pitch)
    mem_mb = estimate_voxel_memory_mb(original_extents, pitch)
    print(f"  Pitch: {pitch:.6g} | Grid ~ {grid_shape} | ~{mem_mb:.1f} MB")

    try:
        # Voxelize; fill occupancy to get a solid => marching cubes closed surface
        vox = mesh.voxelized(pitch=pitch)
        # Fill interior and small holes if the API exists
        try:
            if hasattr(vox, 'fill'):
                vox = vox.fill()
                print("  Filled voxel grid (closed occupancy).")
        except Exception as e:
            print(f"  Voxel fill skipped: {e}")

        # Marching cubes
        shell = vox.marching_cubes

        # Robust axis-wise rescale back to original extents + recenter
        shell = ensure_axiswise_rescale(shell, original_center=original_center, original_extents=original_extents)

        # Cleanup + (optional) decimate
        shell = robust_cleanup_mesh(shell, weld_tol=weld_tol, loop_passes=2)

        if isinstance(target_faces, int) and target_faces > 0 and len(shell.faces) > target_faces:
            try:
                shell = shell.simplify_quadratic_decimation(target_faces)
                print(f"  Decimated to ~{target_faces} faces.")
            except Exception as e:
                print(f"  Decimation failed/skipped: {e}")

        print(f"  Result: {len(shell.vertices)} v, {len(shell.faces)} f | Watertight: {shell.is_watertight}")
        # Extra visibility for boundary edges
        try:
            be = getattr(shell, "edges_boundary", None)
            if be is not None:
                print(f"  Boundary edges: {len(be)}")
        except Exception:
            pass

        return shell

    except (MemoryError, Exception) as e:
        print(f"  Error voxelizing {name}: {e}")
        print(f"  Returning original mesh (post-cleaned).")
        return robust_cleanup_mesh(mesh, weld_tol=weld_tol, loop_passes=1)

def process_file(input_path, output_path, voxel_resolution=100, min_thickness_voxels=3, mem_limit_mb=400,
                 weld_tol=1e-6, target_faces=0):
    """Process a single GLB/GLTF file and save to output path."""
    try:
        print(f"Loading GLB/GLTF: {input_path}")
        scene = trimesh.load(input_path, force='scene')

        if isinstance(scene, trimesh.Scene):
            print(f"Found {len(scene.geometry)} parts")
            processed = {}
            for name, mesh in scene.geometry.items():
                processed[name] = process_single_part(mesh, name, voxel_resolution, min_thickness_voxels,
                                                      mem_limit_mb, weld_tol, target_faces)
            new_scene = trimesh.Scene(geometry=processed)
        else:
            print("Single mesh loaded")
            new_scene = process_single_part(scene, "main_mesh", voxel_resolution, min_thickness_voxels,
                                            mem_limit_mb, weld_tol, target_faces)

        # Final pass cleanup for every mesh in scene
        print("\nFinal scene cleanup...")
        if isinstance(new_scene, trimesh.Scene):
            for name, mesh in list(new_scene.geometry.items()):
                if isinstance(mesh, trimesh.Trimesh):
                    cleaned = robust_cleanup_mesh(mesh, weld_tol=weld_tol, loop_passes=1)
                    new_scene.geometry[name] = cleaned
                    print(f"  {name}: {len(cleaned.vertices)} v, {len(cleaned.faces)} f | watertight={cleaned.is_watertight}")
        else:
            if isinstance(new_scene, trimesh.Trimesh):
                new_scene = robust_cleanup_mesh(new_scene, weld_tol=weld_tol, loop_passes=1)
                print(f"  Mesh: {len(new_scene.vertices)} v, {len(new_scene.faces)} f | watertight={new_scene.is_watertight}")

        final_scene = new_scene
        final_scene.export(output_path)
        print(f"Saved: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        return False

def process_folder(input_folder, voxel_resolution=100, min_thickness_voxels=3, mem_limit_mb=400,
                   weld_tol=1e-6, target_faces=0):
    """Process all GLB/GLTF files in a folder."""
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder {input_path} does not exist")
        return False
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory")
        return False

    output_folder = input_path.parent / f"{input_path.name}_voxel"
    output_folder.mkdir(exist_ok=True)

    glb_files = list(input_path.glob("*.glb")) + list(input_path.glob("*.gltf"))
    if not glb_files:
        print(f"No GLB/GLTF found in {input_path}")
        return False

    print(f"Found {len(glb_files)} file(s)")
    print(f"Output -> {output_folder}")
    print("-" * 60)

    ok = 0
    for i, p in enumerate(glb_files, 1):
        print(f"\n[{i}/{len(glb_files)}] {p.name}")
        out = output_folder / p.name
        if process_file(p, out, voxel_resolution, min_thickness_voxels, mem_limit_mb, weld_tol, target_faces):
            ok += 1
        print("-" * 40)

    print(f"\nDone: {ok}/{len(glb_files)} succeeded")
    return ok == len(glb_files)

def main():
    ap = argparse.ArgumentParser(description="Extract watertight surface (robust thin-part scaling).")
    ap.add_argument("input_path", help="Path to input GLB/GLTF file OR folder")
    ap.add_argument("-r", "--resolution", type=int, default=100, help="Voxel resolution along largest axis (default 100)")
    ap.add_argument("--min-thickness-voxels", type=int, default=3, help="Guard: minimum voxels across thinnest axis (default 3)")
    ap.add_argument("--mem-limit-mb", type=int, default=400, help="Rough memory cap for voxel grid (default 400MB)")
    ap.add_argument("--weld-tol", type=float, default=1e-6, help="Vertex weld tolerance in world units (default 1e-6)")
    ap.add_argument("--target-faces", type=int, default=0, help="Optional decimation target faces (0=disabled)")
    args = ap.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)

    print(f"Input: {input_path}")
    print(f"Resolution: {args.resolution} | MinThicknessVoxels: {args.min_thickness_voxels} | MemLimit: {args.mem_limit_mb}MB")
    print(f"WeldTol: {args.weld_tol} | TargetFaces: {args.target_faces}")
    print("-" * 60)

    if input_path.is_dir():
        ok = process_folder(input_path, args.resolution, args.min_thickness_voxels,
                            args.mem_limit_mb, args.weld_tol, args.target_faces)
    else:
        # Single file
        out = input_path.parent / f"{input_path.stem}_voxel_{args.resolution}.glb"
        ok = process_file(input_path, out, args.resolution, args.min_thickness_voxels,
                          args.mem_limit_mb, args.weld_tol, args.target_faces)

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
