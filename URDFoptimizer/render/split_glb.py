#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a GLB file into separate GLB files for each part/geometry.

This script:
1. Loads a GLB file containing multiple geometries (parts)
2. Extracts each geometry as a separate mesh
3. Saves each part as an individual GLB file

Usage:
    python split_glb.py --input input/textured.glb --output output/
"""

import argparse
from pathlib import Path
import trimesh


def split_glb(glb_path: str, output_dir: str):
    """
    Split GLB file into separate files for each geometry/part.

    Args:
        glb_path: Path to input GLB file
        output_dir: Directory to save split GLB files
    """
    print(f"Loading GLB from: {glb_path}")
    scene = trimesh.load(glb_path)

    if isinstance(scene, trimesh.Trimesh):
        # Single mesh - save as is
        print("[WARN] GLB contains a single mesh (no separate parts)")
        output_path = Path(output_dir) / "part_0.glb"
        scene.export(str(output_path))
        print(f"Saved single mesh to: {output_path}")
        return

    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"Unexpected GLB type: {type(scene)}")

    # Extract geometries
    geometries = list(scene.geometry.items())
    print(f"\nFound {len(geometries)} parts in GLB:")

    for i, (name, geom) in enumerate(geometries):
        print(f"  [{i}] {name}: {len(geom.vertices)} verts, {len(geom.faces)} faces")

    if len(geometries) == 0:
        raise ValueError("No geometries found in GLB")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save each part
    print(f"\nSaving parts to: {output_dir}")
    for i, (name, geom) in enumerate(geometries):
        # Clean up name for filename (remove special characters)
        safe_name = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_file = output_path / f"part_{i}_{safe_name}.glb"

        geom.export(str(output_file))
        print(f"  Saved part {i} ({name}) to: {output_file.name}")

    print(f"\n[DONE] Split {len(geometries)} parts")


def main():
    parser = argparse.ArgumentParser(description="Split GLB file into separate parts")
    parser.add_argument("--input", type=str, required=True, help="Path to input GLB file")
    parser.add_argument("--output", type=str, default="URDFoptimizer/render/split_output",
                       help="Output directory for split GLB files")
    args = parser.parse_args()

    split_glb(args.input, args.output)


if __name__ == "__main__":
    main()
