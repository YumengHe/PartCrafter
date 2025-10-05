import os
import trimesh
import numpy as np
import argparse
import json
from PIL import Image
from pathlib import Path

from src.utils.data_utils import normalize_mesh
from src.utils.render_utils import render_single_view

RADIUS = 4
IMAGE_SIZE = (2048, 2048)
LIGHT_INTENSITY = 2.0
NUM_ENV_LIGHTS = 36

def load_all_objs_without_texture(textured_objs_dir: Path) -> trimesh.Scene:
    """Load all OBJ files without materials/textures (ignore .mtl files)"""
    scene = trimesh.Scene()

    obj_files = list(textured_objs_dir.glob("*.obj"))
    if not obj_files:
        raise ValueError(f"No OBJ files found in {textured_objs_dir}")

    print(f"Loading {len(obj_files)} OBJ files without textures...")

    for obj_file in obj_files:
        try:
            # Load OBJ without processing materials - just get geometry
            loaded_mesh = trimesh.load(
                str(obj_file),
                process=False,
                force='mesh',  # Force mesh loading
                skip_materials=True  # Skip material loading
            )

            if isinstance(loaded_mesh, trimesh.Trimesh):
                # Remove any visual properties
                loaded_mesh.visual = trimesh.visual.ColorVisuals()
                scene.add_geometry(loaded_mesh, node_name=obj_file.stem)
            elif isinstance(loaded_mesh, trimesh.Scene):
                # Add all geometries from the loaded scene
                for geom_name, geometry in loaded_mesh.geometry.items():
                    if isinstance(geometry, trimesh.Trimesh):
                        # Remove any visual properties
                        geometry.visual = trimesh.visual.ColorVisuals()
                        unique_name = f"{obj_file.stem}_{geom_name}"
                        scene.add_geometry(geometry, geom_name=unique_name)

        except Exception as e:
            print(f"Warning: Failed to load {obj_file.name}: {e}")
            continue

    if len(scene.geometry) == 0:
        raise ValueError("No valid geometries loaded from textured_objs")

    print(f"Successfully loaded {len(scene.geometry)} parts without textures")
    return scene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing textured_objs folder')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    parser.add_argument('--name', type=str, default=None, help='Custom output folder name')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output

    assert input_path.exists(), f'{input_path} does not exist'

    # Use custom name if provided, otherwise extract from folder name
    if args.name:
        mesh_name = args.name
    else:
        mesh_name = input_path.name

    output_dir = Path(output_path) / mesh_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for textured_objs directory
    textured_objs_dir = input_path / "textured_objs"

    if not textured_objs_dir.exists():
        raise ValueError(f"textured_objs directory not found in {input_path}")

    # Load all OBJ files without textures
    scene = load_all_objs_without_texture(textured_objs_dir)

    # Normalize and render
    normalized_scene = normalize_mesh(scene)
    geometry = normalized_scene.to_geometry()

    image = render_single_view(
        geometry,
        radius=RADIUS,
        image_size=IMAGE_SIZE,
        light_intensity=LIGHT_INTENSITY,
        num_env_lights=NUM_ENV_LIGHTS,
        return_type='pil'
    )

    output_image_path = output_dir / 'rendering_notexture.png'
    image.save(str(output_image_path))
    print(f"Saved no-texture rendering: {output_image_path}")
