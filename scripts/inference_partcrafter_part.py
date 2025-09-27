"""
PartCrafter Per-Part Images Inference Script

This script performs inference using PartCrafter with per-part images.
Instead of using a single global image for all parts, it uses individual images for each part.

Usage:
    python scripts/inference_partcrafter_part.py \
        --image_folder /path/to/part/images \
        --num_parts 3 \
        --output_dir ./results \
        --render

Expected folder structure:
    /path/to/part/images/
    ├── link_0.png  # Image for part 0
    ├── link_1.png  # Image for part 1
    ├── link_2.png  # Image for part 2
    └── ...

The number of image files (link_0.png to link_{num_parts-1}.png) must exactly match the --num_parts argument.

Output structure:
    ./results/{timestamp}/
    ├── part_00.glb              # Individual part meshes
    ├── part_01.glb
    ├── part_02.glb
    ├── object.glb               # Merged colored mesh
    ├── input_part_00.png        # Copy of input images for reference
    ├── input_part_01.png
    ├── input_part_02.png
    ├── rendering.gif            # Rendered animation (if --render)
    ├── rendering_normal.gif     # Normal rendering animation
    ├── rendering_grid.gif       # Grid rendering animation
    ├── rendering.png            # Single frame renders
    ├── rendering_normal.png
    └── rendering_grid.png

Requirements:
    - PartCrafter model weights in pretrained_weights/PartCrafter/
    - Per-part images named as link_0.png, link_1.png, etc.
    - CUDA-compatible GPU
"""

import argparse
import os
import sys
from glob import glob
import time
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

def load_per_part_images(image_folder: str, num_parts: int, rmbg_net: Any = None, rmbg: bool = False) -> list:
    """
    Load per-part images from a folder.
    Expected image names: link_0.png, link_1.png, ..., link_{num_parts-1}.png

    Args:
        image_folder: Path to folder containing part images
        num_parts: Number of parts to load
        rmbg_net: Background removal network (optional)
        rmbg: Whether to apply background removal

    Returns:
        List of PIL Images for each part
    """
    print(f"DEBUG: Loading {num_parts} part images from folder: {image_folder}")

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    part_images = []
    for i in range(num_parts):
        image_path = os.path.join(image_folder, f"link_{i}.png")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Part image not found: {image_path}")

        print(f"DEBUG: Loading part {i} image from: {image_path}")

        if rmbg and rmbg_net is not None:
            img_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
            print(f"DEBUG: Applied background removal to part {i}")
        else:
            img_pil = Image.open(image_path)

        part_images.append(img_pil)
        print(f"DEBUG: Successfully loaded part {i} image, size: {img_pil.size}")

    print(f"DEBUG: Successfully loaded all {len(part_images)} part images")
    return part_images

@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_folder: str,
    num_parts: int,
    rmbg_net: Any,
    seed: int,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
    rmbg: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> trimesh.Scene:

    # Load per-part images
    print(f"DEBUG: Starting inference with {num_parts} parts")
    part_images = load_per_part_images(image_folder, num_parts, rmbg_net, rmbg)

    print(f"DEBUG: Loaded {len(part_images)} part images, starting pipeline...")
    start_time = time.time()
    outputs = pipe(
        image=part_images,  # Now passing list of per-part images instead of duplicated single image
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes
    end_time = time.time()
    print(f"DEBUG: Pipeline completed in {end_time - start_time:.2f} seconds")
    print(f"DEBUG: Generated {len(outputs)} meshes")

    for i in range(len(outputs)):
        if outputs[i] is None:
            # If the generated mesh is None (decoding error), use a dummy mesh
            print(f"WARNING: Part {i} generated None mesh, using dummy mesh")
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        else:
            print(f"DEBUG: Part {i} mesh has {len(outputs[i].vertices)} vertices and {len(outputs[i].faces)} faces")

    return outputs, part_images

MAX_NUM_PARTS = 16

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser(description="PartCrafter inference with per-part images")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing part images (link_0.png, link_1.png, ..., link_{num_parts-1}.png)")
    parser.add_argument("--num_parts", type=int, required=True, help="number of parts to generate")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--max_num_expanded_coords", type=int, default=1e9)
    parser.add_argument("--use_flash_decoder", action="store_true")
    parser.add_argument("--rmbg", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("DEBUG: PartCrafter Per-Part Images Inference")
    print("=" * 80)
    print(f"DEBUG: Image folder: {args.image_folder}")
    print(f"DEBUG: Number of parts: {args.num_parts}")
    print(f"DEBUG: Output directory: {args.output_dir}")
    print(f"DEBUG: Seed: {args.seed}")
    print(f"DEBUG: Guidance scale: {args.guidance_scale}")
    print(f"DEBUG: Use RMBG: {args.rmbg}")
    print("=" * 80)

    assert 1 <= args.num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"

    # Validate input folder and images
    if not os.path.exists(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")

    print(f"DEBUG: Checking for required part images...")
    missing_images = []
    for i in range(args.num_parts):
        image_path = os.path.join(args.image_folder, f"link_{i}.png")
        if not os.path.exists(image_path):
            missing_images.append(f"link_{i}.png")

    if missing_images:
        print(f"ERROR: Missing part images: {missing_images}")
        raise FileNotFoundError(f"Missing part images in {args.image_folder}: {missing_images}")

    print(f"DEBUG: All required part images found!")

    # Use local pretrained weights (skip download)
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"

    print(f"DEBUG: Loading PartCrafter weights from: {partcrafter_weights_dir}")
    if not os.path.exists(partcrafter_weights_dir):
        raise FileNotFoundError(f"PartCrafter weights not found: {partcrafter_weights_dir}")

    # Only download RMBG if not exists
    if not os.path.exists(rmbg_weights_dir):
        print(f"DEBUG: Downloading RMBG weights to: {rmbg_weights_dir}")
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    print(f"DEBUG: Loading RMBG model...")
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()

    # init tripoSG pipeline (use local weights)
    print(f"DEBUG: Loading PartCrafter pipeline...")
    pipe: PartCrafterPipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)

    set_seed(args.seed)
    print(f"DEBUG: Set random seed to: {args.seed}")

    # run inference
    print(f"DEBUG: Starting inference...")
    outputs, part_images = run_triposg(
        pipe=pipe,
        image_folder=args.image_folder,
        num_parts=args.num_parts,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
        rmbg=args.rmbg,
        dtype=dtype,
        device=device,
    )

    print(f"DEBUG: Inference completed, saving results...")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"DEBUG: Created output directory: {args.output_dir}")

    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")

    export_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(export_dir, exist_ok=True)
    print(f"DEBUG: Saving results to: {export_dir}")

    # Save individual part meshes
    print(f"DEBUG: Saving {len(outputs)} individual part meshes...")
    for i, mesh in enumerate(outputs):
        part_file = os.path.join(export_dir, f"part_{i:02}.glb")
        mesh.export(part_file)
        print(f"DEBUG: Saved part {i} to: {part_file}")

    # Save merged mesh
    print(f"DEBUG: Creating and saving merged mesh...")
    merged_mesh = get_colored_mesh_composition(outputs)
    merged_file = os.path.join(export_dir, "object.glb")
    merged_mesh.export(merged_file)
    print(f"DEBUG: Saved merged mesh to: {merged_file}")

    # Save input images for reference
    print(f"DEBUG: Saving input part images for reference...")
    for i, img in enumerate(part_images):
        img_file = os.path.join(export_dir, f"input_part_{i:02}.png")
        img.save(img_file)
        print(f"DEBUG: Saved input image {i} to: {img_file}")

    print(f"✅ Generated {len(outputs)} parts and saved to {export_dir}")

    if args.render:
        print("DEBUG: Starting rendering...")
        num_views = 36
        radius = 4
        fps = 18

        print(f"DEBUG: Rendering {num_views} views around merged mesh...")
        rendered_images = render_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        print(f"DEBUG: Rendered {len(rendered_images)} regular images")

        rendered_normals = render_normal_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        print(f"DEBUG: Rendered {len(rendered_normals)} normal images")

        rendered_grids = make_grid_for_images_or_videos(
            [rendered_images, rendered_normals],
            nrow=3
        )
        print(f"DEBUG: Created {len(rendered_grids)} grid images")

        # Export animations
        gif_files = []
        gif_file = os.path.join(export_dir, "rendering.gif")
        export_renderings(rendered_images, gif_file, fps=fps)
        gif_files.append(gif_file)
        print(f"DEBUG: Saved rendering animation to: {gif_file}")

        gif_normal_file = os.path.join(export_dir, "rendering_normal.gif")
        export_renderings(rendered_normals, gif_normal_file, fps=fps)
        gif_files.append(gif_normal_file)
        print(f"DEBUG: Saved normal rendering animation to: {gif_normal_file}")

        gif_grid_file = os.path.join(export_dir, "rendering_grid.gif")
        export_renderings(rendered_grids, gif_grid_file, fps=fps)
        gif_files.append(gif_grid_file)
        print(f"DEBUG: Saved grid rendering animation to: {gif_grid_file}")

        # Export single frame images
        rendered_image = rendered_images[0]
        rendered_normal = rendered_normals[0]
        rendered_grid = rendered_grids[0]

        png_file = os.path.join(export_dir, "rendering.png")
        rendered_image.save(png_file)
        print(f"DEBUG: Saved rendering frame to: {png_file}")

        png_normal_file = os.path.join(export_dir, "rendering_normal.png")
        rendered_normal.save(png_normal_file)
        print(f"DEBUG: Saved normal rendering frame to: {png_normal_file}")

        png_grid_file = os.path.join(export_dir, "rendering_grid.png")
        rendered_grid.save(png_grid_file)
        print(f"DEBUG: Saved grid rendering frame to: {png_grid_file}")

        print("✅ Rendering completed!")

    print("=" * 80)
    print(f"🎉 ALL RESULTS SAVED TO: {export_dir}")
    print("   📁 Individual part meshes: part_00.glb, part_01.glb, ...")
    print("   🔗 Merged object mesh: object.glb")
    print("   🖼️  Input part images: input_part_00.png, input_part_01.png, ...")
    if args.render:
        print("   🎬 Rendered animations: rendering.gif, rendering_normal.gif, rendering_grid.gif")
        print("   📸 Rendered frames: rendering.png, rendering_normal.png, rendering_grid.png")
    print("=" * 80)