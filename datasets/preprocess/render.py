import os
import trimesh
import numpy as np
import argparse
import json
import cv2
from PIL import Image

from src.utils.data_utils import normalize_mesh
from src.utils.render_utils import render_single_view

RADIUS = 4
IMAGE_SIZE = (2048, 2048)
# 改变光照强度
LIGHT_INTENSITY = 1.0
NUM_ENV_LIGHTS = 36

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='assets/objects/scissors.glb')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    parser.add_argument('--name', type=str, default=None, help='Custom output folder name')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    assert os.path.exists(input_path), f'{input_path} does not exist'

    # Use custom name if provided, otherwise extract from filename
    if args.name:
        mesh_name = args.name
    else:
        mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mesh = normalize_mesh(trimesh.load(input_path, process=False))
    mesh = mesh.to_geometry()
    image = render_single_view(
        mesh,
        radius=RADIUS,
        image_size=IMAGE_SIZE,
        light_intensity=LIGHT_INTENSITY,
        num_env_lights=NUM_ENV_LIGHTS,
        return_type='pil'
    )
    image.save(os.path.join(output_path, f'rendering.png'))