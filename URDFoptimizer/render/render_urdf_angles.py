#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render URDF at different joint angles using the same camera/lighting as datasets/preprocess/render.py

This script:
1. Parses URDF to extract parent and child links
2. Loads meshes from GLB files referenced in URDF (already split by split_glb.py)
3. Applies forward kinematics to position child mesh at specified joint angles
4. Renders using pyrender with same settings as render.py (RADIUS=4, FOV=40, etc.)
5. Outputs rendered images for each specified angle

Usage:
    python render_urdf_angles.py --urdf render/mobility.urdf --angles 0,120,160 --output render/output/
"""

import os
import math
import argparse
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

# Camera and rendering settings (matching datasets/preprocess/render.py)
RADIUS = 4
IMAGE_SIZE = (2048, 2048)
LIGHT_INTENSITY = 2.0
NUM_ENV_LIGHTS = 36


# --------------------------
# URDF parsing utilities
# --------------------------

def _parse_xyz(s: Optional[str]) -> np.ndarray:
    if s is None:
        return np.zeros(3, dtype=np.float32)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"xyz must have 3 values, got: {s}")
    return np.array(vals, dtype=np.float32)


def _parse_rpy(s: Optional[str]) -> np.ndarray:
    if s is None:
        return np.zeros(3, dtype=np.float32)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"rpy must have 3 values, got: {s}")
    return np.array(vals, dtype=np.float32)


def rpy_to_matrix(rpy: Tuple[float, float, float]) -> np.ndarray:
    """Convert roll-pitch-yaw to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=np.float32)
    return R


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula for rotation matrix from axis-angle."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s,   x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s,   c + z*z*C]
    ], dtype=np.float32)
    return R


def make_SE3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 SE3 matrix from 3x3 rotation and 3D translation."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def find_single_revolute_joint(root: ET.Element, joint_name: Optional[str] = None) -> ET.Element:
    joints = []
    for j in root.findall('joint'):
        if j.attrib.get('type') == 'revolute':
            joints.append(j)
    if joint_name:
        for j in joints:
            if j.attrib.get('name') == joint_name:
                return j
        raise ValueError(f"Revolute joint named '{joint_name}' not found.")
    if len(joints) == 0:
        raise ValueError("No revolute joint found in URDF.")
    if len(joints) > 1:
        print("[WARN] Multiple revolute joints found; using the first. Use --joint-name to specify.")
    return joints[0]


def get_link_first_visual(root: ET.Element, link_name: str) -> Tuple[Optional[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns (mesh_path, origin_xyz, origin_rpy, scale) for the first <visual> of the link.
    """
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Link '{link_name}' not found.")
    vis = link.find('visual')
    if vis is None:
        print(f"[WARN] Link '{link_name}' has no <visual>.")
        return None, np.zeros(3, np.float32), np.zeros(3, np.float32), None
    origin = vis.find('origin')
    xyz = _parse_xyz(origin.attrib.get('xyz')) if origin is not None else np.zeros(3, np.float32)
    rpy = _parse_rpy(origin.attrib.get('rpy')) if origin is not None else np.zeros(3, np.float32)
    geom = vis.find('geometry')
    if geom is None:
        print(f"[WARN] Link '{link_name}' visual has no <geometry>.")
        return None, xyz, rpy, None
    mesh = geom.find('mesh')
    if mesh is None:
        print(f"[WARN] Link '{link_name}' visual geometry is not a <mesh>.")
        return None, xyz, rpy, None
    filename = mesh.attrib.get('filename')
    scale = mesh.attrib.get('scale')
    scale = np.array([float(v) for v in scale.split()]) if scale is not None else None
    return filename, xyz, rpy, scale


def resolve_mesh_path(mesh_path: str, urdf_dir: str) -> str:
    if mesh_path is None:
        return None
    # Handle package:// or relative paths
    if mesh_path.startswith("package://"):
        mesh_path = mesh_path.replace("package://", "")
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(urdf_dir, mesh_path)
    return mesh_path


# --------------------------
# Mesh loading and FK
# --------------------------

def load_mesh(mesh_path: str, scale: Optional[np.ndarray], unit_scale: float = 1.0) -> trimesh.Trimesh:
    """Load mesh from file and apply scaling."""
    if mesh_path is None or not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    print(f"Loading mesh from: {mesh_path}")
    tm = trimesh.load(mesh_path, force='mesh')

    if tm.is_empty:
        raise ValueError(f"Mesh is empty: {mesh_path}")

    # Apply URDF scale
    if scale is not None:
        if np.isscalar(scale):
            tm.apply_scale(float(scale))
        else:
            # Anisotropic scaling
            tm.vertices = tm.vertices * scale

    # Apply unit scale
    if unit_scale != 1.0:
        tm.apply_scale(float(unit_scale))

    print(f"Loaded mesh: {len(tm.vertices)} vertices, {len(tm.faces)} faces")
    return tm


def apply_forward_kinematics(
    parent_mesh: trimesh.Trimesh,
    child_mesh: trimesh.Trimesh,
    parent_vis_xyz: np.ndarray,
    parent_vis_rpy: np.ndarray,
    joint_xyz: np.ndarray,
    joint_rpy: np.ndarray,
    joint_axis: np.ndarray,
    joint_angle: float,
    child_vis_xyz: np.ndarray,
    child_vis_rpy: np.ndarray
) -> trimesh.Scene:
    """
    Apply forward kinematics to create a scene with parent and child at the specified joint angle.

    Transform chain:
    - Parent: T_parent = T_parent_visual
    - Child: T_child = T_parent_visual @ T_joint @ T_rotation @ T_child_visual
    """
    scene = trimesh.Scene()

    # Parent transform
    Rp = rpy_to_matrix(tuple(parent_vis_rpy))
    Tp_parent = make_SE3(Rp, parent_vis_xyz)

    # Apply parent transform
    parent_transformed = parent_mesh.copy()
    parent_transformed.apply_transform(Tp_parent)
    scene.add_geometry(parent_transformed, node_name='parent')

    # Child transform chain
    # 1. Joint origin
    Rj = rpy_to_matrix(tuple(joint_rpy))
    Tj = make_SE3(Rj, joint_xyz)

    # 2. Joint rotation (around axis by angle)
    Rrot = axis_angle_to_matrix(joint_axis, joint_angle)
    Trot = make_SE3(Rrot, np.zeros(3, dtype=np.float32))

    # 3. Child visual origin
    Rc = rpy_to_matrix(tuple(child_vis_rpy))
    Tc = make_SE3(Rc, child_vis_xyz)

    # Combined transform: world = Tp @ Tj @ Trot @ Tc
    T_child_world = Tp_parent @ Tj @ Trot @ Tc

    # Apply child transform
    child_transformed = child_mesh.copy()
    child_transformed.apply_transform(T_child_world)
    scene.add_geometry(child_transformed, node_name='child')

    return scene


# --------------------------
# Rendering (using src.utils.render_utils)
# --------------------------

def render_scene(scene: trimesh.Scene, output_path: str,
                 translation: np.ndarray = None, scale: float = None):
    """
    Render scene using the same settings as datasets/preprocess/render.py

    Settings:
    - RADIUS = 4
    - IMAGE_SIZE = (2048, 2048)
    - LIGHT_INTENSITY = 2.0
    - NUM_ENV_LIGHTS = 36

    Args:
        scene: Trimesh scene to render
        output_path: Output image path
        translation: Fixed translation for normalization (if None, compute from scene)
        scale: Fixed scale factor for normalization (if None, compute from scene)

    Returns:
        (translation, scale) tuple used for normalization
    """
    from src.utils.render_utils import render_single_view

    # Copy scene to avoid modifying original
    scene_copy = scene.copy()

    if translation is None or scale is None:
        # Compute normalization parameters using the SAME method as normalize_mesh
        # in src.utils.data_utils (lines 14-18)
        bbox = scene_copy.bounding_box
        computed_translation = -bbox.centroid
        computed_scale = 2.0 / bbox.primitive.extents.max()

        if translation is None:
            translation = computed_translation
        if scale is None:
            scale = computed_scale

    # Apply the same normalization as normalize_mesh
    scene_copy.apply_translation(translation)
    scene_copy.apply_scale(scale)

    # Convert to geometry for rendering
    geometry = scene_copy.to_geometry()

    # Render single view with fixed camera
    image = render_single_view(
        geometry,
        azimuth=0.0,      # Fixed azimuth
        elevation=0.0,    # Fixed elevation
        radius=RADIUS,    # Fixed distance
        image_size=IMAGE_SIZE,
        fov=40.0,         # Fixed FOV
        light_intensity=LIGHT_INTENSITY,
        num_env_lights=NUM_ENV_LIGHTS,
        return_type='pil'
    )

    # Save image
    image.save(output_path)
    print(f"Saved rendering to: {output_path}")

    return translation, scale


# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="Render URDF at different joint angles")
    parser.add_argument("--urdf", type=str, default="render/mobility.urdf", help="Path to URDF file")
    parser.add_argument("--joint-name", type=str, default=None, help="Name of revolute joint (optional)")
    parser.add_argument("--angles", type=str, default="0,120,160", help="Comma-separated list of angles in degrees")
    parser.add_argument("--output", type=str, default="render/output", help="Output directory")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale meshes by this factor")
    args = parser.parse_args()

    # Parse angles
    angles = [float(a) for a in args.angles.split(',')]
    print(f"Will render at angles: {angles} degrees")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Parse URDF
    tree = ET.parse(args.urdf)
    root = tree.getroot()

    joint_el = find_single_revolute_joint(root, args.joint_name)
    joint_name = joint_el.attrib.get('name', 'joint')
    parent_link = joint_el.find('parent').attrib['link']
    child_link = joint_el.find('child').attrib['link']

    jorigin = joint_el.find('origin')
    j_xyz = _parse_xyz(jorigin.attrib.get('xyz') if jorigin is not None else None)
    j_rpy = _parse_rpy(jorigin.attrib.get('rpy') if jorigin is not None else None)

    jaxis_el = joint_el.find('axis')
    if jaxis_el is None or 'xyz' not in jaxis_el.attrib:
        raise ValueError("Revolute joint must have an <axis xyz='...'> element.")
    jaxis = _parse_xyz(jaxis_el.attrib['xyz'])

    print(f"\n[INFO] URDF Joint Information:")
    print(f"  Joint name: {joint_name}")
    print(f"  Parent link: {parent_link}")
    print(f"  Child link: {child_link}")
    print(f"  Joint origin xyz: {j_xyz}")
    print(f"  Joint origin rpy: {j_rpy}")
    print(f"  Joint axis: {jaxis}")

    # Get link visuals
    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))

    parent_mesh_path, parent_vis_xyz, parent_vis_rpy, parent_scale = get_link_first_visual(root, parent_link)
    child_mesh_path, child_vis_xyz, child_vis_rpy, child_scale = get_link_first_visual(root, child_link)

    print(f"\n[INFO] Parent visual:")
    print(f"  Mesh: {parent_mesh_path}")
    print(f"  Origin xyz: {parent_vis_xyz}, rpy: {parent_vis_rpy}")
    print(f"  Scale: {parent_scale}")

    print(f"\n[INFO] Child visual:")
    print(f"  Mesh: {child_mesh_path}")
    print(f"  Origin xyz: {child_vis_xyz}, rpy: {child_vis_rpy}")
    print(f"  Scale: {child_scale}")

    # Resolve mesh paths
    parent_mesh_path = resolve_mesh_path(parent_mesh_path, urdf_dir)
    child_mesh_path = resolve_mesh_path(child_mesh_path, urdf_dir)

    # Load meshes
    if parent_mesh_path is None:
        print("[WARN] Parent has no mesh, creating dummy mesh")
        parent_mesh = trimesh.Trimesh(vertices=[[0,0,0],[1e-6,0,0],[0,1e-6,0]], faces=[[0,1,2]])
    else:
        parent_mesh = load_mesh(parent_mesh_path, parent_scale, args.unit_scale)

    if child_mesh_path is None:
        raise ValueError("Child link has no visual mesh")

    child_mesh = load_mesh(child_mesh_path, child_scale, args.unit_scale)

    # Compute normalization parameters from angle=0 scene (reference pose)
    # This ensures camera stays fixed for all angles
    print(f"\n[INFO] Computing normalization from angle=0 reference pose...")
    scene_ref = apply_forward_kinematics(
        parent_mesh=parent_mesh,
        child_mesh=child_mesh,
        parent_vis_xyz=parent_vis_xyz,
        parent_vis_rpy=parent_vis_rpy,
        joint_xyz=j_xyz,
        joint_rpy=j_rpy,
        joint_axis=jaxis,
        joint_angle=0.0,  # Reference at angle=0
        child_vis_xyz=child_vis_xyz,
        child_vis_rpy=child_vis_rpy
    )

    # Get normalization parameters using the SAME method as normalize_mesh
    bbox_ref = scene_ref.bounding_box
    fixed_translation = -bbox_ref.centroid
    fixed_scale = 2.0 / bbox_ref.primitive.extents.max()

    print(f"[INFO] Fixed normalization (same as normalize_mesh):")
    print(f"  Translation: ({fixed_translation[0]:.4f}, {fixed_translation[1]:.4f}, {fixed_translation[2]:.4f})")
    print(f"  Scale: {fixed_scale:.4f}")
    print(f"[INFO] Camera will stay at fixed position (0, 0, {RADIUS}) looking at origin")

    # Render at each angle using the same normalization
    print(f"\n[INFO] Rendering at {len(angles)} angles with fixed camera...")
    for angle_deg in angles:
        angle_rad = math.radians(angle_deg)
        print(f"\nRendering at {angle_deg}° ({angle_rad:.4f} rad)...")

        # Apply FK
        scene = apply_forward_kinematics(
            parent_mesh=parent_mesh,
            child_mesh=child_mesh,
            parent_vis_xyz=parent_vis_xyz,
            parent_vis_rpy=parent_vis_rpy,
            joint_xyz=j_xyz,
            joint_rpy=j_rpy,
            joint_axis=jaxis,
            joint_angle=angle_rad,
            child_vis_xyz=child_vis_xyz,
            child_vis_rpy=child_vis_rpy
        )

        # Render with fixed normalization
        output_path = output_dir / f"angle_{int(angle_deg):03d}.png"
        render_scene(scene, str(output_path), translation=fixed_translation, scale=fixed_scale)

    print(f"\n[DONE] Rendered {len(angles)} images to {output_dir}")


if __name__ == "__main__":
    main()
