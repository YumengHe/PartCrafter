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


def find_all_revolute_joints(root: ET.Element) -> list:
    """Find all revolute joints in URDF."""
    joints = []
    for j in root.findall('joint'):
        if j.attrib.get('type') == 'revolute':
            joints.append(j)
    return joints


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


def apply_multi_joint_fk(root: ET.Element, urdf_dir: str, joint_angles: dict, unit_scale: float = 1.0) -> trimesh.Scene:
    """
    Apply forward kinematics for all links in URDF with multiple revolute joints.

    Args:
        root: URDF root element
        urdf_dir: Directory containing URDF (for resolving mesh paths)
        joint_angles: Dict mapping joint names to angles (in radians)
        unit_scale: Scale factor for all meshes

    Returns:
        Scene with all links transformed
    """
    scene = trimesh.Scene()

    # Build link-to-transform map (from world frame)
    link_transforms = {}  # link_name -> 4x4 transform matrix

    # Find root link (link with no parent joint)
    all_links = {link.attrib['name'] for link in root.findall('link')}
    child_links = set()

    # Get all joints
    joints = root.findall('joint')

    # Find child links
    for joint in joints:
        child = joint.find('child')
        if child is not None:
            child_links.add(child.attrib['link'])

    # Root links are those not children of any joint
    root_links = all_links - child_links

    if len(root_links) == 0:
        raise ValueError("No root link found (all links are children)")

    print(f"[INFO] Found {len(all_links)} links, {len(joints)} joints")
    print(f"[INFO] Root links: {root_links}")

    # Initialize root links with identity transform (link frame = world frame)
    for root_link in root_links:
        link_transforms[root_link] = np.eye(4, dtype=np.float32)
        print(f"[DEBUG] Root link '{root_link}' initialized with identity transform")

    # Process joints in order (simple: iterate until all processed)
    processed_joints = set()
    max_iterations = len(joints) + 1
    iteration = 0

    while len(processed_joints) < len(joints) and iteration < max_iterations:
        iteration += 1
        made_progress = False

        for joint in joints:
            joint_name = joint.attrib.get('name', 'unnamed')
            if joint_name in processed_joints:
                continue

            parent_link = joint.find('parent').attrib['link']
            child_link = joint.find('child').attrib['link']

            # Can only process if parent transform is known
            if parent_link not in link_transforms:
                continue

            # Get joint parameters
            jorigin = joint.find('origin')
            j_xyz = _parse_xyz(jorigin.attrib.get('xyz') if jorigin is not None else None)
            j_rpy = _parse_rpy(jorigin.attrib.get('rpy') if jorigin is not None else None)

            # Get joint angle (default to 0)
            joint_angle = joint_angles.get(joint_name, 0.0)

            # Compute joint transform
            Rj = rpy_to_matrix(tuple(j_rpy))
            Tj = make_SE3(Rj, j_xyz)

            # Get joint axis and apply rotation
            if joint.attrib.get('type') == 'revolute':
                jaxis_el = joint.find('axis')
                if jaxis_el is not None and 'xyz' in jaxis_el.attrib:
                    jaxis = _parse_xyz(jaxis_el.attrib['xyz'])
                    Rrot = axis_angle_to_matrix(jaxis, joint_angle)
                    Trot = make_SE3(Rrot, np.zeros(3, dtype=np.float32))
                else:
                    Trot = np.eye(4, dtype=np.float32)
            else:
                # Fixed joint or other types: no rotation
                Trot = np.eye(4, dtype=np.float32)

            # Child link frame transform in world frame
            # IMPORTANT: This is the LINK FRAME, not the visual frame
            # Transform chain: parent_link_frame @ joint_origin @ joint_rotation
            # Visual origin will be applied later when loading the mesh
            link_transforms[child_link] = link_transforms[parent_link] @ Tj @ Trot

            print(f"[DEBUG] Joint '{joint_name}': {parent_link} -> {child_link}")
            print(f"[DEBUG]   Joint origin xyz: {j_xyz}, rpy: {j_rpy}")
            print(f"[DEBUG]   Joint angle: {np.rad2deg(joint_angle):.2f}°")

            processed_joints.add(joint_name)
            made_progress = True

        if not made_progress:
            break

    if len(processed_joints) < len(joints):
        print(f"[WARN] Could not process all joints ({len(processed_joints)}/{len(joints)})")

    # Now load and transform all links
    for link in root.findall('link'):
        link_name = link.attrib['name']

        if link_name not in link_transforms:
            print(f"[WARN] No transform for link: {link_name}, skipping")
            continue

        # Get visual
        mesh_path, vis_xyz, vis_rpy, scale = get_link_first_visual(root, link_name)

        if mesh_path is None:
            continue  # No visual

        # Resolve and load mesh
        mesh_path = resolve_mesh_path(mesh_path, urdf_dir)
        if not os.path.exists(mesh_path):
            print(f"[WARN] Mesh not found: {mesh_path}")
            continue

        mesh = load_mesh(mesh_path, scale, unit_scale)

        # Apply visual origin transform
        Rv = rpy_to_matrix(tuple(vis_rpy))
        Tv = make_SE3(Rv, vis_xyz)

        # Total transform: world = link_frame_transform @ visual_origin
        # This matches the FK chain in render_urdf_angles.py:
        # - Parent: T_parent = T_parent_visual  (for root link, link_frame is identity)
        # - Child: T_child = T_parent_link @ T_joint @ T_rotation @ T_child_visual
        T_world = link_transforms[link_name] @ Tv

        print(f"[DEBUG] Link '{link_name}' visual:")
        print(f"[DEBUG]   Visual origin xyz: {vis_xyz}, rpy: {vis_rpy}")
        print(f"[DEBUG]   Mesh: {os.path.basename(mesh_path)}")
        print(f"[DEBUG]   Link frame translation: {link_transforms[link_name][:3, 3]}")
        print(f"[DEBUG]   Final world translation: {T_world[:3, 3]}")

        # Apply transform
        mesh_transformed = mesh.copy()
        mesh_transformed.apply_transform(T_world)

        scene.add_geometry(mesh_transformed, node_name=link_name)

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
    parser = argparse.ArgumentParser(description="Render URDF at different joint angles (all revolute joints)")
    parser.add_argument("--urdf", type=str, default="outputs/test4/mobility.urdf", help="Path to URDF file")
    parser.add_argument("--angles", type=str, default="0,120", help="Comma-separated list of angles in degrees (applied to ALL revolute joints)")
    parser.add_argument("--output", type=str, default="outputs/test4", help="Output directory")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale meshes by this factor")
    args = parser.parse_args()

    # Parse angles
    angles = [float(a) for a in args.angles.split(',')]
    print(f"Will render at angles: {angles} degrees (applied to ALL revolute joints)")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Parse URDF
    tree = ET.parse(args.urdf)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))

    # Find all revolute joints
    revolute_joints = find_all_revolute_joints(root)

    if len(revolute_joints) == 0:
        raise ValueError("No revolute joints found in URDF")

    print(f"\n[INFO] Found {len(revolute_joints)} revolute joints:")
    for joint in revolute_joints:
        joint_name = joint.attrib.get('name', 'unnamed')
        parent_link = joint.find('parent').attrib['link']
        child_link = joint.find('child').attrib['link']
        print(f"  - {joint_name}: {parent_link} -> {child_link}")

    # Compute normalization parameters from angle=0 reference pose
    # This ensures camera stays fixed for all angles
    print(f"\n[INFO] Computing normalization from all-joints-at-0 reference pose...")

    # All joints at angle 0
    joint_angles_ref = {joint.attrib.get('name', 'unnamed'): 0.0 for joint in revolute_joints}
    scene_ref = apply_multi_joint_fk(root, urdf_dir, joint_angles_ref, args.unit_scale)

    # Get normalization parameters using the SAME method as normalize_mesh
    bbox_ref = scene_ref.bounding_box
    fixed_translation = -bbox_ref.centroid
    fixed_scale = 2.0 / bbox_ref.primitive.extents.max()

    print(f"[INFO] Fixed normalization (same as normalize_mesh):")
    print(f"  Translation: ({fixed_translation[0]:.4f}, {fixed_translation[1]:.4f}, {fixed_translation[2]:.4f})")
    print(f"  Scale: {fixed_scale:.4f}")
    print(f"[INFO] Camera will stay at fixed position (0, 0, {RADIUS}) looking at origin")

    # Render each joint at different angles
    # For each joint, render it at specified angles while keeping other joints at 0
    print(f"\n[INFO] Rendering {len(revolute_joints)} joints, each at {len(angles)} angles with fixed camera...")

    for joint in revolute_joints:
        joint_name = joint.attrib.get('name', 'unnamed')
        print(f"\n[INFO] Rendering joint: {joint_name}")

        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            print(f"  Rendering at {angle_deg}° ({angle_rad:.4f} rad)...")

            # Set only current joint to the specified angle, all others to 0
            joint_angles = {j.attrib.get('name', 'unnamed'): 0.0 for j in revolute_joints}
            joint_angles[joint_name] = angle_rad

            # Apply FK for all joints
            scene = apply_multi_joint_fk(root, urdf_dir, joint_angles, args.unit_scale)

            # Render with fixed normalization
            output_path = output_dir / f"{joint_name}_angle_{int(angle_deg):03d}.png"
            render_scene(scene, str(output_path), translation=fixed_translation, scale=fixed_scale)

    print(f"\n[DONE] Rendered {len(revolute_joints)} joints × {len(angles)} angles = {len(revolute_joints) * len(angles)} images to {output_dir}")


if __name__ == "__main__":
    main()
