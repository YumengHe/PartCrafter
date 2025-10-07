#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render URDF with ALL revolute joints set to the SAME angle for each frame.
Outputs one composite image per angle.

- Keeps the same camera/lighting/normalization logic as your original script.
- Builds a kinematic tree from the URDF and runs FK for ALL links.
- For each requested angle, applies that angle to every revolute (and continuous) joint simultaneously.

Usage:
    python render_urdf_all_joints.py \
        --urdf outputs/test4/mobility.urdf \
        --angles 0,30,60,90,120 \
        --output outputs/test4_all \
        --unit-scale 1.0
"""

import os
import math
import argparse
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, Dict, List
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
# Basic parsers
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
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --------------------------
# URDF helpers
# --------------------------

def get_link_visual(root: ET.Element, link_name: str):
    """
    Return a dict with the first visual's (mesh_path, origin_xyz, origin_rpy, scale) for a link.
    """
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Link '{link_name}' not found.")
    vis = link.find('visual')
    if vis is None:
        return {"mesh": None, "xyz": np.zeros(3, np.float32), "rpy": np.zeros(3, np.float32), "scale": None}
    origin = vis.find('origin')
    xyz = _parse_xyz(origin.attrib.get('xyz')) if origin is not None else np.zeros(3, np.float32)
    rpy = _parse_rpy(origin.attrib.get('rpy')) if origin is not None else np.zeros(3, np.float32)
    geom = vis.find('geometry')
    if geom is None:
        return {"mesh": None, "xyz": xyz, "rpy": rpy, "scale": None}
    mesh = geom.find('mesh')
    if mesh is None:
        return {"mesh": None, "xyz": xyz, "rpy": rpy, "scale": None}
    filename = mesh.attrib.get('filename')
    scale = mesh.attrib.get('scale')
    scale = np.array([float(v) for v in scale.split()]) if scale is not None else None
    return {"mesh": filename, "xyz": xyz, "rpy": rpy, "scale": scale}


def resolve_mesh_path(mesh_path: Optional[str], urdf_dir: str) -> Optional[str]:
    if mesh_path is None:
        return None
    if mesh_path.startswith("package://"):
        mesh_path = mesh_path.replace("package://", "")
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(urdf_dir, mesh_path)
    return mesh_path


def find_all_joints(root: ET.Element) -> List[ET.Element]:
    return list(root.findall('joint'))


def find_all_links(root: ET.Element) -> List[str]:
    return [el.attrib['name'] for el in root.findall('link')]


def find_root_links(root: ET.Element) -> List[str]:
    """
    A root link is a link that never appears as a child in any joint.
    """
    links = set(find_all_links(root))
    children = set()
    for j in find_all_joints(root):
        child_link = j.find('child').attrib['link']
        children.add(child_link)
    return list(links - children)


# --------------------------
# Mesh loading
# --------------------------

def load_mesh(mesh_path: str, scale: Optional[np.ndarray], unit_scale: float = 1.0) -> trimesh.Trimesh:
    if mesh_path is None or not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    tm = trimesh.load(mesh_path, force='mesh')
    if tm.is_empty:
        raise ValueError(f"Mesh is empty: {mesh_path}")
    if scale is not None:
        if np.isscalar(scale):
            tm.apply_scale(float(scale))
        else:
            tm.vertices = tm.vertices * scale
    if unit_scale != 1.0:
        tm.apply_scale(float(unit_scale))
    return tm


# --------------------------
# Kinematic tree and FK
# --------------------------

class JointInfo:
    def __init__(self, name, jtype, parent, child, T_origin, axis):
        self.name = name
        self.jtype = jtype
        self.parent = parent
        self.child = child
        self.T_origin = T_origin  # 4x4: parent_link -> joint frame
        self.axis = axis  # 3, in joint frame (URDF convention)


def build_kinematic_graph(root: ET.Element) -> Tuple[Dict[str, List[JointInfo]], Dict[str, Dict]]:
    """
    Returns:
      - graph: parent_link -> list of JointInfo outgoing to children
      - link_visuals: link_name -> dict(mesh, xyz, rpy, scale)
    """
    link_visuals = {}
    for lname in find_all_links(root):
        link_visuals[lname] = get_link_visual(root, lname)

    graph: Dict[str, List[JointInfo]] = {}
    for j in find_all_joints(root):
        name = j.attrib.get('name', 'joint')
        jtype = j.attrib.get('type', 'fixed')
        parent = j.find('parent').attrib['link']
        child = j.find('child').attrib['link']
        jorigin = j.find('origin')
        j_xyz = _parse_xyz(jorigin.attrib.get('xyz') if jorigin is not None else None)
        j_rpy = _parse_rpy(jorigin.attrib.get('rpy') if jorigin is not None else None)
        Rj = rpy_to_matrix(tuple(j_rpy))
        Tj = make_SE3(Rj, j_xyz)
        axis_el = j.find('axis')
        if axis_el is not None and 'xyz' in axis_el.attrib:
            jaxis = _parse_xyz(axis_el.attrib['xyz'])
        else:
            jaxis = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # default
        info = JointInfo(name, jtype, parent, child, Tj, jaxis)
        graph.setdefault(parent, []).append(info)
    return graph, link_visuals


def fk_collect_geometry(root_links: List[str],
                        graph: Dict[str, List[JointInfo]],
                        link_visuals: Dict[str, Dict],
                        link_meshes: Dict[str, Optional[trimesh.Trimesh]],
                        all_joint_angles_rad: Dict[str, float]) -> trimesh.Scene:
    """
    Traverse from each root link, place link visuals using FK into a Trimesh Scene.
    - For revolute/continuous: apply rotation about axis by angle
    - For prismatic: (not handled; treated as zero translation here)
    - For fixed: just propagate
    """
    scene = trimesh.Scene()

    def recurse(link_name: str, T_world_link: np.ndarray):
        # 1) Add this link's visual (if any)
        vis = link_visuals[link_name]
        mesh = link_meshes.get(link_name, None)
        if mesh is not None:
            Rc = rpy_to_matrix(tuple(vis["rpy"]))
            Tc = make_SE3(Rc, vis["xyz"])
            geom = mesh.copy()
            geom.apply_transform(T_world_link @ Tc)
            scene.add_geometry(geom, node_name=f"link::{link_name}")

        # 2) Recurse into children via joints
        for jinfo in graph.get(link_name, []):
            T_child = T_world_link @ jinfo.T_origin
            if jinfo.jtype in ("revolute", "continuous"):
                angle = all_joint_angles_rad.get(jinfo.name, 0.0)
                Rrot = axis_angle_to_matrix(jinfo.axis, angle)
                T_child = T_child @ make_SE3(Rrot, np.zeros(3, dtype=np.float32))
            elif jinfo.jtype == "prismatic":
                # Not implementing prismatic motion; keep at zero displacement
                pass
            elif jinfo.jtype == "fixed":
                pass

            recurse(jinfo.child, T_child)

    I4 = np.eye(4, dtype=np.float32)
    for root_link in root_links:
        recurse(root_link, I4)

    return scene


# --------------------------
# Rendering (using src.utils.render_utils)
# --------------------------

def render_scene(scene: trimesh.Scene, output_path: str,
                 translation: np.ndarray = None, scale: float = None):
    from src.utils.render_utils import render_single_view

    scene_copy = scene.copy()

    if translation is None or scale is None:
        bbox = scene_copy.bounding_box
        computed_translation = -bbox.centroid
        computed_scale = 2.0 / bbox.primitive.extents.max()
        if translation is None:
            translation = computed_translation
        if scale is None:
            scale = computed_scale

    scene_copy.apply_translation(translation)
    scene_copy.apply_scale(scale)

    geometry = scene_copy.to_geometry()
    image = render_single_view(
        geometry,
        azimuth=0.0,
        elevation=0.0,
        radius=RADIUS,
        image_size=IMAGE_SIZE,
        fov=40.0,
        light_intensity=LIGHT_INTENSITY,
        num_env_lights=NUM_ENV_LIGHTS,
        return_type='pil'
    )
    image.save(output_path)
    print(f"Saved rendering to: {output_path}")
    return translation, scale


# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="Render URDF with ALL revolute joints set to SAME angle")
    parser.add_argument("--urdf", type=str, default="outputs/test4/mobility.urdf", help="Path to URDF file")
    parser.add_argument("--angles", type=str, default="0,120,160", help="Comma-separated list of angles in degrees")
    parser.add_argument("--output", type=str, default="outputs/test4_all", help="Output directory")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale meshes by this factor")
    args = parser.parse_args()

    angles_deg = [float(a) for a in args.angles.split(",")]
    angles_rad_list = [math.radians(a) for a in angles_deg]
    print(f"Will render {len(angles_deg)} frames with ALL revolute joints = angle: {angles_deg}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse URDF
    tree = ET.parse(args.urdf)
    root = tree.getroot()

    # Build graph & visuals
    graph, link_visuals = build_kinematic_graph(root)
    root_links = find_root_links(root)
    if len(root_links) == 0:
        raise RuntimeError("No root link found; URDF may be malformed.")
    print(f"Root link(s): {root_links}")

    # Resolve and load meshes for all links
    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))
    link_meshes: Dict[str, Optional[trimesh.Trimesh]] = {}
    for lname, vis in link_visuals.items():
        mesh_path = resolve_mesh_path(vis["mesh"], urdf_dir)
        if mesh_path is None:
            link_meshes[lname] = None
            print(f"[WARN] Link '{lname}' has no visual mesh; adding nothing for this link.")
            continue
        try:
            link_meshes[lname] = load_mesh(mesh_path, vis["scale"], args.unit_scale)
        except Exception as e:
            print(f"[WARN] Failed to load mesh for link '{lname}': {e}")
            link_meshes[lname] = None

    # Create angle=0 reference scene for normalization
    zero_angles = {}
    for parent, joints in graph.items():
        for jinfo in joints:
            if jinfo.jtype in ("revolute", "continuous"):
                zero_angles[jinfo.name] = 0.0
    print(f"[INFO] Computing normalization from reference pose (all revolute=0 rad) ...")
    scene_ref = fk_collect_geometry(root_links, graph, link_visuals, link_meshes, zero_angles)

    bbox_ref = scene_ref.bounding_box
    fixed_translation = -bbox_ref.centroid
    fixed_scale = 2.0 / bbox_ref.primitive.extents.max()
    print(f"[INFO] Fixed normalization: translation={fixed_translation}, scale={fixed_scale:.6f}")

    # Render for each requested angle (applied to ALL revolute joints)
    for a_deg, a_rad in zip(angles_deg, angles_rad_list):
        all_angles = {}
        for parent, joints in graph.items():
            for jinfo in joints:
                if jinfo.jtype in ("revolute", "continuous"):
                    all_angles[jinfo.name] = a_rad
        scene = fk_collect_geometry(root_links, graph, link_visuals, link_meshes, all_angles)

        output_path = out_dir / f"angle_{int(a_deg):03d}.png"
        render_scene(scene, str(output_path), translation=fixed_translation, scale=fixed_scale)

    print(f"[DONE] Rendered {len(angles_deg)} images to {out_dir}")


if __name__ == "__main__":
    main()
