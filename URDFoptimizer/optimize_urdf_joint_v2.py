#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize a URDF revolute joint pivot (origin xyz) by matching a rendered "opened" silhouette.

Inputs:
  1) A GLB/mesh set implied by the URDF (the URDF visuals should point to mesh files; a single GLB is also fine if referenced).
  2) A URDF file describing the object. Only ONE revolute joint is supported in this script.
     The joint currently has the WRONG origin xyz; we optimize it.
  3) A "closed" image (optional, for sanity checks). Same camera/viewpoint as the "opened" image.
  4) An "opened" image used as optimization target (we compare silhouettes).

What it does:
  - Builds a minimal differentiable FK graph for a single revolute joint:
      world -> parent link (identity) -> joint(origin xyz+rpy) -> rotate around axis by target angle -> child visual origin
  - Loads parent and child meshes from the URDF visuals (first visual per link).
  - Renders the composite mesh as a soft silhouette (PyTorch3D).
  - Computes a pixelwise MSE loss against the target opened silhouette (from the provided photo with a simple white-background heuristic or given a pre-made mask).
  - Optimizes the joint origin (dx, dy, dz).
  - Writes a new URDF:
        joint/@origin xyz += (dx, dy, dz)
        child_link/visual/@origin xyz -= (dx, dy, dz)   (also applies to child collision origins if they exist)
  - Saves to <original_basename>_optimized.urdf

Assumptions / Simplifications:
  - Exactly one revolute joint in the URDF (or specify --joint-name).
  - We treat the parent link as the base (identity). If your URDF uses a chain, reduce to the single moving child for now.
  - The camera is fixed; you can pass camera distance/elev/azim/FOV via CLI.
  - Unit is meters. If your meshes are in another unit, use --unit-scale to rescale meshes (e.g., 0.001 for mm->m).
  - Background in the target image is (approximately) white. You can pass --target-mask to provide a binary mask yourself.

Install (example):
  pip install torch torchvision
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  pip install trimesh Pillow numpy

Example:
  python optimize_urdf_joint.py 
"""

import os
import math
import argparse
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, Dict
import shutil

import numpy as np
from PIL import Image

import torch
from torch import nn

# PyTorch3D imports
try:
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.renderer import (
        look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings,
        MeshRenderer, MeshRasterizer, HardPhongShader, TexturesVertex,
        PointLights, Materials
    )
    from pytorch3d.structures import Meshes
except Exception as e:
    raise RuntimeError("PyTorch3D is required. Install it first. Error: {}".format(e))

# We will use trimesh for loading GLB/other mesh formats that PyTorch3D's loader doesn't natively handle.
try:
    import trimesh
except Exception as e:
    raise RuntimeError("trimesh is required. Install it first. Error: {}".format(e))


# --------------------------
# Utility math / transforms
# --------------------------

def rpy_to_matrix(rpy: Tuple[float, float, float]) -> torch.Tensor:
    """Convert roll-pitch-yaw (XYZ fixed angles) to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # XYZ order: R = Rz(y) * Ry(p) * Rx(r) if using fixed-axis? URDF uses rpy (roll about X, pitch about Y, yaw about Z) applied in that order.
    # That corresponds to R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = torch.tensor([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=torch.float32)
    return R


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula for 3x3 rotation. axis: (3,), angle: scalar tensor."""
    axis = axis / (axis.norm() + 1e-9)
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c
    R = torch.stack([
        torch.stack([c + x*x*C,     x*y*C - z*s,   x*z*C + y*s]),
        torch.stack([y*x*C + z*s,   c + y*y*C,     y*z*C - x*s]),
        torch.stack([z*x*C - y*s,   z*y*C + x*s,   c + z*z*C])
    ])
    return R


def make_SE3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build 4x4 SE3 from 3x3 R and 3 t."""
    T = torch.eye(4, dtype=torch.float32, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points_h(T: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Apply homogeneous 4x4 transform T to Nx3 points V -> Nx3."""
    ones = torch.ones((V.shape[0], 1), dtype=V.dtype, device=V.device)
    Vh = torch.cat([V, ones], dim=1)  # Nx4
    Vw = (T @ Vh.T).T  # Nx4
    return Vw[:, :3]


# -------------------------------------
# URDF parsing limited to what we need
# -------------------------------------

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
        print("[WARN] Multiple revolute joints found; defaulting to the first. Use --joint-name to specify.")
    return joints[0]


def get_link_first_visual(root: ET.Element, link_name: str) -> Tuple[Optional[str], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns (mesh_path, origin_xyz, origin_rpy, scale) for the first <visual> of the link.
    If no <visual> or <mesh>, returns (None, zeros, zeros, None).
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


def get_child_collision_origins(root: ET.Element, link_name: str) -> list:
    """Return list of origin elements (ET.Element) for collision blocks of the link."""
    link = root.find(f"./link[@name='{link_name}']")
    out = []
    if link is None:
        return out
    for col in link.findall('collision'):
        ori = col.find('origin')
        if ori is not None:
            out.append(ori)
    return out


def update_urdf_joint_and_child(urdf_path: str, joint_name: str, child_link: str, delta: np.ndarray, out_path: str, silent: bool = False):
    """
    Update URDF:
      - joint/@origin xyz += delta
      - child_link/visual/@origin xyz -= delta
      - child_link/collision/@origin xyz -= delta (if exist)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joint = root.find(f"./joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Joint '{joint_name}' not found in URDF when writing back.")

    jorigin = joint.find('origin')
    if jorigin is None:
        jorigin = ET.SubElement(joint, 'origin')

    cur_xyz = _parse_xyz(jorigin.attrib.get('xyz'))
    new_xyz = cur_xyz + delta
    jorigin.attrib['xyz'] = f"{new_xyz[0]:.8f} {new_xyz[1]:.8f} {new_xyz[2]:.8f}"

    # child visual
    link = root.find(f"./link[@name='{child_link}']")
    if link is None:
        if not silent:
            print(f"[WARN] Child link '{child_link}' not found while updating. Skipping link updates.")
    else:
        vis = link.find('visual')
        if vis is not None:
            vorigin = vis.find('origin')
            if vorigin is None:
                vorigin = ET.SubElement(vis, 'origin')
                cur_vxyz = np.zeros(3, np.float32)
            else:
                cur_vxyz = _parse_xyz(vorigin.attrib.get('xyz'))
            new_vxyz = cur_vxyz - delta
            vorigin.attrib['xyz'] = f"{new_vxyz[0]:.8f} {new_vxyz[1]:.8f} {new_vxyz[2]:.8f}"

        # collisions too
        for cori in get_child_collision_origins(root, child_link):
            cur_cxyz = _parse_xyz(cori.attrib.get('xyz'))
            new_cxyz = cur_cxyz - delta
            cori.attrib['xyz'] = f"{new_cxyz[0]:.8f} {new_cxyz[1]:.8f} {new_cxyz[2]:.8f}"

    # Write out
    ET.indent(tree, space="  ", level=0)  # Py3.9+
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    if not silent:
        print(f"[OK] Wrote optimized URDF to: {out_path}")


# ------------------------------------
# Mesh loading helpers (via trimesh)
# ------------------------------------

def resolve_mesh_path(mesh_path: str, urdf_dir: str) -> str:
    if mesh_path is None:
        return None
    # Handle package:// or relative paths
    if mesh_path.startswith("package://"):
        # naive: drop scheme, join remainder to urdf_dir
        mesh_path = mesh_path.replace("package://", "")
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(urdf_dir, mesh_path)
    return mesh_path


def load_mesh_as_tensors(mesh_path: str, scale: Optional[np.ndarray], unit_scale: float, device: torch.device,
                         normalize: bool = False, normalize_scale: float = 2.0):
    """
    Load mesh (GLB, OBJ, etc.) using trimesh, return (verts: Nx3 torch, faces: Fx3 long).
    Applies scale from URDF and unit_scale.

    Args:
        normalize: If True, normalize mesh to match preprocess (center at origin, scale to normalize_scale)
        normalize_scale: Target max extent for normalization (default 2.0 matching preprocess)
    """
    if mesh_path is None:
        # empty
        V = torch.zeros((0, 3), dtype=torch.float32, device=device)
        F = torch.zeros((0, 3), dtype=torch.int64, device=device)
        print("[DEBUG] No mesh path provided, returning empty mesh")
        return V, F

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    print(f"[DEBUG] Loading mesh from: {mesh_path}")
    tm = trimesh.load(mesh_path, force='mesh')

    if tm.is_empty:
        raise ValueError(f"Mesh is empty: {mesh_path}")

    print(f"[DEBUG] Loaded mesh: {len(tm.vertices)} vertices, {len(tm.faces)} faces")

    V = torch.from_numpy(np.asarray(tm.vertices, dtype=np.float32)).to(device)
    F = torch.from_numpy(np.asarray(tm.faces, dtype=np.int64)).to(device)

    # Compute bounding box before scaling
    v_min = V.min(dim=0)[0].cpu().numpy()
    v_max = V.max(dim=0)[0].cpu().numpy()
    print(f"[DEBUG] Mesh bbox before scaling: min={v_min}, max={v_max}")

    # Apply URDF scaling first
    if scale is not None:
        # URDF mesh scale can be anisotropic
        if np.isscalar(scale):
            V = V * float(scale)
            print(f"[DEBUG] Applied isotropic scale: {scale}")
        else:
            # anisotropic scaling per-axis
            V = V * torch.tensor(scale, dtype=torch.float32, device=device)
            print(f"[DEBUG] Applied anisotropic scale: {scale}")
    if unit_scale != 1.0:
        V = V * float(unit_scale)
        print(f"[DEBUG] Applied unit scale: {unit_scale}")

    # Apply normalization if requested (matching preprocess)
    if normalize:
        # Center at origin
        centroid = V.mean(dim=0)
        V = V - centroid
        print(f"[DEBUG] Centered mesh at origin (centroid: {centroid.cpu().numpy()})")

        # Scale to normalize_scale
        bbox_min = V.min(dim=0)[0]
        bbox_max = V.max(dim=0)[0]
        max_extent = (bbox_max - bbox_min).max().item()
        scale_factor = normalize_scale / max_extent
        V = V * scale_factor
        print(f"[DEBUG] Normalized mesh: max_extent={max_extent:.4f}, scale_factor={scale_factor:.4f}, target_scale={normalize_scale}")

    # Compute bounding box after all transformations
    v_min = V.min(dim=0)[0].cpu().numpy()
    v_max = V.max(dim=0)[0].cpu().numpy()
    print(f"[DEBUG] Mesh bbox after all transforms: min={v_min}, max={v_max}")

    return V, F


def compute_scene_bounds_raw_mesh(parent_V: torch.Tensor, child_V: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Compute the bounding box of the raw mesh geometry (without URDF transforms).
    This matches the approach in render_utils.py where camera distance is set based on original mesh.
    Returns (center, max_extent) for auto camera positioning.
    """
    # Combine raw mesh vertices without any URDF transforms
    all_verts = torch.cat([parent_V, child_V], dim=0)

    # Compute bounding box
    bbox_min = all_verts.min(dim=0)[0]
    bbox_max = all_verts.max(dim=0)[0]
    center = (bbox_min + bbox_max) / 2.0
    extent = (bbox_max - bbox_min).max().item()

    return center, extent


# ----------------------
# Rendering + loss
# ----------------------

def build_phong_renderer(image_size: int, device: torch.device, fov: float,
                         cam_dist: float, cam_elev: float, cam_azim: float,
                         at: Tuple[float, float, float] = ((0.0, 0.0, 0.0),)):
    """
    Build Phong renderer with camera and lighting setup similar to render_utils.py.

    Args:
        at: Camera look-at point (x, y, z). Default is origin.
    """
    # Camera setup - similar to render_utils.py
    # Using spherical coordinates: azimuth and elevation
    R, T = look_at_view_transform(dist=cam_dist, elev=cam_elev, azim=cam_azim, at=at, device=device)

    # Use FoVPerspectiveCameras similar to pyrender's PerspectiveCamera with yfov
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,  # Hard rendering, no anti-aliasing
        bin_size=None
    )

    # Lighting setup - place light at camera position similar to DirectionalLight
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 0.0]],  # Light at camera position
        ambient_color=((0.3, 0.3, 0.3),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.1, 0.1, 0.1),)
    )

    # Materials for Phong shading
    materials = Materials(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((0.2, 0.2, 0.2),),
        shininess=32.0
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights, materials=materials)
    )
    return renderer, cameras


def image_to_silhouette(path: str, image_size: int, threshold: float = 0.9) -> torch.Tensor:
    """
    Load a photo and convert to a binary silhouette (1=object, 0=background) by assuming white background.
    threshold = 0.9 means pixels with brightness < 0.9 are considered foreground.
    """
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Simple heuristic: brightness = mean(R,G,B); foreground if darker than near-white
    bright = arr.mean(axis=2)
    mask = (bright < threshold).astype(np.float32)  # 1 if object (dark), 0 if white-ish
    return torch.from_numpy(mask).unsqueeze(0)  # 1xHxW


# ----------------------
# Core optimization
# ----------------------

class SingleRevoluteOptimizer(nn.Module):
    def __init__(self,
                 parent_V: torch.Tensor, parent_F: torch.Tensor,
                 child_V: torch.Tensor, child_F: torch.Tensor,
                 joint_origin_xyz: np.ndarray, joint_origin_rpy: np.ndarray,
                 joint_axis: np.ndarray,
                 parent_vis_xyz: np.ndarray, parent_vis_rpy: np.ndarray,
                 child_vis_xyz: np.ndarray, child_vis_rpy: np.ndarray,
                 target_angle_rad: float,
                 device: torch.device):
        super().__init__()
        self.device = device

        # Constant geometry
        self.parent_V = parent_V  # (Np, 3)
        self.parent_F = parent_F  # (Fp, 3)
        self.child_V = child_V    # (Nc, 3)
        self.child_F = child_F    # (Fc, 3)

        # Parent visual transform (applied to parent mesh)
        self.parent_vis_r = torch.tensor(parent_vis_rpy, dtype=torch.float32, device=device)
        self.parent_vis_t = torch.tensor(parent_vis_xyz, dtype=torch.float32, device=device)
        self.Rp = rpy_to_matrix(tuple(parent_vis_rpy)).to(device)

        # Constant transforms - precompute rotation matrices
        self.joint_origin_r = torch.tensor(joint_origin_rpy, dtype=torch.float32, device=device)
        self.joint_origin_t0 = torch.tensor(joint_origin_xyz, dtype=torch.float32, device=device)
        # Precompute constant rotation matrix for joint origin
        self.Rj = rpy_to_matrix(tuple(joint_origin_rpy)).to(device)

        self.child_vis_r = torch.tensor(child_vis_rpy, dtype=torch.float32, device=device)
        self.child_vis_t0 = torch.tensor(child_vis_xyz, dtype=torch.float32, device=device)
        # Precompute constant rotation matrix for child visual
        self.Rc = rpy_to_matrix(tuple(child_vis_rpy)).to(device)

        self.axis = torch.tensor(joint_axis, dtype=torch.float32, device=device)
        self.angle = torch.tensor(target_angle_rad, dtype=torch.float32, device=device)
        # Precompute constant rotation matrix for joint axis rotation
        self.Rrot = axis_angle_to_matrix(self.axis, self.angle).to(device)

        # Learnable delta on joint origin translation
        self.delta_t = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))

    def forward(self) -> Meshes:
        """
        Build world-space merged mesh (parent static + child rotated about joint pivot with delta shift).
        """
        # Parent world = apply parent visual transform
        Tp = make_SE3(self.Rp, self.parent_vis_t)
        Vp_w = transform_points_h(Tp, self.parent_V)

        # Joint origin: R_origin, t_origin + delta
        # Use precomputed Rj, only tj changes with delta_t (learnable)
        tj = self.joint_origin_t0 + self.delta_t
        Tj = make_SE3(self.Rj, tj)

        # Rotation about joint axis by target angle (precomputed)
        Trot = make_SE3(self.Rrot, torch.zeros(3, dtype=torch.float32, device=self.device))

        # Child visual local (precomputed)
        tc = self.child_vis_t0
        Tc = make_SE3(self.Rc, tc)

        # World transform for child: T_world = Tj @ Trot @ Tc
        Tw = Tj @ Trot @ Tc

        Vc_w = transform_points_h(Tw, self.child_V)

        # Merge
        V = torch.cat([Vp_w, Vc_w], dim=0)
        F_child = self.child_F + self.parent_V.shape[0]
        F = torch.cat([self.parent_F, F_child], dim=0)

        # Simple white vertex color (unused by silhouette shader but required by Meshes)
        Vtx = torch.ones_like(V)
        meshes = Meshes(verts=[V], faces=[F], textures=TexturesVertex(verts_features=[Vtx]))
        return meshes


def main():
    parser = argparse.ArgumentParser(description="Optimize URDF revolute joint origin xyz by silhouette matching.")
    parser.add_argument("--urdf", type=str, default="input/mobility.urdf", help="Path to URDF file.")
    parser.add_argument("--joint-name", type=str, default=None, help="Name of the revolute joint to optimize (optional).")
    parser.add_argument("--opened-img", type=str, default="input/rendering_open.png", help="Target opened photo.")
    parser.add_argument("--closed-img", type=str, default="input/rendering.png", help="Optional closed photo (sanity checks only).")
    parser.add_argument("--target-mask", type=str, default=None, help="Optional binary mask (white=background, black/object) for opened image.")
    parser.add_argument("--image-size", type=int, default=1024, help="Render size.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Foreground threshold for white background heuristic.")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale meshes by this factor (e.g., 0.001 for mm->m).")
    parser.add_argument("--target-angle-deg", type=float, default=90.0, help="Angle to open the joint for matching (degrees).")
    parser.add_argument("--fov", type=float, default=40.0, help="Perspective FOV in degrees (matching render.py).")
    parser.add_argument("--cam-dist", type=float, default=4.0, help="Camera distance (matching render.py RADIUS=4).")
    parser.add_argument("--cam-elev", type=float, default=0.0, help="Camera elevation in degrees (matching render.py).")
    parser.add_argument("--cam-azim", type=float, default=0.0, help="Camera azimuth in degrees (matching render.py).")
    parser.add_argument("--normalize-mesh", action="store_true", help="Normalize mesh to match preprocess (scale=2.0, centered at origin).")
    parser.add_argument("--normalize-scale", type=float, default=2.0, help="Target scale for mesh normalization (matching preprocess).")
    parser.add_argument("--auto-camera", action="store_true", help="Force auto camera distance even if --cam-dist is set.")
    parser.add_argument("--iters", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for Adam.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu.")
    parser.add_argument("--out-urdf", type=str, default=None, help="Output URDF path (default: <basename>_optimized.urdf).")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Parse URDF
    tree = ET.parse(args.urdf)
    root = tree.getroot()

    # Find the revolute joint
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

    # Child visual (first)
    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))
    print(f"\n[DEBUG] ===== URDF Parsing =====")
    print(f"[DEBUG] URDF directory: {urdf_dir}")
    print(f"[DEBUG] Joint name: {joint_name}")
    print(f"[DEBUG] Parent link: {parent_link}, Child link: {child_link}")
    print(f"[DEBUG] Joint origin xyz: {j_xyz}, rpy: {j_rpy}")
    print(f"[DEBUG] Joint axis: {jaxis}")

    child_mesh_path, child_vis_xyz, child_vis_rpy, child_scale = get_link_first_visual(root, child_link)
    parent_mesh_path, parent_vis_xyz, parent_vis_rpy, parent_scale = get_link_first_visual(root, parent_link)

    print(f"[DEBUG] Parent visual: xyz={parent_vis_xyz}, rpy={parent_vis_rpy}, scale={parent_scale}")
    print(f"[DEBUG] Child visual: xyz={child_vis_xyz}, rpy={child_vis_rpy}, scale={child_scale}")

    # Resolve and load meshes (we apply visual origins separately in the FK graph, so we load raw geometry)
    child_mesh_path = resolve_mesh_path(child_mesh_path, urdf_dir) if child_mesh_path else None
    parent_mesh_path = resolve_mesh_path(parent_mesh_path, urdf_dir) if parent_mesh_path else None

    print(f"\n[DEBUG] ===== Mesh Loading =====")
    print(f"[DEBUG] Parent mesh path: {parent_mesh_path}")
    print(f"[DEBUG] Child mesh path: {child_mesh_path}")

    # If parent has no mesh, create a tiny dummy triangle (to keep structures valid)
    if parent_mesh_path is None:
        print("[WARN] Parent link has no visual mesh; using a tiny dummy triangle so the renderer works.")
        Vp = torch.tensor([[0,0,0],[1e-6,0,0],[0,1e-6,0]], dtype=torch.float32, device=device)
        Fp = torch.tensor([[0,1,2]], dtype=torch.int64, device=device)
    else:
        print(f"\n[DEBUG] Loading parent mesh:")
        Vp, Fp = load_mesh_as_tensors(parent_mesh_path, parent_scale, args.unit_scale, device,
                                       normalize=args.normalize_mesh, normalize_scale=args.normalize_scale)

    if child_mesh_path is None:
        raise ValueError("Child link has no visual mesh, cannot proceed. Provide a mesh in the URDF visual.")

    print(f"\n[DEBUG] Loading child mesh:")
    Vc, Fc = load_mesh_as_tensors(child_mesh_path, child_scale, args.unit_scale, device,
                                   normalize=args.normalize_mesh, normalize_scale=args.normalize_scale)

    print(f"\n[DEBUG] Parent mesh: {Vp.shape[0]} verts, {Fp.shape[0]} faces")
    print(f"[DEBUG] Child mesh: {Vc.shape[0]} verts, {Fc.shape[0]} faces")

    # Compute scene bounds based on RAW mesh (without URDF transforms)
    # This matches render_utils.py approach where camera is set based on original mesh
    print(f"\n[DEBUG] ===== Scene Bounds Computation (Raw Mesh) =====")
    target_angle_rad = math.radians(args.target_angle_deg)
    print(f"[DEBUG] Target angle: {args.target_angle_deg} deg = {target_angle_rad:.4f} rad")

    scene_center, scene_extent = compute_scene_bounds_raw_mesh(Vp, Vc)

    # Auto-adjust camera distance if not manually specified or if --auto-camera is set
    print(f"\n[DEBUG] ===== Camera Setup =====")
    if args.normalize_mesh:
        # When using normalized mesh, use fixed camera distance matching preprocess
        cam_dist = args.cam_dist  # Should be 4.0 by default
        print(f"[INFO] Using normalized mesh mode, camera distance: {cam_dist} (matching render.py RADIUS=4)")
    elif args.auto_camera or args.cam_dist == 4.0:  # new default value matching render.py
        # Calculate distance to fit the whole scene in view
        # dist = extent / (2 * tan(fov/2)) * safety_factor
        fov_rad = math.radians(args.fov)
        auto_cam_dist = (scene_extent / (2.0 * math.tan(fov_rad / 2.0))) * 1.8
        cam_dist = max(auto_cam_dist, 0.5)  # minimum distance 0.5
        print(f"[INFO] Raw mesh extent: {scene_extent:.4f}, Center: [{scene_center[0]:.4f}, {scene_center[1]:.4f}, {scene_center[2]:.4f}]")
        print(f"[INFO] Auto-adjusted camera distance: {cam_dist:.4f} (original: {args.cam_dist})")
    else:
        cam_dist = args.cam_dist
        print(f"[INFO] Using manual camera distance: {cam_dist}")

    print(f"[DEBUG] Camera params: dist={cam_dist:.4f}, elev={args.cam_elev}, azim={args.cam_azim}, fov={args.fov}")

    # Camera looks at origin (0,0,0) similar to render_utils.py
    scene_center_tuple = ((0.0, 0.0, 0.0),)

    # Build optimizer model
    model = SingleRevoluteOptimizer(
        parent_V=Vp, parent_F=Fp,
        child_V=Vc, child_F=Fc,
        joint_origin_xyz=j_xyz, joint_origin_rpy=j_rpy,
        joint_axis=jaxis,
        parent_vis_xyz=parent_vis_xyz, parent_vis_rpy=parent_vis_rpy,
        child_vis_xyz=child_vis_xyz, child_vis_rpy=child_vis_rpy,
        target_angle_rad=target_angle_rad,
        device=device
    ).to(device)

    # Build renderer with HardPhongShader - camera looks at origin
    renderer, cameras = build_phong_renderer(
        image_size=args.image_size, device=device, fov=args.fov,
        cam_dist=cam_dist, cam_elev=args.cam_elev, cam_azim=args.cam_azim,
        at=scene_center_tuple
    )

    # Target silhouette
    print(f"\n[DEBUG] ===== Target Image Loading =====")
    if args.target_mask is not None:
        print(f"[DEBUG] Loading target mask from: {args.target_mask}")
        # If a mask is provided (0/255), load directly
        m = Image.open(args.target_mask).convert("L").resize((args.image_size, args.image_size), Image.Resampling.NEAREST)
        targ = (torch.from_numpy(np.asarray(m, dtype=np.float32)) / 255.0).unsqueeze(0)
        targ = (targ < 0.5).float()  # black -> 1, white -> 0 (invert if needed)
    else:
        print(f"[DEBUG] Extracting silhouette from opened image: {args.opened_img}")
        print(f"[DEBUG] Using threshold: {args.threshold}")
        targ = image_to_silhouette(args.opened_img, args.image_size, threshold=args.threshold)
    targ = targ.to(device)
    print(f"[DEBUG] Target silhouette loaded, shape: {targ.shape}, device: {targ.device}")
    print(f"[DEBUG] Target min/max/mean: {targ.min().item():.4f}/{targ.max().item():.4f}/{targ.mean().item():.4f}")

    # Optimizer
    optim = torch.optim.Adam([model.delta_t], lr=args.lr)

    # Create output directory for rendered images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"[INFO] Output directory created: {output_dir}")

    # Optimization loop
    print(f"\n[DEBUG] ===== Starting Optimization =====")
    for it in range(args.iters):
        optim.zero_grad()
        meshes = model()
        # Render with HardPhongShader
        images = renderer(meshes)
        # images: [1, H, W, 4] with RGBA
        # Extract silhouette from alpha channel
        sil = images[..., 3].squeeze(0)  # HxW

        # loss: L2 between silhouette and target
        loss = torch.mean((sil - targ.squeeze(0))**2)

        loss.backward()
        optim.step()

        if (it+1) % max(1, args.iters // 10) == 0 or it == 0:
            dt = model.delta_t.detach().cpu().numpy()

            # Debug info for first iteration
            if it == 0:
                print(f"\n[DEBUG] First iteration diagnostics:")
                print(f"[DEBUG] Rendered silhouette shape: {sil.shape}")
                print(f"[DEBUG] Silhouette min/max: {sil.min().item():.4f}/{sil.max().item():.4f}")
                print(f"[DEBUG] Silhouette mean: {sil.mean().item():.4f}")
                print(f"[DEBUG] Non-zero pixels in silhouette: {(sil > 0.1).sum().item()}/{sil.numel()}")
                print(f"[DEBUG] Target silhouette shape: {targ.shape}")
                print(f"[DEBUG] Target min/max: {targ.min().item():.4f}/{targ.max().item():.4f}")
                print(f"[DEBUG] Target mean: {targ.mean().item():.4f}")
                print(f"[DEBUG] Non-zero pixels in target: {(targ > 0.1).sum().item()}/{targ.numel()}")
                print(f"[DEBUG] Delta has gradient: {model.delta_t.grad is not None}")
                if model.delta_t.grad is not None:
                    print(f"[DEBUG] Delta gradient: {model.delta_t.grad.cpu().numpy()}")

            print(f"[{it+1:04d}/{args.iters}] loss={loss.item():.6f}  delta=({dt[0]:.5f},{dt[1]:.5f},{dt[2]:.5f})")

            # Save intermediate URDF with current delta
            urdf_path_iter = os.path.join(output_dir, f"iter_{it+1:04d}.urdf")
            update_urdf_joint_and_child(
                urdf_path=args.urdf,
                joint_name=joint_name,
                child_link=child_link,
                delta=dt,
                out_path=urdf_path_iter,
                silent=True  # Suppress "[OK] Wrote..." message for intermediate saves
            )
            print(f"[INFO] Saved intermediate URDF to: {urdf_path_iter}")

            # Save RGB rendered image
            rgb_np = images[0, ..., :3].detach().cpu().numpy()  # [H, W, 3]
            rgb_img = (rgb_np * 255).astype(np.uint8)
            img_path_rgb = os.path.join(output_dir, f"iter_{it+1:04d}_rgb.png")
            Image.fromarray(rgb_img, mode='RGB').save(img_path_rgb)

            # Save silhouette (alpha channel)
            sil_np = sil.detach().cpu().numpy()
            sil_img = (sil_np * 255).astype(np.uint8)
            img_path_sil = os.path.join(output_dir, f"iter_{it+1:04d}_silhouette.png")
            Image.fromarray(sil_img, mode='L').save(img_path_sil)

            # Also save a thresholded binary version for clearer visualization
            sil_binary = (sil_np > 0.5).astype(np.uint8) * 255
            img_path_binary = os.path.join(output_dir, f"iter_{it+1:04d}_binary.png")
            Image.fromarray(sil_binary, mode='L').save(img_path_binary)

            print(f"[INFO] Saved: {img_path_rgb} (RGB), {img_path_sil} (silhouette), {img_path_binary} (binary)")

    # Write back URDF
    delta = model.delta_t.detach().cpu().numpy()
    out_urdf = args.out_urdf if args.out_urdf is not None else os.path.splitext(args.urdf)[0] + "_optimized.urdf"
    update_urdf_joint_and_child(
        urdf_path=args.urdf,
        joint_name=joint_name,
        child_link=child_link,
        delta=delta,
        out_path=out_urdf
    )

    print("[DONE] Optimization complete.")
    print("Recommended next steps:")
    print("  1) Visually inspect the new URDF in a viewer.")
    print("  2) (Optional) Re-run with a better binary mask (--target-mask) instead of the white-background heuristic.")
    print("  3) Adjust camera params if alignment differs from your photo (same viewpoint assumed).")


if __name__ == "__main__":
    main()
