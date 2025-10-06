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
        look_at_view_transform, FoVPerspectiveCameras, PerspectiveCameras, RasterizationSettings,
        MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, TexturesVertex,
        DirectionalLights, Materials, BlendParams
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

def rpy_to_matrix_np(rpy: Tuple[float, float, float]) -> np.ndarray:
    """Convert roll-pitch-yaw (XYZ fixed angles) to 3x3 rotation matrix (numpy version)."""
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


def rpy_to_matrix(rpy: Tuple[float, float, float]) -> torch.Tensor:
    """Convert roll-pitch-yaw (XYZ fixed angles) to 3x3 rotation matrix (torch version)."""
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
      - joint/@origin xyz += delta (in parent frame)
      - child_link/visual/@origin xyz -= Rj^T @ delta (in joint frame)
      - child_link/collision/@origin xyz -= Rj^T @ delta (in joint frame)

    where Rj is the rotation of the joint origin (from rpy).
    This ensures angle=0 pose remains unchanged.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joint = root.find(f"./joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Joint '{joint_name}' not found in URDF when writing back.")

    jorigin = joint.find('origin')
    if jorigin is None:
        jorigin = ET.SubElement(joint, 'origin')

    # Get joint origin rotation (rpy)
    cur_rpy = _parse_rpy(jorigin.attrib.get('rpy'))

    # Compute Rj matrix from rpy (numpy version)
    Rj = rpy_to_matrix_np(tuple(cur_rpy))

    # Transform delta from parent frame to joint frame
    delta_in_joint_frame = Rj.T @ delta

    # Update joint origin xyz
    cur_xyz = _parse_xyz(jorigin.attrib.get('xyz'))
    new_xyz = cur_xyz + delta
    jorigin.attrib['xyz'] = f"{new_xyz[0]:.8f} {new_xyz[1]:.8f} {new_xyz[2]:.8f}"

    # child visual - subtract delta in joint frame
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
            new_vxyz = cur_vxyz - delta_in_joint_frame
            vorigin.attrib['xyz'] = f"{new_vxyz[0]:.8f} {new_vxyz[1]:.8f} {new_vxyz[2]:.8f}"

        # collisions too
        for cori in get_child_collision_origins(root, child_link):
            cur_cxyz = _parse_xyz(cori.attrib.get('xyz'))
            new_cxyz = cur_cxyz - delta_in_joint_frame
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


def load_mesh_as_tensors(mesh_path: str, scale: Optional[np.ndarray], unit_scale: float, device: torch.device):
    """
    Load mesh (GLB, OBJ, etc.) using trimesh, return (verts: Nx3 torch, faces: Fx3 long).
    Applies scale from URDF and unit_scale.
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

    # Compute bounding box after all transformations
    v_min = V.min(dim=0)[0].cpu().numpy()
    v_max = V.max(dim=0)[0].cpu().numpy()
    print(f"[DEBUG] Mesh bbox after all transforms: min={v_min}, max={v_max}")

    return V, F


def compute_scene_bounds_raw_mesh(parent_V: torch.Tensor, child_V: torch.Tensor) -> float:
    """
    Compute the bounding box extent of the raw mesh geometry (without URDF transforms).
    This is used to automatically set camera distance based on the original mesh size.
    Returns max_extent for auto camera positioning.
    """
    # Combine raw mesh vertices without any URDF transforms
    all_verts = torch.cat([parent_V, child_V], dim=0)

    # Compute bounding box
    bbox_min = all_verts.min(dim=0)[0]
    bbox_max = all_verts.max(dim=0)[0]
    extent = (bbox_max - bbox_min).max().item()

    return extent


# ----------------------
# Rendering + loss
# ----------------------

def build_silhouette_renderer(image_size: int, device: torch.device, fov: float = 40.0, radius: float = 4.0):
    """
    Build soft silhouette renderer for differentiable optimization.

    Uses SoftSilhouetteShader with carefully tuned parameters for:
    - Clean interior (solid white/black, no artifacts)
    - Soft edges only at true boundary (2-4px gradient band)
    - Smooth gradients for optimization

    Camera setup:
    - Fixed at (0, 0, radius) looking at origin
    - FOV: 40 degrees (matching render_notexture.py)
    - radius: 4 (matching RADIUS in render_notexture.py)

    Rasterization:
    - faces_per_pixel=50 for proper occlusion handling
    - sigma=1e-5, gamma=1e-6 for sharp interior, soft boundary only
    - cull_backfaces=True to prevent internal edges showing
    - No lighting/materials needed (silhouette only uses alpha)

    Background: White (1.0, 1.0, 1.0) matching GT extraction
    """
    # Camera setup - same as before
    fov_rad = math.radians(fov)
    focal_length = image_size / (2.0 * math.tan(fov_rad / 2.0))
    R, T = look_at_view_transform(dist=radius, elev=0.0, azim=0.0, device=device)

    cameras = PerspectiveCameras(
        device=device,
        R=R,
        T=T,
        focal_length=((focal_length, focal_length),),
        principal_point=((image_size / 2.0, image_size / 2.0),),
        image_size=((image_size, image_size),),
        in_ndc=False
    )

    # Soft rasterization parameters (key for differentiability)
    # Adjusted for clean silhouettes: only edges should be soft, interior solid white
    faces_per_pixel = 50  # More faces to handle occlusion properly
    sigma = 1e-5  # Smaller sigma = sharper interior, soft only at true boundary
    blur_radius = np.log(1.0 / 1e-4 - 1.0) * sigma  # Narrow blur band

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=True,  # Cull backfaces to avoid internal edges showing
        perspective_correct=True,  # More accurate rasterization
        bin_size=None
    )

    # Blend parameters for soft silhouette
    # Smaller gamma = cleaner interior (less blending of overlapping faces)
    blend_params = BlendParams(
        sigma=sigma,
        gamma=1e-6,  # Very small gamma for clean interior
        background_color=(1.0, 1.0, 1.0)  # White background matching GT
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params)
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


def compute_silhouette_loss(sil: torch.Tensor, targ: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute combined loss for silhouette optimization.

    Args:
        sil: Rendered silhouette [H, W] in [0, 1]
        targ: Target silhouette [1, H, W] in [0, 1]
        device: torch device

    Returns:
        Combined loss (BCE + Dice + Edge)
    """
    import torch.nn.functional as F

    eps = 1e-6
    targ_2d = targ.squeeze(0)

    # 1. Binary Cross Entropy - stable pixel-wise loss
    bce = F.binary_cross_entropy(sil, targ_2d, reduction='mean')

    # 2. Soft Dice Loss - handles class imbalance (sparse foreground)
    intersection = (sil * targ_2d).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (sil.sum() + targ_2d.sum() + eps)

    # 3. Edge Loss - emphasize boundary matching (Sobel gradients)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=device, dtype=sil.dtype).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      device=device, dtype=sil.dtype).view(1, 1, 3, 3) / 8.0

    def edge_map(x):
        x4d = x.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        gx = F.conv2d(x4d, kx, padding=1)
        gy = F.conv2d(x4d, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + eps).squeeze(0).squeeze(0)

    edge = F.l1_loss(edge_map(sil), edge_map(targ_2d))

    # Combined loss with tuned weights
    loss = 1.0 * bce + 0.5 * dice + 0.2 * edge

    return loss


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
                 device: torch.device,
                 initial_delta_t: Optional[torch.Tensor] = None):
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
        self.initial_angle = torch.tensor(target_angle_rad, dtype=torch.float32, device=device)

        # Learnable parameters
        if initial_delta_t is not None:
            self.delta_t = nn.Parameter(initial_delta_t.clone())
        else:
            self.delta_t = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
        self.delta_angle = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))

        # Compute normalization parameters based on angle=0 reference pose
        # This matches the normalization in render_urdf_angles.py
        print("[INFO] Computing normalization parameters from angle=0 reference pose...")
        with torch.no_grad():
            # Build reference mesh at angle=0 (same as forward() but with angle=0)
            Tp = make_SE3(self.Rp, self.parent_vis_t)
            Vp_w = transform_points_h(Tp, self.parent_V)

            # Joint at initial position (no delta)
            Tj = make_SE3(self.Rj, self.joint_origin_t0)

            # No rotation (angle=0)
            Rrot = torch.eye(3, dtype=torch.float32, device=device)
            Trot = make_SE3(Rrot, torch.zeros(3, dtype=torch.float32, device=device))

            # Child visual at initial position
            Tc = make_SE3(self.Rc, self.child_vis_t0)

            # Compute child world position
            Tw = Tp @ Tj @ Trot @ Tc
            Vc_w = transform_points_h(Tw, self.child_V)

            # Merge vertices to compute bbox
            V_all = torch.cat([Vp_w, Vc_w], dim=0)

            # Compute bbox (same as trimesh.bounding_box)
            bbox_min = V_all.min(dim=0)[0]
            bbox_max = V_all.max(dim=0)[0]
            bbox_center = (bbox_min + bbox_max) / 2.0
            bbox_extents = bbox_max - bbox_min
            bbox_max_extent = bbox_extents.max()

            # Normalization parameters (matching normalize_mesh in src/utils/data_utils.py)
            self.norm_translation = -bbox_center  # translation = -bbox.centroid
            self.norm_scale = 2.0 / bbox_max_extent  # scale = 2.0 / bbox.extents.max()

            print(f"[INFO] Normalization (matching normalize_mesh):")
            print(f"  Translation: ({self.norm_translation[0]:.4f}, {self.norm_translation[1]:.4f}, {self.norm_translation[2]:.4f})")
            print(f"  Scale: {self.norm_scale:.4f}")

    def forward(self, angle: Optional[torch.Tensor] = None, use_learned_angle: bool = True) -> Meshes:
        """
        Build world-space merged mesh (parent static + child rotated about joint pivot with delta shift).

        Args:
            angle: Optional fixed joint angle (radians). If None, uses learned angle.
            use_learned_angle: If True and angle is None, use initial_angle + delta_angle.
                               If False and angle is None, just use initial_angle.

        Transform chain for child:
        world = parent_visual @ joint_origin @ joint_rotation @ child_visual @ local_vertices

        This ensures the joint is defined in the parent's frame, not world frame.
        """
        # Determine which angle to use
        if angle is None:
            if use_learned_angle:
                angle = self.initial_angle + self.delta_angle.squeeze()
            else:
                angle = self.initial_angle

        # Parent world = apply parent visual transform
        Tp = make_SE3(self.Rp, self.parent_vis_t)
        Vp_w = transform_points_h(Tp, self.parent_V)

        # Joint origin: R_origin, t_origin + delta
        # Use precomputed Rj, only tj changes with delta_t (learnable)
        tj = self.joint_origin_t0 + self.delta_t
        Tj = make_SE3(self.Rj, tj)

        # Rotation about joint axis by specified angle
        Rrot = axis_angle_to_matrix(self.axis, angle).to(self.device)
        Trot = make_SE3(Rrot, torch.zeros(3, dtype=torch.float32, device=self.device))

        # Child visual local: must compensate for delta_t in joint's coordinate frame
        # delta_t is in parent frame, but child visual is in joint frame (after Rj rotation)
        # So we need: child_visual_offset = child_vis_t0 - Rj^T @ delta_t
        delta_in_joint_frame = self.Rj.T @ self.delta_t
        tc = self.child_vis_t0 - delta_in_joint_frame
        Tc = make_SE3(self.Rc, tc)

        # World transform for child: T_world = Tp @ Tj @ Trot @ Tc
        # CRITICAL: Include Tp so joint is in parent's frame, not world frame
        Tw = Tp @ Tj @ Trot @ Tc

        Vc_w = transform_points_h(Tw, self.child_V)

        # Merge
        V = torch.cat([Vp_w, Vc_w], dim=0)
        F_child = self.child_F + self.parent_V.shape[0]
        F = torch.cat([self.parent_F, F_child], dim=0)

        # Apply normalization (matching normalize_mesh in src/utils/data_utils.py)
        # This ensures camera view matches render_urdf_angles.py
        V = V + self.norm_translation.unsqueeze(0)  # translate
        V = V * self.norm_scale  # scale

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
    parser.add_argument("--target-angle-deg", type=float, default=35.0, help="Initial angle to open the joint (degrees). Used as starting point if --learn-angle is enabled.")
    parser.add_argument("--learn-angle", action="store_true", help="Learn the opened joint angle instead of using fixed --target-angle-deg.")
    parser.add_argument("--angle-bounds", type=str, default="35,180", help="Min and max angle bounds in degrees for learned angle (e.g., '0,90').")
    parser.add_argument("--iters", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for Adam (pivot position).")
    parser.add_argument("--lr-angle", type=float, default=1e-2, help="Learning rate for joint angle (if --learn-angle is enabled).")
    parser.add_argument("--closed-weight", type=float, default=0.2, help="Weight for closed-state silhouette constraint (0 to disable).")
    parser.add_argument("--regularization", type=float, default=1e-3, help="L2 regularization on delta_t to prevent drift (0 to disable).")
    parser.add_argument("--angle-regularization", type=float, default=1e-4, help="L2 regularization on delta_angle to prefer initial angle (0 to disable).")
    parser.add_argument("--constrain-to-child", action="store_true", help="Constrain pivot to be inside child mesh bounding box.")
    parser.add_argument("--constrain-to-parent", action="store_true", help="Constrain pivot to be inside parent mesh geometry (using tight bbox to avoid holes).")
    parser.add_argument("--init-random", action="store_true", help="Initialize pivot at random position inside child bbox (requires --constrain-to-child).")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for initialization (if --init-random is used).")
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
        Vp, Fp = load_mesh_as_tensors(parent_mesh_path, parent_scale, args.unit_scale, device)

    if child_mesh_path is None:
        raise ValueError("Child link has no visual mesh, cannot proceed. Provide a mesh in the URDF visual.")

    print(f"\n[DEBUG] Loading child mesh:")
    Vc, Fc = load_mesh_as_tensors(child_mesh_path, child_scale, args.unit_scale, device)

    print(f"\n[DEBUG] Parent mesh: {Vp.shape[0]} verts, {Fp.shape[0]} faces")
    print(f"[DEBUG] Child mesh: {Vc.shape[0]} verts, {Fc.shape[0]} faces")

    # Compute parent mesh bounding box (in parent's local coordinate frame)
    # For tight bbox to avoid holes, we use axis-aligned bbox of all vertices
    if parent_mesh_path is not None and Vp.shape[0] > 3:
        parent_bbox_min = Vp.min(dim=0)[0].cpu().numpy()
        parent_bbox_max = Vp.max(dim=0)[0].cpu().numpy()
        parent_bbox_center = (parent_bbox_min + parent_bbox_max) / 2.0
        parent_bbox_size = parent_bbox_max - parent_bbox_min

        print(f"\n[DEBUG] ===== Parent Mesh Bounding Box =====")
        print(f"[DEBUG] BBox min: ({parent_bbox_min[0]:.4f}, {parent_bbox_min[1]:.4f}, {parent_bbox_min[2]:.4f})")
        print(f"[DEBUG] BBox max: ({parent_bbox_max[0]:.4f}, {parent_bbox_max[1]:.4f}, {parent_bbox_max[2]:.4f})")
        print(f"[DEBUG] BBox center: ({parent_bbox_center[0]:.4f}, {parent_bbox_center[1]:.4f}, {parent_bbox_center[2]:.4f})")
        print(f"[DEBUG] BBox size: ({parent_bbox_size[0]:.4f}, {parent_bbox_size[1]:.4f}, {parent_bbox_size[2]:.4f})")
    else:
        parent_bbox_min = None
        parent_bbox_max = None

    # Compute child mesh bounding box (in child's local coordinate frame)
    # This will be used to constrain pivot optimization
    child_bbox_min = Vc.min(dim=0)[0].cpu().numpy()
    child_bbox_max = Vc.max(dim=0)[0].cpu().numpy()
    child_bbox_center = (child_bbox_min + child_bbox_max) / 2.0
    child_bbox_size = child_bbox_max - child_bbox_min

    print(f"\n[DEBUG] ===== Child Mesh Bounding Box =====")
    print(f"[DEBUG] BBox min: ({child_bbox_min[0]:.4f}, {child_bbox_min[1]:.4f}, {child_bbox_min[2]:.4f})")
    print(f"[DEBUG] BBox max: ({child_bbox_max[0]:.4f}, {child_bbox_max[1]:.4f}, {child_bbox_max[2]:.4f})")
    print(f"[DEBUG] BBox center: ({child_bbox_center[0]:.4f}, {child_bbox_center[1]:.4f}, {child_bbox_center[2]:.4f})")
    print(f"[DEBUG] BBox size: ({child_bbox_size[0]:.4f}, {child_bbox_size[1]:.4f}, {child_bbox_size[2]:.4f})")

    # Camera and rendering setup
    print(f"\n[DEBUG] ===== Camera & Rendering Setup =====")
    print(f"[INFO] Using soft silhouette rendering for differentiable optimization:")
    print(f"[INFO]   - Camera: (0, 0, 4) looking at origin")
    print(f"[INFO]   - Radius: 4.0")
    print(f"[INFO]   - FOV: 40.0 degrees")
    print(f"[INFO]   - Renderer: SoftSilhouetteShader")
    print(f"[INFO]   - Rasterization: faces_per_pixel=50, sigma=1e-5, gamma=1e-6")
    print(f"[INFO]   - Backface culling: enabled (clean interior)")
    print(f"[INFO]   - Background: white (1.0, 1.0, 1.0)")
    print(f"[INFO]   - Loss: BCE + Dice + Edge (combined)")
    print(f"[INFO]   - Closed-state constraint: {args.closed_weight if args.closed_weight > 0 else 'disabled'}")
    print(f"[INFO]   - Regularization: {args.regularization if args.regularization > 0 else 'disabled'}")

    # Target angle for optimization
    target_angle_rad = math.radians(args.target_angle_deg)
    print(f"[DEBUG] Target angle for optimization: {args.target_angle_deg} deg = {target_angle_rad:.4f} rad")

    # Initialize pivot position (delta_t)
    if args.init_random:
        if not args.constrain_to_child:
            print("[WARN] --init-random requires --constrain-to-child, enabling constraint.")
            args.constrain_to_child = True

        # Set random seed for reproducibility
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

        # Sample random point inside child bbox (in child's local frame)
        random_point = np.random.uniform(child_bbox_min, child_bbox_max).astype(np.float32)

        # Transform to joint frame: need to account for child_visual origin
        # The bbox is in child mesh's local coords, but joint is defined relative to child visual
        # So initial_delta = random_point - child_vis_xyz (both in child's coords)
        # But we need it in joint frame, which is child's parent frame
        # Actually, j_xyz is already in parent frame, so we initialize delta_t relative to current j_xyz

        # For simplicity: random offset within bbox, then transform to parent frame if needed
        # Initial delta in child's local frame
        initial_delta_local = random_point - child_vis_xyz

        # Transform to parent frame through joint rotation
        Rj_np = rpy_to_matrix_np(tuple(j_rpy))
        initial_delta_parent = Rj_np @ initial_delta_local

        initial_delta_t = torch.tensor(initial_delta_parent, dtype=torch.float32, device=device)
        print(f"[INFO] Random initialization inside child bbox:")
        print(f"[INFO]   - Random point (child local): ({random_point[0]:.4f}, {random_point[1]:.4f}, {random_point[2]:.4f})")
        print(f"[INFO]   - Initial delta_t (parent frame): ({initial_delta_t[0]:.4f}, {initial_delta_t[1]:.4f}, {initial_delta_t[2]:.4f})")
    else:
        initial_delta_t = None
        if args.constrain_to_child:
            print(f"[INFO] Pivot constrained to child bbox, starting from joint origin (delta_t=0)")
        else:
            print(f"[INFO] No bbox constraint, starting from joint origin (delta_t=0)")

    # Build optimizer model
    model = SingleRevoluteOptimizer(
        parent_V=Vp, parent_F=Fp,
        child_V=Vc, child_F=Fc,
        joint_origin_xyz=j_xyz, joint_origin_rpy=j_rpy,
        joint_axis=jaxis,
        parent_vis_xyz=parent_vis_xyz, parent_vis_rpy=parent_vis_rpy,
        child_vis_xyz=child_vis_xyz, child_vis_rpy=child_vis_rpy,
        target_angle_rad=target_angle_rad,
        device=device,
        initial_delta_t=initial_delta_t  # Pass initial value
    ).to(device)

    # Build soft silhouette renderer for differentiable optimization
    renderer, cameras = build_silhouette_renderer(
        image_size=args.image_size,
        device=device,
        fov=40.0,  # Fixed FOV matching render_notexture.py
        radius=4.0  # Fixed radius matching render_notexture.py
    )

    # Target silhouettes (opened and closed states)
    print(f"\n[DEBUG] ===== Target Image Loading =====")

    # Load opened-state target
    if args.target_mask is not None:
        print(f"[DEBUG] Loading opened target mask from: {args.target_mask}")
        # If a mask is provided (0/255), load directly
        m = Image.open(args.target_mask).convert("L").resize((args.image_size, args.image_size), Image.Resampling.NEAREST)
        targ_opened = (torch.from_numpy(np.asarray(m, dtype=np.float32)) / 255.0).unsqueeze(0)
        targ_opened = (targ_opened < 0.5).float()  # black -> 1, white -> 0 (invert if needed)
    else:
        print(f"[DEBUG] Extracting silhouette from opened image: {args.opened_img}")
        print(f"[DEBUG] Using threshold: {args.threshold}")
        targ_opened = image_to_silhouette(args.opened_img, args.image_size, threshold=args.threshold)
    targ_opened = targ_opened.to(device)
    print(f"[DEBUG] Opened target loaded, shape: {targ_opened.shape}")
    print(f"[DEBUG] Opened min/max/mean: {targ_opened.min().item():.4f}/{targ_opened.max().item():.4f}/{targ_opened.mean().item():.4f}")

    # Load closed-state target (if constraint is enabled)
    targ_closed = None
    if args.closed_weight > 0 and args.closed_img:
        print(f"[DEBUG] Loading closed-state constraint from: {args.closed_img}")
        targ_closed = image_to_silhouette(args.closed_img, args.image_size, threshold=args.threshold)
        targ_closed = targ_closed.to(device)
        print(f"[DEBUG] Closed target loaded, shape: {targ_closed.shape}")
        print(f"[DEBUG] Closed constraint weight: {args.closed_weight}")
    else:
        print(f"[DEBUG] Closed-state constraint disabled (weight={args.closed_weight})")

    # Optimizer - separate learning rates for position and angle
    if args.learn_angle:
        optim = torch.optim.Adam([
            {'params': [model.delta_t], 'lr': args.lr},
            {'params': [model.delta_angle], 'lr': args.lr_angle}
        ])
        print(f"[INFO] Optimizing both pivot position (lr={args.lr}) and angle (lr={args.lr_angle})")

        # Parse angle bounds
        angle_min, angle_max = map(float, args.angle_bounds.split(','))
        angle_min_rad = math.radians(angle_min)
        angle_max_rad = math.radians(angle_max)
        print(f"[INFO] Angle bounds: [{angle_min}, {angle_max}] degrees = [{angle_min_rad:.4f}, {angle_max_rad:.4f}] rad")
    else:
        optim = torch.optim.Adam([model.delta_t], lr=args.lr)
        print(f"[INFO] Optimizing pivot position only (lr={args.lr}), angle fixed at {args.target_angle_deg} degrees")

    # Create output directory for rendered images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"[INFO] Output directory created: {output_dir}")

    # Optimization loop
    print(f"\n[DEBUG] ===== Starting Optimization =====")
    print(f"[INFO] Regularization weight: {args.regularization}")

    for it in range(args.iters):
        optim.zero_grad()

        # Render opened state (target angle)
        meshes_opened = model()  # Uses default target_angle
        images_opened = renderer(meshes_opened)
        sil_opened = images_opened[..., 3].squeeze(0)  # HxW
        sil_opened = torch.clamp(sil_opened, 0.0, 1.0)

        # Primary loss: opened-state silhouette matching
        loss_opened = compute_silhouette_loss(sil_opened, targ_opened, device)
        loss = loss_opened

        # Closed-state constraint (if enabled)
        loss_closed = torch.tensor(0.0, device=device)
        if targ_closed is not None and args.closed_weight > 0:
            # Render at angle=0 (closed state)
            angle_closed = torch.tensor(0.0, dtype=torch.float32, device=device)
            meshes_closed = model(angle=angle_closed)
            images_closed = renderer(meshes_closed)
            sil_closed = images_closed[..., 3].squeeze(0)
            sil_closed = torch.clamp(sil_closed, 0.0, 1.0)

            loss_closed = compute_silhouette_loss(sil_closed, targ_closed, device)
            loss = loss + args.closed_weight * loss_closed

        # L2 regularization on delta_t (prevent drift)
        loss_reg_t = torch.tensor(0.0, device=device)
        if args.regularization > 0:
            loss_reg_t = args.regularization * torch.sum(model.delta_t ** 2)
            loss = loss + loss_reg_t

        # L2 regularization on delta_angle (prefer initial angle)
        loss_reg_angle = torch.tensor(0.0, device=device)
        if args.learn_angle and args.angle_regularization > 0:
            loss_reg_angle = args.angle_regularization * torch.sum(model.delta_angle ** 2)
            loss = loss + loss_reg_angle

        loss.backward()
        optim.step()

        # Project delta_t onto child bbox (if constrained)
        if args.constrain_to_child:
            with torch.no_grad():
                # Current joint origin in parent frame
                current_joint_xyz = model.joint_origin_t0 + model.delta_t

                # Transform current joint position to child's local frame
                # We need to inverse the transform: child_local = Rj^T @ (joint - child_vis_offset)
                # But this gets complex with all the transforms...
                # Simpler: constrain delta_t such that the pivot stays in bbox

                # The pivot in child's local frame is:
                # pivot_child = Rj^T @ (joint_xyz + delta_t) - child_vis_xyz
                # We want: child_bbox_min <= pivot_child <= child_bbox_max

                # For now, use a simpler projection: constrain delta_t magnitude
                # Or project current joint position into bbox in child frame

                # Transform to child frame
                Rj_inv = model.Rj.T
                pivot_in_child = Rj_inv @ model.delta_t

                # Clamp to bbox (in child's local frame, relative to child visual origin)
                bbox_min_torch = torch.tensor(child_bbox_min - child_vis_xyz, dtype=torch.float32, device=device)
                bbox_max_torch = torch.tensor(child_bbox_max - child_vis_xyz, dtype=torch.float32, device=device)

                clamped_pivot = torch.clamp(pivot_in_child, bbox_min_torch, bbox_max_torch)

                # Transform back to parent frame
                model.delta_t.data = model.Rj @ clamped_pivot

        # Project delta_t onto parent bbox (if constrained)
        if args.constrain_to_parent:
            if parent_bbox_min is None or parent_bbox_max is None:
                print("[WARN] Parent bbox not available, skipping parent constraint.")
            else:
                with torch.no_grad():
                    # Current joint origin in parent frame
                    # The joint is at: joint_origin_t0 + delta_t
                    # This is already in parent's coordinate frame
                    current_joint_xyz = model.joint_origin_t0 + model.delta_t

                    # Parent visual origin offset (from parent visual tag)
                    parent_vis_xyz_tensor = torch.tensor(parent_vis_xyz, dtype=torch.float32, device=device)

                    # Joint position relative to parent visual origin
                    # joint_relative = current_joint - parent_vis_xyz
                    joint_relative = current_joint_xyz - parent_vis_xyz_tensor

                    # Clamp to parent bbox (in parent's local frame)
                    parent_bbox_min_torch = torch.tensor(parent_bbox_min, dtype=torch.float32, device=device)
                    parent_bbox_max_torch = torch.tensor(parent_bbox_max, dtype=torch.float32, device=device)

                    clamped_joint_relative = torch.clamp(joint_relative, parent_bbox_min_torch, parent_bbox_max_torch)

                    # Convert back to delta_t
                    clamped_joint_xyz = clamped_joint_relative + parent_vis_xyz_tensor
                    model.delta_t.data = clamped_joint_xyz - model.joint_origin_t0

        # Clamp learned angle to bounds
        if args.learn_angle:
            with torch.no_grad():
                current_angle = model.initial_angle + model.delta_angle.squeeze()
                # Clamp to bounds
                clamped_angle = torch.clamp(current_angle, angle_min_rad, angle_max_rad)
                # Update delta_angle to maintain clamped total
                model.delta_angle.data = (clamped_angle - model.initial_angle).unsqueeze(0)

        if (it+1) % max(1, args.iters // 10) == 0 or it == 0:
            dt = model.delta_t.detach().cpu().numpy()

            # Debug info for first iteration
            if it == 0:
                print(f"\n[DEBUG] First iteration diagnostics:")
                print(f"[DEBUG] Opened silhouette shape: {sil_opened.shape}")
                print(f"[DEBUG] Silhouette min/max: {sil_opened.min().item():.4f}/{sil_opened.max().item():.4f}")
                print(f"[DEBUG] Silhouette mean: {sil_opened.mean().item():.4f}")
                print(f"[DEBUG] Non-zero pixels: {(sil_opened > 0.1).sum().item()}/{sil_opened.numel()}")
                print(f"[DEBUG] Delta has gradient: {model.delta_t.grad is not None}")
                if model.delta_t.grad is not None:
                    print(f"[DEBUG] Delta gradient: {model.delta_t.grad.cpu().numpy()}")

            # Print loss breakdown
            loss_str = f"[{it+1:04d}/{args.iters}] total={loss.item():.6f} (open={loss_opened.item():.6f}"
            if targ_closed is not None and args.closed_weight > 0:
                loss_str += f", closed={loss_closed.item():.6f}"
            if args.regularization > 0:
                loss_str += f", reg_t={loss_reg_t.item():.6f}"
            if args.learn_angle and args.angle_regularization > 0:
                loss_str += f", reg_a={loss_reg_angle.item():.6f}"
            loss_str += f")  delta_t=({dt[0]:.5f},{dt[1]:.5f},{dt[2]:.5f})"

            # Add angle info if learning
            if args.learn_angle:
                da = model.delta_angle.detach().cpu().item()
                current_angle = model.initial_angle.item() + da
                current_angle_deg = math.degrees(current_angle)
                loss_str += f"  angle={current_angle_deg:.2f}° (Δ={math.degrees(da):.2f}°)"

            print(loss_str)

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

            # Save opened-state silhouette (primary optimization target)
            sil_np = sil_opened.detach().cpu().numpy()
            sil_img = (sil_np * 255).astype(np.uint8)
            img_path_sil = os.path.join(output_dir, f"iter_{it+1:04d}_opened.png")
            Image.fromarray(sil_img, mode='L').save(img_path_sil)
            print(f"[INFO] Saved opened silhouette: {img_path_sil}")

            # Optionally save closed-state silhouette for debugging
            if targ_closed is not None and args.closed_weight > 0:
                sil_closed_np = sil_closed.detach().cpu().numpy()
                sil_closed_img = (sil_closed_np * 255).astype(np.uint8)
                img_path_closed = os.path.join(output_dir, f"iter_{it+1:04d}_closed.png")
                Image.fromarray(sil_closed_img, mode='L').save(img_path_closed)
                print(f"[INFO] Saved closed silhouette: {img_path_closed}")

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

    print("\n[DONE] Optimization complete.")
    print(f"Final pivot delta: ({delta[0]:.5f}, {delta[1]:.5f}, {delta[2]:.5f})")

    if args.learn_angle:
        final_delta_angle = model.delta_angle.detach().cpu().item()
        final_angle_rad = model.initial_angle.item() + final_delta_angle
        final_angle_deg = math.degrees(final_angle_rad)
        print(f"Final learned angle: {final_angle_deg:.2f}° (initial: {args.target_angle_deg:.2f}°, delta: {math.degrees(final_delta_angle):.2f}°)")
        print(f"NOTE: The learned angle ({final_angle_deg:.2f}°) is NOT written to the URDF.")
        print(f"      You may want to manually update the joint limits or use this angle for visualization.")

    print("\nRecommended next steps:")
    print("  1) Visually inspect the new URDF in a viewer.")
    print("  2) (Optional) Re-run with a better binary mask (--target-mask) instead of the white-background heuristic.")
    print("  3) Check intermediate renderings in output/ to verify convergence.")


if __name__ == "__main__":
    main()
