#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check camera settings by rendering a single view without optimization.
This script uses the same camera and rendering setup as optimize_urdf_joint_v2.py
to verify that the camera settings match render.py.
"""

import os
import math
import argparse
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
import numpy as np
from PIL import Image

import torch

# PyTorch3D imports
try:
    from pytorch3d.renderer import (
        look_at_view_transform, PerspectiveCameras, RasterizationSettings,
        MeshRenderer, MeshRasterizer, HardPhongShader, TexturesVertex,
        PointLights, Materials
    )
    from pytorch3d.structures import Meshes
except Exception as e:
    raise RuntimeError("PyTorch3D is required. Install it first. Error: {}".format(e))

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
# URDF parsing
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


# ------------------------------------
# Mesh loading
# ------------------------------------

def resolve_mesh_path(mesh_path: str, urdf_dir: str) -> str:
    if mesh_path is None:
        return None
    if mesh_path.startswith("package://"):
        mesh_path = mesh_path.replace("package://", "")
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(urdf_dir, mesh_path)
    return mesh_path


def load_mesh_as_tensors(mesh_path: str, scale: Optional[np.ndarray], unit_scale: float, device: torch.device,
                         normalize: bool = False, normalize_scale: float = 2.0):
    """
    Load mesh using trimesh, return (verts: Nx3 torch, faces: Fx3 long).
    """
    if mesh_path is None:
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

    # Apply URDF scaling first
    if scale is not None:
        if np.isscalar(scale):
            V = V * float(scale)
            print(f"[DEBUG] Applied isotropic scale: {scale}")
        else:
            V = V * torch.tensor(scale, dtype=torch.float32, device=device)
            print(f"[DEBUG] Applied anisotropic scale: {scale}")
    if unit_scale != 1.0:
        V = V * float(unit_scale)
        print(f"[DEBUG] Applied unit scale: {unit_scale}")

    # Apply normalization if requested (matching preprocess)
    if normalize:
        centroid = V.mean(dim=0)
        V = V - centroid
        print(f"[DEBUG] Centered mesh at origin (centroid: {centroid.cpu().numpy()})")

        bbox_min = V.min(dim=0)[0]
        bbox_max = V.max(dim=0)[0]
        max_extent = (bbox_max - bbox_min).max().item()
        scale_factor = normalize_scale / max_extent
        V = V * scale_factor
        print(f"[DEBUG] Normalized mesh: max_extent={max_extent:.4f}, scale_factor={scale_factor:.4f}, target_scale={normalize_scale}")

    v_min = V.min(dim=0)[0].cpu().numpy()
    v_max = V.max(dim=0)[0].cpu().numpy()
    print(f"[DEBUG] Mesh bbox after all transforms: min={v_min}, max={v_max}")

    return V, F


# ----------------------
# Rendering
# ----------------------

def build_phong_renderer(image_size: int, device: torch.device, fov: float,
                         cam_dist: float, cam_elev: float, cam_azim: float,
                         at: Tuple[float, float, float] = ((0.0, 0.0, 0.0),)):
    """
    Build Phong renderer with camera and lighting setup similar to render_utils.py.

    Args:
        fov: Vertical field of view in degrees (matching pyrender's yfov)
        at: Camera look-at point (x, y, z). Default is origin.
    """
    R, T = look_at_view_transform(dist=cam_dist, elev=cam_elev, azim=cam_azim, at=at, device=device)

    # Convert vertical FOV to focal length to match pyrender exactly
    fov_rad = math.radians(fov)
    focal_length = image_size / (2.0 * math.tan(fov_rad / 2.0))

    # Use PerspectiveCameras with focal_length for exact match with pyrender
    cameras = PerspectiveCameras(
        device=device,
        R=R,
        T=T,
        focal_length=((focal_length, focal_length),),
        principal_point=((image_size / 2.0, image_size / 2.0),),
        image_size=((image_size, image_size),),
        in_ndc=False  # Use screen space coordinates
    )

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None
    )

    # Lighting setup
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 0.0]],
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


def main():
    parser = argparse.ArgumentParser(description="Check camera settings by rendering URDF mesh.")
    parser.add_argument("--urdf", type=str, default="input/mobility.urdf", help="Path to URDF file.")
    parser.add_argument("--joint-name", type=str, default=None, help="Name of the revolute joint (optional).")
    parser.add_argument("--joint-angle-deg", type=float, default=0.0, help="Joint angle in degrees for visualization.")
    parser.add_argument("--image-size", type=int, default=1024, help="Render size.")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale meshes by this factor.")
    parser.add_argument("--fov", type=float, default=40.0, help="Vertical FOV in degrees.")
    parser.add_argument("--cam-dist", type=float, default=4.0, help="Camera distance.")
    parser.add_argument("--cam-elev", type=float, default=0.0, help="Camera elevation in degrees.")
    parser.add_argument("--cam-azim", type=float, default=0.0, help="Camera azimuth in degrees.")
    parser.add_argument("--normalize-mesh", action="store_true", help="Normalize mesh to match preprocess.")
    parser.add_argument("--normalize-scale", type=float, default=2.0, help="Target scale for mesh normalization.")
    parser.add_argument("--output", type=str, default="check_camera_output.png", help="Output image path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu.")
    args = parser.parse_args()

    device = torch.device(args.device)

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

    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))
    print(f"\n[INFO] URDF: {args.urdf}")
    print(f"[INFO] Joint: {joint_name}, Parent: {parent_link}, Child: {child_link}")
    print(f"[INFO] Joint origin xyz: {j_xyz}, rpy: {j_rpy}")
    print(f"[INFO] Joint axis: {jaxis}")

    # Get visual info
    child_mesh_path, child_vis_xyz, child_vis_rpy, child_scale = get_link_first_visual(root, child_link)
    parent_mesh_path, parent_vis_xyz, parent_vis_rpy, parent_scale = get_link_first_visual(root, parent_link)

    print(f"[INFO] Parent visual: xyz={parent_vis_xyz}, rpy={parent_vis_rpy}, scale={parent_scale}")
    print(f"[INFO] Child visual: xyz={child_vis_xyz}, rpy={child_vis_rpy}, scale={child_scale}")

    # Resolve mesh paths
    child_mesh_path = resolve_mesh_path(child_mesh_path, urdf_dir) if child_mesh_path else None
    parent_mesh_path = resolve_mesh_path(parent_mesh_path, urdf_dir) if parent_mesh_path else None

    print(f"\n[INFO] Loading meshes...")
    # Load meshes
    if parent_mesh_path is None:
        print("[WARN] Parent link has no visual mesh; using tiny dummy triangle.")
        Vp = torch.tensor([[0,0,0],[1e-6,0,0],[0,1e-6,0]], dtype=torch.float32, device=device)
        Fp = torch.tensor([[0,1,2]], dtype=torch.int64, device=device)
    else:
        Vp, Fp = load_mesh_as_tensors(parent_mesh_path, parent_scale, args.unit_scale, device,
                                       normalize=args.normalize_mesh, normalize_scale=args.normalize_scale)

    if child_mesh_path is None:
        raise ValueError("Child link has no visual mesh.")

    Vc, Fc = load_mesh_as_tensors(child_mesh_path, child_scale, args.unit_scale, device,
                                   normalize=args.normalize_mesh, normalize_scale=args.normalize_scale)

    print(f"[INFO] Parent mesh: {Vp.shape[0]} verts, {Fp.shape[0]} faces")
    print(f"[INFO] Child mesh: {Vc.shape[0]} verts, {Fc.shape[0]} faces")

    # Apply transforms to get final mesh positions
    print(f"\n[INFO] Applying URDF transforms...")

    # Parent transform
    Rp = rpy_to_matrix(tuple(parent_vis_rpy)).to(device)
    parent_vis_t = torch.tensor(parent_vis_xyz, dtype=torch.float32, device=device)
    Tp = make_SE3(Rp, parent_vis_t)
    Vp_w = transform_points_h(Tp, Vp)

    # Child transform: joint origin + rotation + child visual
    Rj = rpy_to_matrix(tuple(j_rpy)).to(device)
    joint_origin_t = torch.tensor(j_xyz, dtype=torch.float32, device=device)
    Tj = make_SE3(Rj, joint_origin_t)

    # Rotation about joint axis
    axis = torch.tensor(jaxis, dtype=torch.float32, device=device)
    angle = torch.tensor(math.radians(args.joint_angle_deg), dtype=torch.float32, device=device)
    Rrot = axis_angle_to_matrix(axis, angle).to(device)
    Trot = make_SE3(Rrot, torch.zeros(3, dtype=torch.float32, device=device))

    # Child visual
    Rc = rpy_to_matrix(tuple(child_vis_rpy)).to(device)
    child_vis_t = torch.tensor(child_vis_xyz, dtype=torch.float32, device=device)
    Tc = make_SE3(Rc, child_vis_t)

    # Combined child transform
    Tw = Tj @ Trot @ Tc
    Vc_w = transform_points_h(Tw, Vc)

    # Merge meshes
    V = torch.cat([Vp_w, Vc_w], dim=0)
    F_child = Fc + Vp.shape[0]
    F = torch.cat([Fp, F_child], dim=0)

    print(f"[INFO] Combined mesh: {V.shape[0]} verts, {F.shape[0]} faces")

    # Create PyTorch3D mesh
    Vtx = torch.ones_like(V)  # White vertex colors
    meshes = Meshes(verts=[V], faces=[F], textures=TexturesVertex(verts_features=[Vtx]))

    # Setup camera
    print(f"\n[INFO] Camera settings:")
    print(f"  Distance: {args.cam_dist}")
    print(f"  Elevation: {args.cam_elev}")
    print(f"  Azimuth: {args.cam_azim}")
    print(f"  FOV (vertical): {args.fov}")
    print(f"  Image size: {args.image_size}")

    scene_center_tuple = ((0.0, 0.0, 0.0),)  # Look at origin

    renderer, cameras = build_phong_renderer(
        image_size=args.image_size, device=device, fov=args.fov,
        cam_dist=args.cam_dist, cam_elev=args.cam_elev, cam_azim=args.cam_azim,
        at=scene_center_tuple
    )

    # Render
    print(f"\n[INFO] Rendering...")
    images = renderer(meshes)

    # Save RGB image
    rgb_np = images[0, ..., :3].detach().cpu().numpy()
    rgb_img = (rgb_np * 255).astype(np.uint8)
    Image.fromarray(rgb_img, mode='RGB').save(args.output)
    print(f"[INFO] Saved rendered image to: {args.output}")


if __name__ == "__main__":
    main()
