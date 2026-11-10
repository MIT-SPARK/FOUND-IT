import os
import re
import numpy as np
import itertools
import matplotlib
import scipy
import time
from PIL import Image
from torchvision import transforms as TF
import torch

import small_gicp
import open3d as o3d


def slice_with_overlap(lst, n, k):
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    def extract_number(path):
        filename = os.path.basename(path)
        # Look for digits, allowing zero or more non-numeric sub-extensions (like .color) before the final extension
        match = re.search(r'\d+(?:\.\d+)?(?=(?:\.[a-zA-Z_-][^.]*)*\.[^.]+$)', filename)
        return float(match.group()) if match else float('inf')
        
    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.
    
    Args:
        image_names (list of str): List of image filenames.
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.

    Returns:
        list of str: Downsampled list of image filenames.
    """
    return image_names[::downsample_factor]

def perform_registration(target_raw_numpy, source_raw_numpy, voxel_size, max_corr_factor=5):

    target, target_tree = small_gicp.preprocess_points(target_raw_numpy, downsampling_resolution=voxel_size)
    source, _ = small_gicp.preprocess_points(source_raw_numpy, downsampling_resolution=voxel_size)

    T_target_source = small_gicp.align(target, source, target_tree,
                                       max_correspondence_distance = \
                                               voxel_size * max_corr_factor,
                                       registration_type="GICP").T_target_source

    return T_target_source

def decompose_camera(P, no_inverse=False):
    """
    Decompose a 3x4 or 4x4 camera projection matrix P into intrinsics K,
    rotation R, and translation t.
    """
    if P.shape[0] != 3:
        P = P / P[-1,-1]
        P = P[0:3, :]

    # Ensure P is (3,4)
    assert P.shape == (3, 4)

    # Left 3x3 part
    M = P[:, :3]

    # RQ decomposition
    K, R = scipy.linalg.rq(M)

    # Make sure intrinsics have positive diagonal
    if K[0,0] < 0:
        K[:,0] *= -1
        R[0,:] *= -1
    if K[1,1] < 0:
        K[:,1] *= -1
        R[1,:] *= -1
    if K[2,2] < 0:
        K[:,2] *= -1
        R[2,:] *= -1

    scale = K[2,2]
    # print("Scale factor from K[2,2]:", scale)
    if not no_inverse:
        R = np.linalg.inv(R)
        t = -R @ np.linalg.inv(K) @ P[:, 3]
    else:
        t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
    
    return K, R, t, scale

def compute_image_embeddings(model, preprocess, image_paths, batch_size=64, device="cuda"):
    all_embs = []

    # Send images to wherever the model lives (avoids device mismatch)
    model_device = next(model.parameters()).device

    # Load all images into memory (PIL -> tensor)
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))

    # Stack into a single tensor
    imgs = torch.stack(imgs).to(model_device)

    # Loop over batches
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())

    # Combine into one (N, D) array
    return torch.cat(all_embs, dim=0).numpy()

def compute_text_embeddings(clip_model, tokenizer, text, device="cuda"):
    model_device = next(clip_model.parameters()).device
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(model_device)
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.cpu().numpy()

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors a and b.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T

def normalize_to_sl4(H):
    """
    Normalize a 4x4 homography matrix H to be in SL(4).
    """
    det = np.linalg.det(H)
    if det == 0:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    scale = det ** (1/4)
    H_normalized = H / scale
    return H_normalized

def denoise_points(points, nb_neighbors=40, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, 
                                        std_ratio=std_ratio)
    filtered_points = np.asarray(cl.points)
    
    return filtered_points, ind

def compute_obb_from_points(points: np.ndarray):
    """
    Compute an oriented bounding box (OBB) for a Nx3 point cloud.

    Returns:
        center      : (3,) world-space center of OBB
        extent      : (3,) lengths of OBB along its principal axes
        rotation    : (3,3) rotation matrix (columns = principal axes)
        or None if the cloud is empty / invalid (e.g. denoising removed every
        point), so callers can skip the detection instead of crashing.
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be Nx3 points"

    # Remove NaN/inf if any
    points = points[np.isfinite(points).all(axis=1)]
    points, _ = denoise_points(points)
    if len(points) == 0:
        return None

    # 1. Compute centroid
    centroid = points.mean(axis=0)

    # 2. PCA on centered points
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)

    # Eigen decomposition (sorted by eigenvalue descending)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # eigh doesn't guarantee det=+1; flip last column if needed to get a
    # proper right-handed rotation matrix (required by scipy Rotation, etc.)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1

    rotation = eigvecs  # columns = principal axes (R)

    # 3. Project points to PCA frame
    points_local = centered @ rotation

    # 4. Compute min/max in PCA frame → box extents
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner

    # 5. Compute box center (local), then world
    center_local = 0.5 * (min_corner + max_corner)
    center_world = centroid + center_local @ rotation.T

    return center_world, extent, rotation

def overlay_masks(image, masks):
    image = image.convert("RGBA")

    # masks: (N, 1, H, W) or (N, H, W)
    masks = (255 * masks.cpu().numpy()).astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        # Now mask is shape (H, W)

        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image

def overlay_contour(image, mask, color=(0, 255, 0), thickness=3):
    """Draw a contour boundary around a binary mask on the image."""
    from scipy.ndimage import binary_dilation, binary_erosion

    mask_2d = (np.squeeze(mask) > 0.5).astype(np.uint8)
    dilated = binary_dilation(mask_2d, iterations=thickness)
    boundary = dilated.astype(np.uint8) - binary_erosion(mask_2d, iterations=1).astype(np.uint8)
    boundary = np.clip(boundary, 0, 1)

    img_np = np.array(image.convert("RGB"))
    img_np[boundary > 0] = color
    return Image.fromarray(img_np)

def get_obb_corners(center, extent, rotation):
    """
    Generates the 8 corner points (World Frame) of an OBB.
    center: (3,) center of the box
    extent: (3,) full side lengths (width, height, depth)
    rotation: (3, 3) rotation matrix
    """
    # Create all 8 combinations of +/- half-extents
    # extent is full length, so we divide by 2 for offsets from center
    half_extents = extent / 2.0
    
    # Generate combinations: [(-x, -y, -z), (-x, -y, +z), ... (+x, +y, +z)]
    # This creates a list of 8 arrays, each shape (3,)
    signs = [-1, 1]
    local_corners = []
    for sx, sy, sz in itertools.product(signs, signs, signs):
        local_corners.append(half_extents * np.array([sx, sy, sz]))

    # Transform to World Frame: Center + (Rotation * Local)
    world_corners = []
    for local_point in local_corners:
        # Rotate then translate
        world_point = center + (rotation @ local_point)
        world_corners.append(world_point)
        
    return np.array(world_corners)

def is_object_visible(corners, proj_mat, image_size):
    """
    Checks if AT LEAST ONE corner of the object is visible.
    corners: (8, 3) array of corner points in World Frame
    proj_mat: 3x4 Projection Matrix (World -> Image Pixels)
    image_size: (width, height)
    """
    w, h = image_size
    
    for point_world in corners:
        point_h = np.append(point_world, 1.0) # Homogeneous (4,)
        
        # Calculate projection
        point_cam = proj_mat @ point_h
        
        # Flatten to 1D and take only the first 3 elements
        # This handles cases where point_cam is (4,) or (3, 1) or (4, 1)
        u_z, v_z, z = point_cam.flatten()[:3]
        
        # 1. Check if point is behind the camera (avoid division by zero or negative depth)
        if z <= 0.1: # Use a small near-clip epsilon
            continue

        # 2. Project to pixel coordinates
        u = u_z / z
        v = v_z / z

        # 3. Check bounds
        if 0 <= u < w and 0 <= v < h:
            return True # We found one visible corner
    
    return False

def voxel_downsample(points, frame_indices, pixel_rows, pixel_cols, colors,
                     voxel_size, min_points_per_voxel=1):
    """Keep one representative point per voxel cell.

    A voxel yields a survivor point only if it contains at least
    ``min_points_per_voxel`` points; sparser voxels are dropped as noise (a lone
    floater seen in a single frame is almost never real geometry, whereas a real
    surface accumulates many points across frames in the same voxel). The default
    of 1 keeps every occupied voxel (original behaviour).

    All input arrays must be the same length N.
    Returns (sparse_points, sparse_frame_indices, sparse_pixel_rows,
             sparse_pixel_cols, sparse_colors).
    """
    if len(points) == 0:
        return (
            np.empty((0, 3),    dtype=np.float32),
            np.empty(0,         dtype=np.int8),
            np.empty(0,         dtype=np.uint16),
            np.empty(0,         dtype=np.uint16),
            np.empty((0, 3),    dtype=np.uint8),
        )

    voxel_coords = np.floor(points / voxel_size).astype(np.int32)

    # void-view trick: treat each 3-int row as one opaque key for np.unique.
    # return_index → first point in each voxel (the survivor); return_counts →
    # how many points fell in that voxel (used to drop sparse/noisy voxels).
    dt = np.dtype((np.void, voxel_coords.dtype.itemsize * 3))
    _, unique_idx, counts = np.unique(
        np.ascontiguousarray(voxel_coords).view(dt),
        return_index=True, return_counts=True,
    )

    if min_points_per_voxel > 1:
        unique_idx = unique_idx[counts >= min_points_per_voxel]

    return (
        points[unique_idx],
        frame_indices[unique_idx],
        pixel_rows[unique_idx],
        pixel_cols[unique_idx],
        colors[unique_idx],
    )

class Accumulator:
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.total_time += (time.perf_counter() - self.start)