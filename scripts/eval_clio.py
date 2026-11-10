import numpy as np
import struct
import os
import copy
import re
import yaml
from termcolor import colored
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from packaging import version

from foundit.eval_utils.iou import compute_iou

min_version = "0.17.0"
current_version = o3d.__version__
assert version.parse(current_version) >= version.parse(min_version), \
    f"Open3D version {min_version} or higher is required to avoid OBB bugs. " \
    f"Current version is {current_version}."
    

class LineMesh:
    """
    Helper class to create thick cylindrical lines in Open3D.
    """
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        self.points = np.array(points)
        self.lines = np.array(lines) if lines is not None else self._lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []
        self._create_line_mesh()

    @staticmethod
    def _lines_from_ordered_points(points):
        return np.array([[i, i + 1] for i in range(points.shape[0] - 1)])

    @staticmethod
    def _align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
        if np.array_equal(a, b):
            return None, None
        axis = np.cross(a, b)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            return None, None
        axis = axis / axis_norm
        angle = np.arccos(np.dot(a, b))
        return axis, angle

    @staticmethod
    def _normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis), l2

    def _create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = self._normalized(line_segments)
        z_axis = np.array([0, 0, 1])

        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            axis, angle = self._align_vector_to_another(z_axis, line_segment)
            translation = first_points[i, :] + line_segment * line_length * 0.5
            
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(self.radius, line_length)
            cylinder.translate(translation, relative=False)
            
            if axis is not None:
                axis_a = axis * angle
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)
                cylinder.rotate(rotation_matrix, center=cylinder.get_center())

            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder.paint_uniform_color(color)
            self.cylinder_segments.append(cylinder)

    def get_geometries(self):
        return self.cylinder_segments

class BboxResults:
    def __init__(self, avg_iou, strict_sr, weak_sr):
        self.avg_iou = avg_iou
        self.strict_sr = strict_sr
        self.weak_sr = weak_sr

    def to_dict(self, decimals=3):
        output = {}
        output["avg_iou"] = round(float(self.avg_iou), decimals)
        output["strict_sr"] = round(float(self.strict_sr), decimals)
        output["weak_sr"] = round(float(self.weak_sr), decimals)
        return output
        
    def to_list(self, fields, decimals=3):
        mapping = self.to_dict(decimals=decimals)
        return [mapping[f] for f in fields]

    def save_to_yaml(self, filename, decimals=3):
        data = self.to_dict(decimals=decimals)
        with open(filename, 'w') as file:
            yaml.dump(data, file)


def obb_to_corners(obb):
    # Extract the eight corners of the OBB
    corners = np.asarray(obb.get_box_points())
    return corners

def is_centroid_inside_bounds(box1, box2):
    """
    Check if the centroid of each OrientedBoundingBox is inside the bounds of the other box.
    
    Parameters:
    - box1: Open3D OrientedBoundingBox
    - box2: Open3D OrientedBoundingBox
    
    Returns:
    - list of bool: 
        [0]: True if BOTH box centroids are inside the other box.
        [1]: True if AT LEAST ONE box centroid is inside the other box.
    """
    def is_point_inside_obb(point, box):
        # Extract box properties
        center = np.asarray(box.get_center())
        R = np.asarray(box.R)             # 3x3 Rotation matrix
        extent = np.asarray(box.extent)   # Full lengths along local axes
        
        # Translate point to the box's local origin
        local_point = point - center
        
        # Rotate the point into the box's local coordinate frame.
        # R transforms from local to world, so R.T (transpose) transforms world to local.
        local_point_rotated = R.T @ local_point
        
        
        # np.abs checks the distance from the local origin along each axis
        return np.all(np.abs(local_point_rotated) <= extent)

    # Extract world centroids
    centroid1 = np.asarray(box1.get_center())
    centroid2 = np.asarray(box2.get_center())

    # Check cross-containment
    check_one = is_point_inside_obb(centroid1, box2)  # Is centroid1 inside box2?
    check_two = is_point_inside_obb(centroid2, box1)  # Is centroid2 inside box1?

    return [(check_one and check_two), (check_one or check_two)]

def get_id_from_image_name(image_name):
    """Parses frame ID from filename (e.g., 'frame_0012.jpg' -> 12)."""
    name_no_ext = os.path.splitext(image_name)[0]
    digits = re.findall(r'\d+', name_no_ext)
    return int(digits[-1]) if digits else None

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary_with_names(path_to_model_file):
    """Reads COLMAP images.bin. Returns {id: (center_xyz, filename)}."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_props = read_next_bytes(fid, 64, "idddddddi")
            qvec = np.array(binary_props[1:5])
            tvec = np.array(binary_props[5:8])
            
            image_name = ""
            char = read_next_bytes(fid, 1, "c")[0]
            while char != b"\x00":
                image_name += char.decode("utf-8")
                char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1) # Skip 2D points

            # Convert World-to-Camera to Camera Center (World Coordinates)
            rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
            center = -rot.as_matrix().T @ tvec
            
            extracted_id = get_id_from_image_name(image_name)
            if extracted_id is not None:
                images[extracted_id] = (center, image_name)
    return images

def read_est_trajectory(file_path):
    """Reads est trajectory (frame_id x y z ...)."""
    traj_points = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            fid = int(float(parts[0]))
            traj_points[fid] = np.array([float(x) for x in parts[1:4]])
    return traj_points

def get_transform_from_yaml(transforms_file, scene_name):
    """Loads scene-specific transform and scale from YAML."""
    with open(transforms_file, "r") as file:
        data = yaml.safe_load(file)
    return np.array(data[scene_name]['T']), float(data[scene_name]['scale'])

def compute_sim3(p_a, p_b):
    """
    Computes Sim3 sfrom p_a to p_b
    """
    mu_c = np.mean(p_a, axis=0)
    mu_e = np.mean(p_b, axis=0)
    
    p_c_centered = p_a - mu_c
    p_e_centered = p_b - mu_e
    
    n = p_a.shape[0]
    sigma = (1/n) * (p_e_centered.T @ p_c_centered)
    
    U, D, V_T = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        S[2, 2] = -1
        
    R_align = U @ S @ V_T
    var_c = (1/n) * np.trace(p_c_centered.T @ p_c_centered)
    scale = (1/var_c) * np.trace(np.diag(D) @ S)
    t_align = mu_e - scale * (R_align @ mu_c)
    
    return scale, R_align, t_align


def get_open3d_bbox_from_dict(bbox_dict):
    """Parses a dictionary into an Open3D OrientedBoundingBox."""
    center = np.array(bbox_dict['center'], dtype=np.float64).reshape(3, 1)
    extents = np.array(bbox_dict['extents'], dtype=np.float64).reshape(3, 1)
    quat = bbox_dict['rotation']
    
    # Handle both list and dict quaternion formats
    if isinstance(quat, dict):
        q_list = [quat['x'], quat['y'], quat['z'], quat['w']]
    else:
        # Assuming list format is [w, x, y, z] -> change to [x, y, z, w] for Scipy
        q_list = [quat[1], quat[2], quat[3], quat[0]]
        
    rot = R.from_quat(q_list).as_matrix().astype(np.float64)
    return o3d.geometry.OrientedBoundingBox(center, rot, extents)

def visualize_trajectory_plt(poses_colmap, poses_aligned):
    """Matplotlib 3D plot for trajectory alignment verification."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(poses_colmap[:, 0], poses_colmap[:, 1], poses_colmap[:, 2], c='black', label='COLMAP ground truth', linewidth=2)
    ax.plot(poses_aligned[:, 0], poses_aligned[:, 1], poses_aligned[:, 2], c='g', label='Esimated Aligned Trajectory')

    ax.set_title('Sim3 Alignment Verification')
    ax.legend()
    plt.show()

def generate_bbox_viz_geometries(bboxes, color):
    """Generates thick wireframe geometries for a list of bounding boxes."""
    geometries = []
    for bbox in bboxes:
        # Create a deepcopy and set color
        box = copy.deepcopy(bbox)
        box.color = color
        
        # Convert to LineSet -> Points/Lines -> Thick LineMesh
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        points = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)
        
        line_mesh = LineMesh(points, lines, colors=color, radius=0.02)
        geometries.extend(line_mesh.get_geometries())
    return geometries

def visualize_scene_o3d(colmap_mesh_path, est_pcd_path, 
                       T_est_to_colmap, est_to_colmap_scale, 
                       T_colmap_world, colmap_to_world_scale, gt_bboxes_dict, 
                       est_bboxes_dict, show_gt=True):
    
    # --- Initialization and Data Loading ---
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("VGGT-SLAM Query Visualizer", 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    
    # Materials
    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"
    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 2.0
    
    print("Loading Geometries...")
    mesh_colmap = o3d.io.read_triangle_mesh(colmap_mesh_path).simplify_vertex_clustering(0.01)
    pcd_est = o3d.io.read_point_cloud(est_pcd_path)
    center = np.array([0., 0., 0.])

    # Transforms
    pcd_est.scale(est_to_colmap_scale, center).transform(T_est_to_colmap)
    mesh_colmap.scale(colmap_to_world_scale, center).transform(T_colmap_world)
    pcd_est.scale(colmap_to_world_scale, center).transform(T_colmap_world)

    scene.scene.add_geometry("mesh_colmap", mesh_colmap, mat_mesh)
    scene.scene.add_geometry("pcd_est", pcd_est, mat_pcd)

    # --- UI Setup ---
    em = window.theme.font_size
    layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
    
    query_info = gui.Label("Filter by Query (e.g. 'chair')")
    query_input = gui.TextEdit()
    
    layout.add_child(query_info)
    layout.add_child(query_input)
    
    # --- Highlight Logic ---
    def update_highlights(text):
        query = text.lower().strip()
        # Remove previous boxes
        scene.scene.clear_geometry()
        scene.scene.add_geometry("mesh_colmap", mesh_colmap, mat_mesh)
        scene.scene.add_geometry("pcd_est", pcd_est, mat_pcd)

        # Helper to filter and add boxes
        def add_filtered_boxes(box_dict, color, prefix):
            for label, boxes in box_dict.items():
                # If query is empty, show all. If not, match substring.
                if not query or query in label.lower():
                    # Check if boxes is a list or single object
                    box_list = boxes if isinstance(boxes, list) else [boxes]
                    for i, b in enumerate(box_list):
                        o3d_bbox = get_open3d_bbox_from_dict(b)
                        o3d_bbox.color = color
                        # Wireframe handling in New API
                        name = f"{prefix}_{label}_{i}"
                        scene.scene.add_geometry(name, o3d_bbox, mat_pcd)

        if show_gt:
            add_filtered_boxes(gt_bboxes_dict, [0, 0, 0], "gt") # Black
        add_filtered_boxes(est_bboxes_dict, [0, 1, 0], "est")   # Green

    # Bind the Enter key/text change
    query_input.set_on_value_changed(update_highlights)

    # Initial draw
    update_highlights("")

    # --- Final Layout Logic ---
    def on_layout(layout_context):
        # Define the sidebar width
        sidebar_width = 200 * layout_context.theme.font_size / 16 # Responsive-ish width
        rect = window.content_rect
        
        # Position the sidebar (layout) on the left
        layout.frame = gui.Rect(rect.x, rect.y, sidebar_width, rect.height)
        
        # Position the 3D scene to fill the rest of the window
        scene.frame = gui.Rect(rect.x + sidebar_width, rect.y, 
                            rect.width - sidebar_width, rect.height)

    # Set the layout callback
    window.set_on_layout(on_layout)

    # Add children to the window AFTER defining layout behavior
    window.add_child(scene)
    window.add_child(layout)
    
    # View setup
    bounds = mesh_colmap.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    gui.Application.instance.run()
    

def create_obb_from_dict(bbox_dict, transform_mat):
    # extract box pose
    quat = bbox_dict['rotation']
    w, x, y, z = quat['w'], quat['x'], quat['y'], quat['z']
    quat = [x, y, z, w]
    rot = R.from_quat(quat).as_matrix().astype(np.float64)
    center = np.array(bbox_dict['center'])
    box_pose = np.eye(4)
    box_pose[0:3,0:3] = rot
    box_pose[0:3,3] = center

    # apply transform to box pose
    box_tranformed_pose = transform_mat @ box_pose

    # repackage bounding box into dictionary
    box_tranformed_rot = box_tranformed_pose[0:3,0:3]
    box_tranformed_center = box_tranformed_pose[0:3,3].tolist()
    quat = R.from_matrix(box_tranformed_rot).as_quat().astype(np.float64)
    x, y, z, w = [float(i) for i in quat]
    bbox_dict_transformed = {
        'center': box_tranformed_center,
        'extents': bbox_dict['extents'],
        'rotation': {'w':w, 'x':x, 'y':y, 'z':z}
    }

    return bbox_dict_transformed

def rescale(detection, scale):
        return {'center':[scale * i for i in detection['center']],
                'extents':[scale * i for i in detection['extents']],
                'rotation':detection['rotation']}

def evaluate_bboxes(gt_results, est_results):
    all_ious = []
    capture_success = []
    captured_success_weaker = []
    total_num_est_objects = sum(len(est_boxes) for est_boxes in est_results.values())
    print("Total estimated objects:", total_num_est_objects)
    for task, gt_detections in gt_results.items():
        for gt_detection in gt_detections:
            gt_bbox = get_open3d_bbox_from_dict(gt_detection)
            # if multiple objects for task, make assignments based on best iou
            best_iou = 0.0
            best_bbox = None
            est_boxes = est_results[task]
            captured_object = False
            captured_weaker = False
            for est_box in est_boxes:
                est_bbox = get_open3d_bbox_from_dict(est_box)
                try:
                    iou = compute_iou(gt_bbox, est_bbox)
                except np.linalg.LinAlgError:
                    print("unable to compute IoU, assigning IoU of 0.0")
                    iou = 0.0
                if iou >= best_iou:
                    best_iou = iou
                    best_bbox = est_bbox
                    captured_object, captured_weaker = is_centroid_inside_bounds(
                        gt_bbox, est_bbox)
            all_ious.append(best_iou)
            capture_success.append(captured_object)
            captured_success_weaker.append(captured_weaker)
            if captured_object:
                print(colored(task + " " + str(best_iou) +
                              " object captured: strict", 'green'))
            elif captured_weaker:
                print(colored(task + " " + str(best_iou) +
                              " object captured: relaxed", 'yellow'))
            else:
                print(colored(task + " " + str(best_iou) +
                              " object captured: missed", 'red'))

    average_iou = np.average(np.array(all_ious))
    strict_success_rate = sum(capture_success)/len(capture_success)
    weak_success_rate = sum(captured_success_weaker) / \
        len(captured_success_weaker)

    return BboxResults(average_iou, strict_success_rate, weak_success_rate)

if __name__ == "__main__":
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCENES = ("apartment", "cubicle", "office")
    # COLMAP sparse-model subfolder per scene (defaults to 0; cubicle uses 1).
    SPARSE_INDEX = {"cubicle": 1}

    parser = argparse.ArgumentParser(description="Evaluate estimated bboxes on Clio dataset")
    parser.add_argument('--clio_dataset_path', default="/home/dominic/cam_ws/imgs/clio_datasets",
                        help="Root directory of the Clio datasets; each scene lives in "
                             "<clio_dataset_path>/<scene> with sparse/<N>/ and dense/ subdirs.")
    parser.add_argument('--scene', choices=SCENES, default="apartment",
                        help="Clio scene to evaluate (default: apartment)")
    parser.add_argument('--vis', action='store_true', help='Visualize trajectories and scene')
    args = parser.parse_args()

    scene_dir = os.path.join(args.clio_dataset_path, args.scene)
    sparse_index = SPARSE_INDEX.get(args.scene, 0)

    # The estimated-map inputs come from a `main.py ... --log_os --log_results` run:
    # the trajectory and point cloud land in the repo root, the estimated bboxes in
    # clio_tasks/est_bboxes_<scene>.yaml.  COLMAP reference data lives under the
    # scene directory; the GT tasks and scene transform ship with the repo.
    CONFIG = {
        "colmap_bin": os.path.join(scene_dir, "sparse", str(sparse_index), "images.bin"),
        "colmap_mesh": os.path.join(scene_dir, "dense", "meshed-poisson.ply"),
        "gt_yaml": os.path.join(_REPO_ROOT, "clio_tasks", f"tasks_{args.scene}.yaml"),
        "est_traj": os.path.join(_REPO_ROOT, "poses.txt"),
        "est_pcd": os.path.join(_REPO_ROOT, "poses_points.pcd"),
        "est_yaml": os.path.join(_REPO_ROOT, "clio_tasks", f"est_bboxes_{args.scene}.yaml"),
        "scene_tf_yaml": os.path.join(_SCRIPT_DIR, "scene_transform.yaml"),
        "scene_name": args.scene,
    }

    # Load Data
    colmap_data = read_images_binary_with_names(CONFIG["colmap_bin"])
    est_data = read_est_trajectory(CONFIG["est_traj"])

    # Match Frames
    common_ids = sorted(list(set(colmap_data.keys()).intersection(set(est_data.keys()))))
    if len(common_ids) < 3:
        raise ValueError(f"Insufficient overlap. Found only {len(common_ids)} common frames.")

    print(f"Aligning based on {len(common_ids)} common frames.")
    poses_colmap = np.array([colmap_data[fid][0] for fid in common_ids])
    poses_est = np.array([est_data[fid] for fid in common_ids])

    # Compute initial Sim3 from trajectory (poses)
    s_est_colmap, R_est_colmap, t_est_colmap = compute_sim3(poses_est, poses_colmap)

    # Calculate Error
    print(f"Sim3 Alignment Results:\n Scale: {s_est_colmap:.4f}\n Translation: {t_est_colmap}\n")

    poses_est_aligned = (s_est_colmap * (R_est_colmap @ poses_est.T).T) + t_est_colmap
    error = np.linalg.norm(poses_est_aligned - poses_colmap, axis=1).mean()
    print(f"Post-refinement alignment error: {error:.4f}m")
    
    T_est_colmap = np.eye(4)
    T_est_colmap[:3, :3] = R_est_colmap
    T_est_colmap[:3, 3] = t_est_colmap

    # Get ground truth data and transform from COLMAP to World
    T_colmap_world, s_colmap_world = get_transform_from_yaml(CONFIG["scene_tf_yaml"], CONFIG["scene_name"])
    gt_data = yaml.safe_load(open(CONFIG["gt_yaml"]))

    est_bboxes = yaml.safe_load(open(CONFIG["est_yaml"]))

    # rescale est_bboxes to be in world frame
    print(s_est_colmap, s_colmap_world)
    est_bboxes_scaled = dict()
    for task, detections in est_bboxes.items():
        est_bboxes_scaled[task] = []
        for detection in detections:
            scaled_box = rescale(detection, s_est_colmap)
            scaled_box = create_obb_from_dict(scaled_box, T_est_colmap)
            scaled_box = rescale(scaled_box, s_colmap_world)
            scaled_box = create_obb_from_dict(scaled_box, T_colmap_world)
            est_bboxes_scaled[task].append(scaled_box)
    
    bbox_eval_result = evaluate_bboxes(gt_data, est_bboxes_scaled)

    print()
    print(colored("Average IOU: " + str(bbox_eval_result.avg_iou), 'green'))
    print(colored("Average Strict Success Rate: " + str(bbox_eval_result.strict_sr), 'green'))
    print(colored("Average Relax Success Rate: " + str(bbox_eval_result.weak_sr), 'green'))
    print()


    if args.vis:
        # Visualize Trajectories
        visualize_trajectory_plt(poses_colmap, poses_est_aligned)

        # Visualize point clouds of estimated scene, colmap, and estimated and ground truth bounding boxes
        visualize_scene_o3d(
            colmap_mesh_path=CONFIG["colmap_mesh"],
            est_pcd_path=CONFIG["est_pcd"],
            T_est_to_colmap=T_est_colmap,
            est_to_colmap_scale=s_est_colmap,
            T_colmap_world=T_colmap_world,
            colmap_to_world_scale=s_colmap_world,
            gt_bboxes_dict=gt_data,
            est_bboxes_dict=est_bboxes_scaled,
            show_gt=True
        )