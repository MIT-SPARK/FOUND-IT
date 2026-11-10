import argparse
import yaml
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from termcolor import colored

import model.iou as IOU

assert o3d.__version__ == '0.17.0', "There is a major bug in open3d version 0.13.0 that messes up oriented bounding boxes, use 0.17.0"

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
    print(x,y,z,w)
    bbox_dict_transformed = {
        'center': box_tranformed_center,
        'extents': bbox_dict['extents'],
        'rotation': {'w':w, 'x':x, 'y':y, 'z':z}
    }

    return bbox_dict_transformed

def get_transform_from_yaml(transforms_file, scene_name):
    with open(transforms_file, "r") as file:
        data = yaml.safe_load(file)

    T = np.array(data[scene_name]['T'])
    scale = float(data[scene_name]['scale'])

    return T, scale

def rescale(detection, scale):
        return {'center':[scale * i for i in detection['center']],
                'extents':[scale * i for i in detection['extents']],
                'rotation':detection['rotation']}

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
    - bool,bool: first is true if both box centroids are inside bounds of other box
        centroid, False otherwise. second is true if at least one box centroid is
        inside the other box
    """
    def is_point_inside_box(point, bounds):
        return (bounds[0] <= point[0] <= bounds[3] and
                bounds[1] <= point[1] <= bounds[4] and
                bounds[2] <= point[2] <= bounds[5])

    # Extract centroids
    centroid1 = np.asarray(box1.get_center())
    centroid2 = np.asarray(box2.get_center())

    # Extract half-lengths (extent) of the bounding boxes
    extent1 = np.asarray(box1.extent)
    extent2 = np.asarray(box2.extent)

    # Compute bounds for each box
    bounds1 = np.concatenate([centroid1 - extent1, centroid1 + extent1])
    bounds2 = np.concatenate([centroid2 - extent2, centroid2 + extent2])

    check_one = is_point_inside_box(centroid1, bounds2)
    check_two = is_point_inside_box(centroid2, bounds1)

    return [(check_one and check_two), (check_one or check_two)]


def get_open3d_bbox(bbox_dict):
    center = np.array(bbox_dict['center'], dtype=np.float64).reshape(3, 1)
    extents = np.array(bbox_dict['extents'], dtype=np.float64).reshape(3, 1)
    quat = bbox_dict['rotation']
    if type(quat) is not list:
        w, x, y, z = quat['w'], quat['x'], quat['y'], quat['z']
        quat = [x, y, z, w]
    else:
        quat.append(quat.pop(0))
    rot = R.from_quat(quat).as_matrix().astype(np.float64)

    return o3d.geometry.OrientedBoundingBox(center, rot, extents)

def evaluate_bboxes(gt_yaml, est_results, visualize=False, visualize_task=None):

    tasks_to_vis = []
    visualize_all_tasks = False
    if visualize and visualize_task is not None:
        if visualize_task[0] == "all":
            visualize_all_tasks = True
            print("Debug visualizing all tasks.")
        else:
            tasks_to_vis = ' '.join(visualize_task).split("*")
            print("Debug visualizing tasks: ", tasks_to_vis)

    print()

    all_ious = []
    gt_bboxes = []
    est_bboxes = []
    capture_success = []
    captured_success_weaker = []
    for task, gt_detections in yaml.safe_load(open(gt_yaml)).items():
        for gt_detection in gt_detections:
            gt_bbox = get_open3d_bbox(gt_detection)
            if task in tasks_to_vis or visualize_all_tasks:
                gt_bbox.color = [0, 1, 0]
                gt_bboxes.append(gt_bbox)
            # if multiple objects for task, make assignments based on best iou
            best_iou = 0.0
            best_bbox = None
            est_boxes = est_results[task]
            captured_object = False
            captured_weaker = False
            for est_box in est_boxes:
                est_bbox = get_open3d_bbox(est_box)
                iou = IOU.compute_iou(gt_bbox, est_bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_bbox = est_bbox
                    captured_object, captured_weaker = is_centroid_inside_bounds(
                        gt_bbox, est_bbox)
            if best_bbox is not None:
                if task in tasks_to_vis or visualize_all_tasks:
                    best_bbox.color = [1, 0, 0]
                    est_bboxes.append(best_bbox)
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

    if visualize:
        bboxes_viz = gt_bboxes + est_bboxes
        o3d.visualization.draw_geometries(bboxes_viz)
    average_iou = np.average(np.array(all_ious))
    strict_success_rate = sum(capture_success)/len(capture_success)
    weak_success_rate = sum(captured_success_weaker) / \
        len(captured_success_weaker)

    return BboxResults(average_iou, strict_success_rate, weak_success_rate)

def main():
    parser = argparse.ArgumentParser(description="Rescale Coordinates in yaml. ")

    parser.add_argument('--gt_yaml', type=str, default=None)
    parser.add_argument('--est_yaml', type=str, default=None)
    parser.add_argument('--scene_name', type=str, default=None, help="office,cubicle,apartment")
    parser.add_argument('-s', '--save', default=False, action="store_true")

    args = parser.parse_args()
    
    cwd = os.path.abspath(os.path.dirname(__file__))
    scene_transforms_path = os.path.abspath(os.path.join(cwd, "scene_transforms.yaml"))
    transform_mat, scale = get_transform_from_yaml(scene_transforms_path, args.scene_name)

    with open(args.yaml_in) as f:
        old_data = yaml.safe_load(f)

    scaled_dict = dict()
    for task, detections in old_data.items():
        scaled_dict[task] = []
        for detection in detections:
            scaled_box = rescale(detection, scale)
            transformed_box = create_obb_from_dict(scaled_box, transform_mat)
            scaled_dict[task].append(transformed_box)

    args = parser.parse_args()

    bbox_eval_result = evaluate_bboxes(
        args.gt_yaml, scaled_dict)

    print()
    print(colored("Average IOU: " + str(bbox_eval_result.avg_iou), 'green'))
    print(colored("Average Strict Success Rate: " + str(bbox_eval_result.strict_sr), 'green'))
    print(colored("Average Relax Success Rate: " + str(bbox_eval_result.weak_sr), 'green'))
    print()

    if args.save:
        bbox_eval_result.save_to_yaml(args.save_path)

if __name__ == "__main__":
    main()