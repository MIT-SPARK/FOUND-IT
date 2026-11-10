import yaml
import numpy as np
from scipy.spatial.transform import Rotation as sprot

def write_objects_to_yaml(output_yaml_file, est_bboxes):
    output = {}
    for prompt in est_bboxes:
        objects = []
        for bbox_detection in est_bboxes[prompt]:
            bbox = bbox_detection.bbox
            obj = {}
            center = bbox.center
            extents = bbox.extent
            orientation = np.array(bbox.rotation)
            try:
                quat = sprot.from_matrix(orientation).as_quat().tolist()
            except ValueError as e:
                print(f"Error converting rotation matrix to quaternion for prompt '{prompt}': {e}")
                print("Using default quaternion [0, 0, 0, 1] for Identity rotation.")
                quat = [0, 0, 0, 1]
            rotation = {'w': quat[3], 'x': quat[0], 'y': quat[1], 'z': quat[2]}
            obj['center'] = center.tolist()
            obj['extents'] = extents.tolist()
            obj['rotation'] = rotation
            objects.append(obj)

        output[prompt] = objects

    with open(output_yaml_file, 'w') as outfile:
        yaml.dump(output, outfile, default_flow_style=False)

def run_object_evals(task_map_file, output_yaml_file, objects_module, solver):
    with open(task_map_file) as f:
        task_id_map = yaml.safe_load(f)

    est_bboxes = dict()
    for text_prompt in task_id_map:
        bboxes = objects_module.get_3D_object(text_prompt, solver)

        if text_prompt not in est_bboxes:
            est_bboxes[text_prompt] = []
        est_bboxes[text_prompt].extend(bboxes)
    write_objects_to_yaml(output_yaml_file, est_bboxes)