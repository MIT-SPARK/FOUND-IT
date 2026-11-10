import struct
import numpy as np

def read_cameras_binary(path):
    '''
    Read cameras.bin into a dict: {camera_id: (model, width, height, params)}
    '''
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            num_params = {
                0: 3,  # SIMPLE_PINHOLE
                1: 4,  # PINHOLE
                2: 4,  # SIMPLE_RADIAL
                3: 5,  # RADIAL
                4: 8,  # OPENCV
                5: 12, # FULL_OPENCV
                6: 5,  # FISHEYE
                7: 9   # SIMPLE_RADIAL_FISHEYE
            }[model_id]
            params = struct.unpack("<" + "d"*num_params, f.read(8*num_params))
            cameras[camera_id] = (model_id, width, height, np.array(params))
    return cameras

def read_images_binary(path):
    '''
    Robust parser for images.bin that handles variable-length image name.
    '''
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            # image_id, qvec, tvec, camera_id
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<i", f.read(4))[0]

            # read image name (null-terminated)
            name_bytes = b""
            while True:
                char = f.read(1)
                if char == b'\x00':
                    break
                name_bytes += char
            image_name = name_bytes.decode("utf-8")

            images[image_id] = (qvec, tvec, camera_id, image_name)

            # skip points2D
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)

    return images



def qvec2rotmat(qvec):
    ''' Quaternion to 3x3 rotation matrix '''
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
