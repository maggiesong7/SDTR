import os
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from config import *


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


def worker(info):
    lidar_path = info['lidar_infos'][LIDAR_NAMES]['filename']
    points = np.fromfile(os.path.join(DATA_DIR, lidar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)[..., :4]
    lidar_calibrated_sensor = info['lidar_infos'][LIDAR_NAMES][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][LIDAR_NAMES]['ego_pose']
    for i, cam_key in enumerate(CAMERA_NAMES):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        img = mmcv.imread(
            os.path.join(DATA_DIR, info['cam_infos'][cam_key]['filename']))
        pts_img, depth = map_pointcloud_to_image(
            points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, depth[:, None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(DATA_DIR, 'pv_labels/depth',
                           # os.path.join('/tmp_hdd', 'depth_trainval',
                                        f'{file_name}.bin'))
    # plt.savefig(f"{sample_idx}")


if __name__ == '__main__':
    po = Pool(24)
    mmcv.mkdir_or_exist(SAVE_DIR_DEP)
    infos = mmcv.load(os.path.join(DATA_DIR, 'nuscenes_12hz_infos_train.pkl'))
    # import ipdb; ipdb.set_trace()
    for info in infos:
        po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()
