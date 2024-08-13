"""
Created on June, 2022

@author: QiSong
"""


from sre_parse import FLAGS
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os
import os.path as osp
from typing import Tuple, List, Dict, Union

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

# from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, \
#     get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name

from config import *


class MyNuScenes(NuScenes):
    def __init__(self, version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        super(MyNuScenes, self).__init__(version=version,
                                         dataroot=dataroot,
                                         verbose=verbose,
                                         map_resolution=map_resolution)
        self.explorer = MyNuScenesExplorer(self)

    def _render_sample_data(self, sample_data_token: str, with_anns: bool = True,
                            box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float = 40, ax: Axes = None, file_name: str = None,
                            nsweeps: int = 1, out_path: str = None, underlay_map: bool = True,
                            use_flat_vehicle_coordinates: bool = True,
                            show_lidarseg: bool = False,
                            show_lidarseg_legend: bool = False,
                            filter_lidarseg_labels: List = None,
                            lidarseg_preds_bin_path: str = None, verbose: bool = True) -> None:

        self.explorer._render_sample_data(sample_data_token, with_anns, box_vis_level, axes_limit, ax, file_name=file_name, nsweeps=nsweeps,
                                          out_path=out_path, underlay_map=underlay_map,
                                          use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                          show_lidarseg=show_lidarseg,
                                          show_lidarseg_legend=show_lidarseg_legend,
                                          filter_lidarseg_labels=filter_lidarseg_labels,
                                          lidarseg_preds_bin_path=lidarseg_preds_bin_path, verbose=verbose)


class MyNuScenesExplorer(NuScenesExplorer):
    def _render_sample_data(self,
                            sample_data_token: str,
                            with_anns: bool = True,
                            box_vis_level: BoxVisibility = BoxVisibility.ANY,
                            axes_limit: float = 40,
                            ax: Axes = None,
                            file_name: str = None,
                            nsweeps: int = 1,
                            out_path: str = None,
                            underlay_map: bool = True,
                            use_flat_vehicle_coordinates: bool = True,
                            show_lidarseg: bool = False,
                            show_lidarseg_legend: bool = False,
                            filter_lidarseg_labels: List = None,
                            lidarseg_preds_bin_path: str = None,
                            verbose: bool = True) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        """
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                if show_lidarseg:
                    assert hasattr(self.nusc, 'lidarseg'), 'Error: nuScenes-lidarseg not installed!'

                    # Ensure that lidar pointcloud is from a keyframe.
                    assert sd_record['is_key_frame'], \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                    assert nsweeps == 1, \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                        'be set to 1.'

                    # Load a single lidar point cloud.
                    pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                else:
                    # Get aggregated lidar point cloud in lidar frame.
                    pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,
                                                                     nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            if sensor_modality == 'lidar' and show_lidarseg:
                # Load labels for pointcloud.
                if lidarseg_preds_bin_path:
                    sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
                    lidarseg_labels_filename = lidarseg_preds_bin_path
                    assert os.path.exists(lidarseg_labels_filename), \
                        'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                        'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
                else:
                    if len(self.nusc.lidarseg) > 0:  # Ensure lidarseg.json is not empty (e.g. in case of v1.0-test).
                        lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                            self.nusc.get('lidarseg', sample_data_token)['filename'])
                    else:
                        lidarseg_labels_filename = None

                if lidarseg_labels_filename:
                    # Paint each label in the pointcloud with a RGBA value.
                    colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)

                    if show_lidarseg_legend:
                        # Since the labels are stored as class indices, we get the RGB colors from the colormap
                        # in an array where the position of the RGB color corresponds to the index of the class
                        # it represents.
                        color_legend = colormap_to_colors(self.nusc.colormap, self.nusc.lidarseg_name2idx_mapping)

                        # If user does not specify a filter, then set the filter to contain the classes present in
                        # the pointcloud after it has been projected onto the image; this will allow displaying the
                        # legend only for classes which are present in the image (instead of all the classes).
                        if filter_lidarseg_labels is None:
                            filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)

                        create_lidarseg_legend(filter_lidarseg_labels,
                                               self.nusc.lidarseg_idx2name_mapping, self.nusc.colormap,
                                               loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1.0))
                else:
                    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
                    print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                          'from the ego vehicle instead.'.format(self.nusc.version))
            else:
                colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0

            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

            ax.axis('off')
            # ax.set_title('{} {labels_type}'.format(
            #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
            ax.set_aspect('equal')

            if out_path is not None:
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=229.4)

            if verbose:
                plt.show()

            ax.clear()
            plt.close()
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            bin_img = np.zeros((NUM_OBJ, data.size[1], data.size[0]), dtype=np.uint8)

            if GET_BIN_IMAGE:
                if with_anns:
                    for box in boxes:
                        if NUM_OBJ == 1:
                            if not box.name.split('.')[0] == 'vehicle':  # for class vehicle
                                continue

                            render(box, bin_img, flag=GET_2D_BOX, class_id=0, view=camera_intrinsic, normalize=True,
                                   colors=(255, 255, 255))

                        else:
                            # Get the index of the class
                            det_name = category_to_detection_name(box.name)
                            if det_name not in OBJ_CLASSES:
                                continue
                            else:
                                class_id = OBJ_CLASSES.index(det_name)

                            render(box, bin_img, flag=GET_2D_BOX, class_id=class_id, view=camera_intrinsic,
                                   normalize=True, colors=(255, 255, 255))

                # Encode masks as integer bitmask
                labels = encode_binary_labels(bin_img)

                if out_path is not None:
                    # Save outputs to disk
                    if TASK_FLAG == 'det':
                        save_name = file_name.split('/')[-1]
                        output_path = os.path.join(os.path.expandvars(out_path),
                                                save_name + '.png')
                    else:
                        output_path = os.path.join(os.path.expandvars(out_path),
                                                sample_data_token + '.png')
                    Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)
            else:
                # Init axes.
                if ax is None:
                    _, ax = plt.subplots(1, 1, figsize=(9, 16))

                # Show image.
                ax.imshow(data)

                # Show boxes.
                if with_anns:
                    for box in boxes:
                        c = np.array(self.get_color(box.name)) / 255.0
                        box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

                # Limit visible range.
                ax.set_xlim(0, data.size[0])
                ax.set_ylim(data.size[1], 0)

                ax.axis('off')
                # ax.set_title('{} {labels_type}'.format(
                #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
                ax.set_aspect('equal')

                if out_path is not None:
                    # Save outputs to disk
                    output_path = os.path.join(os.path.expandvars(out_path),
                                               sample_data_token + '.png')
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=229.4)

                if verbose:
                    plt.show()

                ax.clear()
                plt.close()

        else:
            raise ValueError("Error: Unknown sensor modality!")


def render(box,
           bin_img,
           flag,
           class_id,
           view: np.ndarray = np.eye(3),
           normalize: bool = False,
           colors: Tuple = ('b', 'r', 'k'),
           linewidth: float = 2) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param bin_img: The bin image to plot on the labeled box.
    :param flag: The flag to determine whether use 2D bounding box or not.
    :param class_id: The index of the class.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
    if flag:
        # Project 3d box from 3D space to image plane.
        corner_coords = view_points(box.corners(), view, True).T[:, :2].tolist()

        # 3d box(eight points) to 2D box(four points).
        final_coords = post_process_coords(corner_coords)
        min_x, min_y, max_x, max_y = final_coords

        # From the lower-left corner, draw a line in a clockwise direction
        pts = np.round([[min_x, min_y],
                        [min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y],
                        ]).astype(np.int32)
        cv2.fillPoly(bin_img[class_id], [pts], 1.0)

    else:
        corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

        # Draw bottom corners
        bottom = corners.T[[2, 3, 7, 6]]
        pts = np.round([[bottom[0][0], bottom[0][1]],
                        [bottom[1][0], bottom[1][1]],
                        [bottom[2][0], bottom[2][1]],
                        [bottom[3][0], bottom[3][1]],
                        ]).astype(np.int32)
        cv2.fillPoly(bin_img[class_id], [pts], 1.0)


def post_process_coords(
        corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def encode_binary_labels(masks):
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)
