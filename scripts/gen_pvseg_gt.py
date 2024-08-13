"""
Created on June, 2022

Authors: QiSong
"""

import os
import cv2
import tqdm
import torch
import shutil
import logging
import numpy as np
from PIL import Image
from typing import List
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_logs
from nuscenes.map_expansion.map_api import NuScenesMap

from config import *
import map_api as my_map_api
import object_api as my_obj_api


def encode_binary_labels(masks):
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def iterate_samples(nuscenes, start_token):
    sample_token = start_token
    while sample_token != '':
        sample = nuscenes.get('sample', sample_token)
        yield sample
        sample_token = sample['next']


def render_map():
    nusc = NuScenes(version=DATA_VERSION, dataroot=DATA_DIR, verbose=True)
    scenes = nusc.scene

    nusc_map_sin_onenorth = NuScenesMap(dataroot=DATA_DIR, map_name='singapore-onenorth')
    nusc_map_sin_hollandvillage = NuScenesMap(dataroot=DATA_DIR, map_name='singapore-hollandvillage')
    nusc_map_sin_queenstown = NuScenesMap(dataroot=DATA_DIR, map_name='singapore-queenstown')
    nusc_map_bos = NuScenesMap(dataroot=DATA_DIR, map_name='boston-seaport')

    if not os.path.exists(SAVE_DIR_MAP):
        os.makedirs(SAVE_DIR_MAP)

    layer_names = MAP_CLASSES
    
    for cur_camera in CAMERA_NAMES:

        for scene in tqdm.tqdm(scenes):
            log_record = nusc.get('log', scene['log_token'])
            log_location = log_record['location']

            if log_location == 'singapore-onenorth':
                nusc_map_api = nusc_map_sin_onenorth
            elif log_location == 'singapore-hollandvillage':
                nusc_map_api = nusc_map_sin_hollandvillage
            elif log_location == 'singapore-queenstown':
                nusc_map_api = nusc_map_sin_queenstown
            else:
                nusc_map_api = nusc_map_bos

            first_sample_token = scene['first_sample_token']

            for sample in iterate_samples(nusc, first_sample_token):
                sample_token = sample['token']

                image, label = my_map_api.get_image_and_mask(nusc, sample_token, nusc_map_api, layer_names=layer_names,
                                                            camera_channel=cur_camera)

                # Encode masks as integer bitmask
                labels = encode_binary_labels(label)

                # Save outputs to disk and Use sample token to save and recall labels
                img_token = nusc.get('sample_data', sample['data'][cur_camera])['token']
                output_path = os.path.join(os.path.expandvars(SAVE_DIR_MAP),
                                        img_token + '.png')
                Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)


def _split_to_samples(nusc, split_logs: List[str]) -> List[str]:
    """
    Convenience function to get the samples in a particular split.
    :param split_logs: A list of the log names in this split.
    :return: The list of samples.
    """
    samples = []
    for sample in nusc.sample:
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            samples.append(sample['token'])
    return samples


def render_object():
    splits = ['train', 'val']
    nusc = my_obj_api.MyNuScenes(version=DATA_VERSION, dataroot=DATA_DIR, verbose=True)

    if not os.path.exists(SAVE_DIR_OBJ):
        os.mkdir(SAVE_DIR_OBJ)

    for split in splits:
        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(split, nusc)
        # Use only the samples from the current split.
        sample_tokens = _split_to_samples(nusc, split_logs)
        for sample_token in tqdm.tqdm(sample_tokens):
            sample = nusc.get('sample', sample_token)
            for cam in CAMERA_NAMES:
                img_data = nusc.get('sample_data', sample['data'][cam])
                # if os.path.exists(SAVE_DIR_OBJ + split + '/' + cam + "/%s_label.jpg" % k):
                #     continue

                # shutil.copy(DATA_DIR + img_data['filename'],
                #             SAVE_DIR_OBJ + split + '/' + cam + "/%s.jpg" % k,
                #             follow_symlinks=True)

                # img = cv2.imread(dataroot + img_data['filename'], cv2.IMREAD_COLOR)
                # cv2.imwrite(SAVE_DIR_OBJ + split + '/' + cam + "/%s.jpg" % k, img)
                
                nusc._render_sample_data(img_data['token'],
                                        file_name=img_data['filename'],
                                        out_path=SAVE_DIR_OBJ,
                                        verbose=False)



if __name__ == '__main__':

    if TASK_FLAG == 'seg':
        render_object()
        render_map()
    elif TASK_FLAG == 'det':
        render_object()