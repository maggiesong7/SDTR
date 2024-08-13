"""
Created on June, 2022

@author: QiSong
"""

import os


TASK_FLAG = 'det'  # 'seg' or 'det'

if TASK_FLAG == 'seg':

    MAP_CLASSES = ['road_segment', 'lane']

    OBJ_CLASSES = ['vehicle']  # 'vehicle' or 'car'

    NUM_MAP = 1

    NUM_OBJ = 1

elif TASK_FLAG == 'det':

    MAP_CLASSES = None

    OBJ_CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    NUM_MAP = 0

    NUM_OBJ = len(OBJ_CLASSES)


CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

GET_BIN_IMAGE = True

GET_2D_BOX = True

DATA_VERSION = 'v1.0-trainval'

DATA_DIR = '/data/nuscenes/'

SAVE_DIR_MAP = '/data/nuscenes/pv_labels/map/'


if TASK_FLAG == 'seg':

    SAVE_DIR_OBJ = os.path.join(DATA_DIR, 'pv_labels/', *OBJ_CLASSES)

elif TASK_FLAG == 'det':

    SAVE_DIR_OBJ = '/data/nuscenes/pv_labels/det/'
