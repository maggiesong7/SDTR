import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = False
if use_gt:
    out_dir = './gt_vis/'
else:
    out_dir = './result_vis/'

result_json = "work_dirs/pp-nus/results_eval/pts_bbox/results_nusc"
dataroot='./data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

for token in tqdm.tqdm(tokens):
    if os.path.exists(out_dir+token+"_pred.png"):
        continue
    else:
        if use_gt:
            nusc.render_sample(token, out_path = out_dir+token+"_gt.png", verbose=False)
        else:
            nusc.render_sample(token, out_path = out_dir+token+"_pred.png", verbose=False)

