import os
import numpy as np
import open3d as o3d
import tqdm
from natsort import natsorted
import click
import json

from utils.diff_completion_pipeline import DiffCompletion_DistillationDPO, DiffCompletion_lidiff
from utils.histogram_metrics import compute_hist_metrics 
from utils.render import offscreen_render

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses


def get_scan_completion(scan_path, path, diff_completion, max_range):
    pcd_file = os.path.join(path, 'velodyne', scan_path)
    points = np.fromfile(pcd_file, dtype=np.float32)
    points = points.reshape(-1,4) 
    dist = np.sqrt(np.sum(points[:,:3]**2, axis=-1))
    input_points = points[dist < max_range, :3]
    if diff_completion is None:
        pred_path = f'{scan_path.split(".")[0]}.ply'
        pcd_pred = o3d.io.read_point_cloud(os.path.join(path, pred_path))
        points = np.array(pcd_pred.points)
        dist = np.sqrt(np.sum(points**2, axis=-1))
        pcd_pred.points = o3d.utility.Vector3dVector(points[dist < max_range])
    else:
        complete_scan_refined, complete_scan = diff_completion.complete_scan(points)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(complete_scan)
        pcd_pred_refined = o3d.geometry.PointCloud()
        pcd_pred_refined.points = o3d.utility.Vector3dVector(complete_scan_refined)

    return pcd_pred, pcd_pred_refined, input_points

def get_ground_truth(pose, cur_scan, seq_map, max_range):
    trans = pose[:-1,-1]
    dist_gt = np.sum((seq_map - trans)**2, axis=-1)**.5
    scan_gt = seq_map[dist_gt < max_range]
    scan_gt = np.concatenate((scan_gt, np.ones((len(scan_gt),1))), axis=-1)
    scan_gt = (scan_gt @ np.linalg.inv(pose).T)[:,:3]
    scan_gt = scan_gt[(scan_gt[:,2] > -4.) & (scan_gt[:,2] < 4.4)]
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(scan_gt)

    # filter only over the view point
    cur_pcd = o3d.geometry.PointCloud()
    cur_pcd.points = o3d.utility.Vector3dVector(cur_scan)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cur_pcd, voxel_size=10.)
    in_viewpoint = viewpoint_grid.check_if_included(pcd_gt.points)
    points_gt = np.array(pcd_gt.points)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt[in_viewpoint])

    return pcd_gt

def pcd_denoise(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd.select_by_index(ind)
    return pcd_clean

import torch
import random
def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@click.command()
@click.option('--path', '-p', type=str, default='datasets/SemanticKITTI/dataset/sequences/00', help='path to the scan sequence')
@click.option('--max_range', '-m', type=float, default=50, help='max range')
@click.option('--denoising_steps', '-t', type=int, default=8, help='number of denoising steps')
@click.option('--cond_weight', '-s', type=float, default=3.5, help='conditioning weights')
@click.option('--diff', '-d', type=str, default='checkpoints/distillationdpo_st.ckpt', help='trained diffusion model')
@click.option('--refine', '-r', type=str, default='checkpoints/refine_net.ckpt', help='refinement model')
@click.option('--save_path', '-sp', type=str, default='exp/pics/00', help='where to save pics')
def main(path, max_range, denoising_steps, cond_weight, diff, refine, save_path): 

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    diff_completion_distdpo = DiffCompletion_DistillationDPO(diff, refine, denoising_steps, cond_weight)

    poses = load_poses(os.path.join(path, 'calib.txt'), os.path.join(path, 'poses.txt'))
    seq_map = np.load(f'{path}/map_clean.npy')

    import random
    eval_list = list(zip(poses, natsorted(os.listdir(f'{path}/velodyne'))))
    random.shuffle(eval_list)
    for pose, scan_path in tqdm.tqdm(eval_list):

        pcd_pred, pcd_pred_refined, cur_scan = get_scan_completion(scan_path, path, diff_completion_distdpo, max_range)
        pcd_in = o3d.geometry.PointCloud()
        pcd_in.points = o3d.utility.Vector3dVector(cur_scan)
        pcd_gt = get_ground_truth(pose, cur_scan, seq_map, max_range)

        pic_dir = os.path.join(save_path, scan_path.split('.')[0])
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        pic_pred_dir = os.path.join(pic_dir, f'{denoising_steps}steps.png')
        pic_pred_refined_dir = os.path.join(pic_dir, f'{denoising_steps}steps_refined.png')
        pic_gt_dir = os.path.join(pic_dir, 'gt.png')
        pic_in_dir = os.path.join(pic_dir, 'in.png')

        pcd_pred = pcd_denoise(pcd_pred)
        pcd_pred_refined = pcd_denoise(pcd_pred_refined)
        pcd_pred_lidiff = pcd_denoise(pcd_pred_lidiff)
        pcd_pred_refined_lidiff = pcd_denoise(pcd_pred_refined_lidiff)

        offscreen_render(pcd_pred, pic_pred_dir)
        offscreen_render(pcd_pred_refined, pic_pred_refined_dir)
        offscreen_render(pcd_gt, pic_gt_dir)
        offscreen_render(pcd_in, pic_in_dir)


if __name__ == '__main__':
    main()

