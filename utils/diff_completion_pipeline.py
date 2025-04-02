import numpy as np
import MinkowskiEngine as ME
import torch
from models.minkunet import MinkRewardModel,MinkGlobalEnc,MinkUNetDiff, MinkUNet
import open3d as o3d
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning.core.lightning import LightningModule
import yaml
import os
import tqdm
from natsort import natsorted
import click
import time
from os.path import join, dirname, abspath
from os import environ, makedirs
    
class DiffCompletion_DistillationDPO(LightningModule):
    def __init__(self, diff_path, refine_path, denoising_steps, uncond_w):
        super().__init__()

        # load diff net
        ckpt_diff = torch.load(diff_path)

        dm_weights = {k.replace('generator.', ''): v for k, v in ckpt_diff["state_dict"].items() if k.startswith('generator.')}
        encoder_weights = {k.replace('partial_enc.', ''): v for k, v in ckpt_diff["state_dict"].items()  if k.startswith('partial_enc.')}

        # load encoder and model
        self.partial_enc = MinkGlobalEnc().cuda()
        self.partial_enc.load_state_dict(encoder_weights, strict=True)
        self.partial_enc.eval()

        self.model = MinkUNetDiff().cuda()
        self.model.load_state_dict(dm_weights, strict=True)
        self.model.eval()

        # load refiner
        ckpt_refine = torch.load(refine_path)
        refiner_weights = {k.replace('model_refine.', ''): v for k, v in ckpt_refine["state_dict"].items()  if k.startswith('model_refine.')}
        self.model_refine = MinkUNet(in_channels=3, out_channels=3*6).cuda()
        self.model_refine.load_state_dict(refiner_weights, strict=True)
        self.model_refine.eval()
        
        self.cuda()

        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                beta_start=3.5e-5,
                beta_end=0.007,
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(num_inference_steps=denoising_steps)
        self.scheduler_to_cuda()

        self.w_uncond = uncond_w

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def feats_to_coord(self, p_feats, resolution, mean=None, std=None):
        p_feats = p_feats.reshape(mean.shape[0],-1,3)
        p_coord = torch.round(p_feats / resolution)

        return p_coord.reshape(-1,3)

    def points_to_tensor(self, points):
        x_feats = ME.utils.batched_coordinates(list(points[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord = torch.round(x_coord / 0.05)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t                                                                                         

    def reset_partial_pcd(self, x_part):
        x_part = self.points_to_tensor(x_part.F.reshape(1,-1,3).detach())

        return x_part

    def preprocess_scan(self, scan):
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < 50.0) & (dist > 3.5)][:,:3]

        # use farthest point sampling
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.farthest_point_down_sample(int(180000 / 10))
        scan = torch.tensor(np.array(pcd_scan.points)).cuda()
        
        scan = scan.repeat(10,1)
        scan = scan[None,:,:]

        return scan
    

    def postprocess_scan(self, completed_scan, input_scan):
        dist = np.sqrt(np.sum((completed_scan)**2, -1))
        post_scan = completed_scan[dist < 50.0]
        max_z = input_scan[...,2].max().item()
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()

        post_scan = post_scan[(post_scan[:,2] < max_z) & (post_scan[:,2] > min_z)]

        return post_scan

    def complete_scan(self, pcd_part):
        pcd_part_rep = self.preprocess_scan(pcd_part).view(1,-1,3)
        # pcd_part = torch.tensor(pcd_part, device=self.device).view(1,-1,3)
        # print(f'pcd_part_rep.shape = {pcd_part_rep.shape}')
        # print(f'pcd_part.shape = {pcd_part.shape}')

        x_feats = pcd_part_rep + torch.randn(pcd_part_rep.shape, device=self.device)
        x_full = self.points_to_tensor(x_feats) # x_T
        x_cond = self.points_to_tensor(pcd_part_rep) # x_0
        x_uncond = self.points_to_tensor(torch.zeros_like(pcd_part_rep))

        completed_scan = self.completion_loop(pcd_part_rep, x_full, x_cond, x_uncond)
        post_scan = self.postprocess_scan(completed_scan, pcd_part_rep)

        refine_in = self.points_to_tensor(post_scan[None,:,:])
        offset = self.refine_forward(refine_in).reshape(-1,6,3)

        refine_complete_scan = post_scan[:,None,:] + offset.cpu().numpy()

        return refine_complete_scan.reshape(-1,3), post_scan


    def refine_forward(self, x_in):
        with torch.no_grad():
            offset = self.model_refine(x_in)

        return offset

    def forward(self, x_full, x_full_sparse, x_part, t):
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            out = self.model(x_full, x_full_sparse, part_feat, t)

        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)
    
    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def completion_loop(self, x_init, x_t, x_cond, x_uncond):
        self.scheduler_to_cuda()

        # for t in tqdm.tqdm(range(len(self.dpm_scheduler.timesteps))):
        for t in range(len(self.dpm_scheduler.timesteps)):
            t = self.dpm_scheduler.timesteps[t].cuda()[None]

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t)

            x_cond = self.reset_partial_pcd(x_cond)
            torch.cuda.empty_cache()

        return x_t.F.cpu().detach().numpy()

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

class DiffCompletion_lidiff(LightningModule):
    def __init__(self, diff_path, refine_path, denoising_steps, uncond_w):
        super().__init__()

        # load diff net
        ckpt_diff = torch.load(diff_path)

        dm_weights = {k.replace('model.', ''): v for k, v in ckpt_diff["state_dict"].items() if k.startswith('model.')}
        encoder_weights = {k.replace('partial_enc.', ''): v for k, v in ckpt_diff["state_dict"].items()  if k.startswith('partial_enc.')}

        # load encoder and model
        self.partial_enc = MinkGlobalEnc().cuda()
        self.partial_enc.load_state_dict(encoder_weights, strict=True)
        self.partial_enc.eval()

        self.model = MinkUNetDiff().cuda()
        self.model.load_state_dict(dm_weights, strict=True)
        self.model.eval()

        # load refiner
        ckpt_refine = torch.load(refine_path)
        refiner_weights = {k.replace('model_refine.', ''): v for k, v in ckpt_refine["state_dict"].items()  if k.startswith('model_refine.')}
        self.model_refine = MinkUNet(in_channels=3, out_channels=3*6).cuda()
        self.model_refine.load_state_dict(refiner_weights, strict=True)
        self.model_refine.eval()
        
        self.cuda()

        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                beta_start=3.5e-5,
                beta_end=0.007,
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(num_inference_steps=denoising_steps)
        self.scheduler_to_cuda()

        self.w_uncond = uncond_w

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def feats_to_coord(self, p_feats, resolution, mean=None, std=None):
        p_feats = p_feats.reshape(mean.shape[0],-1,3)
        p_coord = torch.round(p_feats / resolution)

        return p_coord.reshape(-1,3)

    def points_to_tensor(self, points):
        x_feats = ME.utils.batched_coordinates(list(points[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord = torch.round(x_coord / 0.05)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t                                                                                         

    def reset_partial_pcd(self, x_part):
        x_part = self.points_to_tensor(x_part.F.reshape(1,-1,3).detach())

        return x_part

    def preprocess_scan(self, scan):
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < 50.0) & (dist > 3.5)][:,:3]

        # use farthest point sampling
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.farthest_point_down_sample(int(180000 / 10))
        scan = torch.tensor(np.array(pcd_scan.points)).cuda()
        
        scan = scan.repeat(10,1)
        scan = scan[None,:,:]

        return scan
    

    def postprocess_scan(self, completed_scan, input_scan):
        dist = np.sqrt(np.sum((completed_scan)**2, -1))
        post_scan = completed_scan[dist < 50.0]
        max_z = input_scan[...,2].max().item()
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()

        post_scan = post_scan[(post_scan[:,2] < max_z) & (post_scan[:,2] > min_z)]

        return post_scan

    def complete_scan(self, pcd_part):
        pcd_part_rep = self.preprocess_scan(pcd_part).view(1,-1,3)
        # pcd_part = torch.tensor(pcd_part, device=self.device).view(1,-1,3)
        # print(f'pcd_part_rep.shape = {pcd_part_rep.shape}')
        # print(f'pcd_part.shape = {pcd_part.shape}')

        x_feats = pcd_part_rep + torch.randn(pcd_part_rep.shape, device=self.device)
        x_full = self.points_to_tensor(x_feats) # x_T
        x_cond = self.points_to_tensor(pcd_part_rep) # x_0
        x_uncond = self.points_to_tensor(torch.zeros_like(pcd_part_rep))

        completed_scan = self.completion_loop(pcd_part_rep, x_full, x_cond, x_uncond)
        post_scan = self.postprocess_scan(completed_scan, pcd_part_rep)

        refine_in = self.points_to_tensor(post_scan[None,:,:])
        offset = self.refine_forward(refine_in).reshape(-1,6,3)

        refine_complete_scan = post_scan[:,None,:] + offset.cpu().numpy()

        return refine_complete_scan.reshape(-1,3), post_scan


    def refine_forward(self, x_in):
        with torch.no_grad():
            offset = self.model_refine(x_in)

        return offset

    def forward(self, x_full, x_full_sparse, x_part, t):
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            out = self.model(x_full, x_full_sparse, part_feat, t)

        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)
    
    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def completion_loop(self, x_init, x_t, x_cond, x_uncond):
        self.scheduler_to_cuda()

        # for t in tqdm.tqdm(range(len(self.dpm_scheduler.timesteps))):
        for t in range(len(self.dpm_scheduler.timesteps)):
            t = self.dpm_scheduler.timesteps[t].cuda()[None]

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t)

            x_cond = self.reset_partial_pcd(x_cond)
            torch.cuda.empty_cache()

        return x_t.F.cpu().detach().numpy()

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")
