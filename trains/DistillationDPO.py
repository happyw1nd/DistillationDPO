import argparse
import torch
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
import datetime
from tqdm import tqdm
from os import makedirs, path
import os
import copy

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from diffusers import DPMSolverMultistepScheduler

from utils.collations import *
from utils.metrics import ChamferDistance, PrecisionRecall
from utils.scheduling import beta_func
from utils.metrics import ChamferDistance, PrecisionRecall, CompletionIoU, RMSE, EMD
from utils.histogram_metrics import compute_hist_metrics 
from models.minkunet import MinkRewardModel,MinkGlobalEnc,MinkUNetDiff
import datasets.SemanticKITTI_dataset as SemanticKITTI_dataset

class DistillationDPO(LightningModule):
    def __init__(self, args):
        super().__init__()        

        # configs
        self.lr = args.lr
        self.timestamp = args.timestamp
        self.args = args
        self.w_uncond = 3.5

        # Load pre-trained DM weights, init reference model
        dm_ckpt = torch.load(args.pre_trained_diff_path)
        # dm_weights = {k.replace('model.', ''): v for k, v in dm_ckpt["state_dict"].items() if k.startswith('model.')}
        dm_weights = {k.replace('DiffModel.', ''): v for k, v in dm_ckpt["state_dict"].items() if k.startswith('DiffModel.')}
        generator_weights = copy.deepcopy(dm_weights)
        auxDiffBetter_weights = copy.deepcopy(dm_weights)
        auxDiffWorse_weights = copy.deepcopy(dm_weights)
        teacher_weights = dm_weights
        # encoder_weights = {k.replace('partial_enc.', ''): v for k, v in dm_ckpt["state_dict"].items()  if k.startswith('partial_enc.')}
        encoder_weights = {k.replace('DM_encoder.', ''): v for k, v in dm_ckpt["state_dict"].items()  if k.startswith('DM_encoder.')}

        self.partial_enc = MinkGlobalEnc()
        self.generator = MinkUNetDiff()
        self.teacher = MinkUNetDiff()
        self.auxDiffBetter = MinkUNetDiff()
        self.auxDiffWorse = MinkUNetDiff()
        self.partial_enc.load_state_dict(encoder_weights, strict=True)
        self.generator.load_state_dict(generator_weights, strict=True)
        self.teacher.load_state_dict(teacher_weights, strict=True)
        self.auxDiffBetter.load_state_dict(auxDiffBetter_weights, strict=True)
        self.auxDiffWorse.load_state_dict(auxDiffWorse_weights, strict=True)

        for param in self.partial_enc.parameters():
            param.requires_grad = False
        for param in self.generator.parameters():
            param.requires_grad = True
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.auxDiffBetter.parameters():
            param.requires_grad = True
        for param in self.auxDiffWorse.parameters():
            param.requires_grad = True

        self.partial_enc.eval()

        # init scheduler for DM
        self.betas = beta_func['linear'](1000, 3.5e-5, 0.007)

        self.t_steps = 1000
        self.s_steps = 1
        self.s_steps_val = 8

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=self.device
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=self.device
        )

        self.betas = torch.tensor(self.betas, device=self.device)
        self.alphas = torch.tensor(self.alphas, device=self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=3.5e-5,
                beta_end=0.007,
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(num_inference_steps=self.s_steps)
        self.dpm_scheduler_val = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=3.5e-5,
                beta_end=0.007,
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler_val.set_timesteps(num_inference_steps=self.s_steps_val)
        self.scheduler_to_cuda()

        # metrcis for validation
        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(0.05 ,2*0.05, 100)
        self.completion_iou = CompletionIoU()
    
    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.to(self.device)
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.to(self.device)
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.to(self.device)
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.to(self.device)
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.to(self.device)
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.to(self.device)
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.to(self.device)
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.to(self.device)

        self.dpm_scheduler_val.timesteps = self.dpm_scheduler_val.timesteps.to(self.device)
        self.dpm_scheduler_val.betas = self.dpm_scheduler_val.betas.to(self.device)
        self.dpm_scheduler_val.alphas = self.dpm_scheduler_val.alphas.to(self.device)
        self.dpm_scheduler_val.alphas_cumprod = self.dpm_scheduler_val.alphas_cumprod.to(self.device)
        self.dpm_scheduler_val.alpha_t = self.dpm_scheduler_val.alpha_t.to(self.device)
        self.dpm_scheduler_val.sigma_t = self.dpm_scheduler_val.sigma_t.to(self.device)
        self.dpm_scheduler_val.lambda_t = self.dpm_scheduler_val.lambda_t.to(self.device)
        self.dpm_scheduler_val.sigmas = self.dpm_scheduler_val.sigmas.to(self.device)

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None].to(self.device) * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None].to(self.device) * noise

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond
    
    def reset_partial_pcd_part(self, x_part, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)

        return x_part

    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond, gt_pts, x_mean, x_std):
        pcd = o3d.geometry.PointCloud()
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(gt_pts.shape[0]).to(self.device).long() * self.dpm_scheduler.timesteps[t].to(self.device)
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

            # this is needed otherwise minkEngine will keep "stacking" coords maps over the x_part and x_uncond
            # i.e. memory leak
            x_cond = self.reset_partial_pcd_part(x_cond, x_mean, x_std)
            torch.cuda.empty_cache()

        makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)

        return x_t
    
    def p_sample_with_final_step_grad(self, x_init, x_t, x_cond, x_mean, x_std):

        assert len(self.dpm_scheduler.timesteps) == self.s_steps

        with torch.no_grad():
            for t in range(len(self.dpm_scheduler.timesteps)-1):
                t = torch.full((x_init.shape[0],), self.dpm_scheduler.timesteps[t]).to(self.device)
                
                noise_t = self.forward_generator(x_t, x_t.sparse(), x_cond, t)
                input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
                x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
                x_t = self.points_to_tensor(x_t, x_mean, x_std)

        t = torch.full((x_init.shape[0],), self.dpm_scheduler.timesteps[-1]).to(self.device)
                    
        noise_t = self.forward_generator(x_t, x_t.sparse(), x_cond, t)
        input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
        x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
        x_t = self.points_to_tensor(x_t, x_mean, x_std)

        x_cond = self.reset_partial_pcd_part(x_cond, x_mean, x_std)
        torch.cuda.empty_cache()

        return x_t

    def p_sample_one_step(self, x_init, x_t, x_cond, t, x_mean, x_std):
                    
        noise_t = self.forward_generator(x_t, x_t.sparse(), x_cond, t)
        input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
        x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
        x_t = self.points_to_tensor(x_t, x_mean, x_std)

        x_cond = self.reset_partial_pcd_part(x_cond, x_mean, x_std)
        torch.cuda.empty_cache()

        return x_t

    def pred_noise(self, model, sparse_TF, x_t_TF, t:torch.Tensor):

        condition = self.DM_encoder(sparse_TF)
        pred_noise = model(x_t_TF, x_t_TF.sparse(), condition, t)

        torch.cuda.empty_cache()
        return pred_noise

    
    def sample_val(self, x_init, x_t, x_cond, x_uncond, x_mean, x_std):

        assert len(self.dpm_scheduler_val.timesteps) == self.s_steps_val

        for t in range(len(self.dpm_scheduler_val.timesteps)):
            t = torch.full((x_init.shape[0],), self.dpm_scheduler_val.timesteps[t]).to(self.device)
            
            noise_t = self.classfree_forward_generator(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler_val.step(noise_t, t[0], input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t, x_mean, x_std)

        x_cond = self.reset_partial_pcd_part(x_cond, x_mean, x_std)
        torch.cuda.empty_cache()

        return x_t

    def do_forward(self, model, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = model(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)
    
    def forward_generator(self, x_full, x_full_sparse, x_part, t):
        part_feat = self.partial_enc(x_part)
        out = self.generator(x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def classfree_forward_generator(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward_generator(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward_generator(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)
    
    def feats_to_coord(self, p_feats, resolution, batch_size):
        p_feats = p_feats.reshape(batch_size,-1,3)
        p_coord = torch.round(p_feats / resolution)

        return p_coord.reshape(-1,3)

    def points_to_tensor(self, x_feats, mean=None, std=None):
        if mean is None:
            batch_size = x_feats.shape[0]
            x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

            x_coord = x_feats.clone()
            x_coord[:,1:] = self.feats_to_coord(x_feats[:,1:], 0.05, batch_size)

            x_t = ME.TensorField(
                features=x_feats[:,1:],
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )

            torch.cuda.empty_cache()

            return x_t

        else:
            x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

            x_coord = x_feats.clone()
            x_coord[:,1:] = feats_to_coord(x_feats[:,1:], 0.05, mean.shape[0])

            x_t = ME.TensorField(
                features=x_feats[:,1:],
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )

            torch.cuda.empty_cache()

            return x_t        

    def point_set_to_sparse(self, p_full, n_part):
        p_full = p_full[0].cpu().detach().numpy()
    
        dist_part = np.sum(p_full**2, -1)**.5
        p_full = p_full[(dist_part < 50.) & (dist_part > 3.5)]
        p_full = p_full[p_full[:,2] > -4.]

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(p_full)

        pcd_part = pcd_part.farthest_point_down_sample(n_part)
        p_part = torch.tensor(np.array(pcd_part.points))

        return p_part
    
    def calc_cd(self, pred:torch.Tensor, gt:torch.Tensor):

        assert pred.shape[0] == 1
        assert gt.shape[0] == 1

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred[0].cpu().detach().numpy())
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt[0].cpu().detach().numpy())

        dist_pt_2_gt = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
        dist_gt_2_pt = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))

        cd = (np.mean(dist_gt_2_pt) + np.mean(dist_pt_2_gt)) / 2

        return cd
    
    def calc_jsd(self, pred:torch.Tensor, gt:torch.Tensor):

        assert pred.shape[0] == 1
        assert gt.shape[0] == 1

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred[0].cpu().detach().numpy())
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt[0].cpu().detach().numpy())

        jsd = compute_hist_metrics(pcd_gt, pcd_pred, bev=False)

        return jsd

    def training_step(self, batch:dict, batch_idx, optimizer_idx):

        # vars to be used
        partial_tensor = batch['pcd_part']
        partial_x10_tensor = partial_tensor.repeat(1,10,1)
        gt_tensor = batch['pcd_full']
        B = batch['pcd_part'].shape[0]
        partial_TF = self.points_to_tensor(partial_tensor)

        if optimizer_idx == 0: # train auxDiffs

            # generate a batch of better & worse samples from Generator
            with torch.no_grad():
                self.generator.eval()
                t_gen = torch.randint(0, self.t_steps, size=(B,)).to(self.device)
                noise = torch.randn(partial_x10_tensor.shape, device=self.device)

                # better
                partial_x10_better_noised_tensor = partial_x10_tensor + noise
                partial_x10_better_noised_TF = self.points_to_tensor(partial_x10_better_noised_tensor)
                generated_sample1 = self.p_sample_with_final_step_grad(partial_x10_tensor, partial_x10_better_noised_TF, partial_TF, batch['mean'], batch['std']).F.reshape(B,-1,3)
                

                # worse
                partial_x10_better_noised_tensor = partial_x10_tensor + 1.1*noise
                partial_x10_better_noised_TF = self.points_to_tensor(partial_x10_better_noised_tensor)
                generated_sample2 = self.p_sample_with_final_step_grad(partial_x10_tensor, partial_x10_better_noised_TF, partial_TF, batch['mean'], batch['std']).F.reshape(B,-1,3)

                # calc cd
                cd1 = self.calc_cd(generated_sample1, gt_tensor).item()
                cd2 = self.calc_cd(generated_sample2, gt_tensor).item()

                # compare
                generated_better_sample = generated_sample1 if cd1 < cd2 else generated_sample2
                generated_worse_sample = generated_sample2 if cd1 < cd2 else generated_sample1
                cd_better = cd1 if cd1 < cd2 else cd2
                cd_worse = cd2 if cd1 < cd2 else cd1
                switch = cd1 > cd2
                self.log('cd/better', cd_better, prog_bar=False)
                self.log('cd/worse', cd_worse, prog_bar=False)
                self.log('cd/switch', switch, prog_bar=False)

            # add noise to generated samples
            with torch.no_grad():
                # better
                t_better = torch.randint(0, self.t_steps, size=(B,)).to(self.device)
                noise_better = torch.randn(generated_better_sample.shape, device=self.device)
                generated_better_noised_sample_tensor = generated_better_sample + self.q_sample(torch.zeros_like(generated_better_sample), 
                                                                                                t_better, noise_better)
                generated_better_noised_sample_TF = self.points_to_tensor(generated_better_noised_sample_tensor)

                # worse
                t_worse = torch.randint(0, self.t_steps, size=(B,)).to(self.device)
                noise_worse = torch.randn(generated_worse_sample.shape, device=self.device)
                generated_worse_noised_sample_tensor = generated_worse_sample + self.q_sample(torch.zeros_like(generated_worse_sample), 
                                                                                              t_worse, noise_worse)
                generated_worse_noised_sample_TF = self.points_to_tensor(generated_worse_noised_sample_tensor)

            # denoise generated samples with auxDiffs
            self.auxDiffBetter.train()
            pred_noise_auxB = self.do_forward(self.auxDiffBetter, generated_better_noised_sample_TF, generated_better_noised_sample_TF.sparse(), 
                                              partial_TF, t_better)
            self.auxDiffWorse.train()
            pred_noise_auxW = self.do_forward(self.auxDiffWorse, generated_worse_noised_sample_TF, generated_worse_noised_sample_TF.sparse(),
                                              partial_TF, t_worse)

            # calculate loss
            auxDiffBetter_loss = F.mse_loss(noise_better, pred_noise_auxB)
            auxDiffWorse_loss = F.mse_loss(noise_worse, pred_noise_auxW)
            loss_aux = auxDiffBetter_loss + auxDiffWorse_loss

            # log info on progress bar
            self.log('loss_auxB', auxDiffBetter_loss, prog_bar=True)
            self.log('loss_auxW', auxDiffWorse_loss, prog_bar=True)

            torch.cuda.empty_cache()

            return loss_aux
        
        if optimizer_idx == 1: # train generator

            # get a batch of sample from Generator
            self.generator.train()
            t_gen = torch.randint(0, self.t_steps, size=(B,)).to(self.device)
            noise = torch.randn(partial_x10_tensor.shape, device=self.device)

            partial_x10_better_noised_tensor = partial_x10_tensor + noise
            partial_x10_better_noised_TF = self.points_to_tensor(partial_x10_better_noised_tensor)
            generated_sample1 = self.p_sample_with_final_step_grad(partial_x10_tensor, partial_x10_better_noised_TF, partial_TF, batch['mean'], batch['std']).F.reshape(B,-1,3)

            partial_x10_worse_noised_tensor = partial_x10_tensor + 1.1*noise
            partial_x10_worse_noised_TF = self.points_to_tensor(partial_x10_worse_noised_tensor)
            generated_sample2 = self.p_sample_with_final_step_grad(partial_x10_tensor, partial_x10_worse_noised_TF, partial_TF, batch['mean'], batch['std']).F.reshape(B,-1,3)

            cd1 = self.calc_cd(generated_sample1, gt_tensor).item()
            cd2 = self.calc_cd(generated_sample2, gt_tensor).item()

            generated_better_sample = generated_sample1 if cd1 < cd2 else generated_sample2
            generated_worse_sample = generated_sample2 if cd1 < cd2 else generated_sample1

            # add noise to generated samples
            t = torch.randint(0, self.t_steps, size=(B,)).to(self.device)
            noise = torch.randn(generated_better_sample.shape, device=self.device)
            generated_better_noised_sample_tensor = generated_better_sample + self.q_sample(torch.zeros_like(generated_better_sample), t, noise)
            generated_better_noised_sample_TF = self.points_to_tensor(generated_better_noised_sample_tensor)

            generated_worse_noised_sample_tensor = generated_worse_sample + self.q_sample(torch.zeros_like(generated_worse_sample), t, noise)
            generated_worse_noised_sample_TF = self.points_to_tensor(generated_worse_noised_sample_tensor)

            # denoise generated samples with axudiffs and teacher, respectively
            with torch.no_grad():
                self.auxDiffBetter.eval()
                noise_auxB = self.do_forward(self.auxDiffBetter, generated_better_noised_sample_TF, generated_better_noised_sample_TF.sparse(), partial_TF, t)

                generated_better_noised_sample_TF = self.points_to_tensor(generated_better_noised_sample_tensor) 
                self.auxDiffWorse.eval()
                noise_auxW = self.do_forward(self.auxDiffWorse, generated_worse_noised_sample_TF, generated_worse_noised_sample_TF.sparse(), partial_TF, t)

                generated_worse_noised_sample_TF = self.points_to_tensor(generated_worse_noised_sample_tensor)
                self.teacher.eval()
                noise_better_teacher = self.do_forward(self.teacher, generated_better_noised_sample_TF, generated_better_noised_sample_TF.sparse(), partial_TF, t)
                noise_worse_teacher = self.do_forward(self.teacher, generated_worse_noised_sample_TF, generated_worse_noised_sample_TF.sparse(), partial_TF, t)

            
            distil_loss = ((noise_better_teacher - noise_auxB) * (generated_better_noised_sample_tensor - generated_better_sample)).mean() \
                        - ((noise_worse_teacher - noise_auxW) * (generated_worse_noised_sample_tensor - generated_worse_sample)).mean()

            generator_loss = distil_loss

            # log info on progress bar
            self.log('loss_g', generator_loss, prog_bar=True)

            torch.cuda.empty_cache()

            return generator_loss

    def configure_optimizers(self):
        
        optimizer_aux = torch.optim.SGD(list(self.auxDiffBetter.parameters())+list(self.auxDiffWorse.parameters()), lr=self.args.lr)
        optimizer_g = torch.optim.SGD(self.generator.parameters(), lr=self.args.lr)

        from torch.optim.lr_scheduler import StepLR
        scheduler_aux = StepLR(optimizer_aux, step_size=1, gamma=0.999)
        scheduler_g = StepLR(optimizer_g, step_size=1, gamma=0.999)

        return [optimizer_aux, optimizer_g], [scheduler_aux, scheduler_g]

    def validation_step(self, batch:dict, batch_idx):

        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()

            # for inference we get the partial pcd and sample the noise around the partial
            x_init = batch['pcd_part'].repeat(1,10,1)
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.sample_val(x_init, x_full, x_part, x_uncond, batch['mean'], batch['std'])
            x_gen_eval = x_gen_eval.F.reshape((gt_pts.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                

                pcd_pred = o3d.geometry.PointCloud()
                # pcd_pred_all = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()

                # pcd_pred_all.points = o3d.utility.Vector3dVector(c_pred)
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < 50.0
                points = c_pred[dist_idx]
                max_z = x_init[i][...,2].max().item()
                min_z = (x_init[i][...,2].mean() - 2 * x_init[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                file_path = f'exp/distill/sdpo_{self.args.timestamp}/samples/{self.args.batch_size*batch_idx+i}.pcd'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                o3d.io.write_point_cloud(file_path, pcd_pred)

                pcd_gt = o3d.geometry.PointCloud()
                # pcd_gt_all = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                # pcd_gt_all.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])

                pcd_part = o3d.geometry.PointCloud()
                pcd_part.points = o3d.utility.Vector3dVector(batch['pcd_part'][i].cpu().detach().numpy())
                pcd_part.paint_uniform_color([0., 1.,0.])

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)
                self.completion_iou.update(pcd_gt, pcd_pred)

        torch.cuda.empty_cache()

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        thr_ious = self.completion_iou.compute()

        return {'val_cd_mean': cd_mean, 'val_cd_std': cd_std, 'val_precision': pr, 'val_recall': re, 'val_fscore': f1, 'val_iou0.5': thr_ious[0.5], 'val_iou0.2': thr_ious[0.2], 'val_iou0.1': thr_ious[0.1]}

    def validation_epoch_end(self, outputs):

        cd_mean = np.mean(np.stack([x["val_cd_mean"] for x in outputs]))
        cd_std = np.mean(np.stack([x["val_cd_std"] for x in outputs]))
        pr = np.mean(np.stack([x["val_precision"] for x in outputs]))
        re = np.mean(np.stack([x["val_recall"] for x in outputs]))
        f1 = np.mean(np.stack([x["val_fscore"] for x in outputs]))
        iou0_5 = np.mean(np.stack([x["val_iou0.5"] for x in outputs]))
        iou0_2 = np.mean(np.stack([x["val_iou0.2"] for x in outputs]))
        iou0_1 = np.mean(np.stack([x["val_iou0.1"] for x in outputs]))


        self.log('val_cd_mean', cd_mean, prog_bar=False)
        self.log('val_cd_std', cd_std)
        self.log('val_precision', pr)
        self.log('val_recall', re)
        self.log('val_fscore', f1, prog_bar=False)
        self.log('val_iou05', iou0_5, prog_bar=True)
        self.log('val_iou02', iou0_2, prog_bar=True)
        self.log('val_iou01', iou0_1, prog_bar=True)

        self.chamfer_distance.reset()
        self.precision_recall.reset()
        self.completion_iou.reset()
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths



def main(args):
    # metadata
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.timestamp = timestamp

    # model
    model = DistillationDPO(args)

    # dataset
    dataloader = SemanticKITTI_dataset.dataloaders['KITTI'](args)

    # ckpt saving config
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"exp/distill/sdpo_{timestamp}/checkpoints",   
        filename="{epoch}-{step}-{val_iou05:.3f}-{val_iou02:.3f}-{val_iou01:.3f}", 
        save_top_k=-1,             # save every ckpt
        every_n_epochs=1,
    )

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(f"exp/distill/sdpo_{timestamp}", default_hp_metric=False)

    # setup trainer
    trainer = Trainer(
        gpus=2, strategy='ddp', accelerator='gpu',
        max_epochs=10,
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.05,
        val_check_interval=200,
        limit_val_batches=1,
        detect_anomaly=True,
    )
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=1, type=int, required=False, help="batch size")
    parser.add_argument("--lr", default=1e-5, type=float, required=False, help="learning rate")
    parser.add_argument(
        "--SemanticKITTI_path", 
        default='datasets/SemanticKITTI', 
        type=str, 
        required=False, 
        help="path to SementicKITTI dataset"
    )
    parser.add_argument(
        "--pre_trained_diff_path", 
        default='checkpoints/lidiff_ddpo_refined.ckpt', 
        type=str, 
        required=False, 
        help="path to pre-trained diffusion model weights"
    )

    args = parser.parse_args()
    main(args)