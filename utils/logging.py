# Code imported from: https://github.com/zhangbingliang2019/DAPS/blob/main/cores/trajectory.py and
# https://github.com/zhangbingliang2019/DAPS/blob/main/posterior_sample.py
# Original author: bingliang


from torch.nn.functional import interpolate
from pathlib import Path
from PIL import Image
import imageio
import numpy as np
import yaml
import json
import torch
from torchvision.utils import save_image
from omegaconf import OmegaConf
import torch.nn as nn

class Trajectory(nn.Module):
    """
        Class for recording and storing trajectory data.
    """

    def __init__(self):
        super().__init__()
        self.tensor_data = {}
        self.value_data = {}
        self._compile = False

    def add_tensor(self, name, images):
        """
            Adds image data to the trajectory.

            Parameters:
                name (str): Name of the image data.
                images (torch.Tensor): Image tensor to add.
        """
        if name not in self.tensor_data:
            self.tensor_data[name] = []
        self.tensor_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        """
            Adds value data to the trajectory.

            Parameters:
                name (str): Name of the value data.
                values (any): Value to add.
        """
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        """
            Compiles the recorded data into tensors.

            Returns:
                Trajectory: The compiled trajectory object.
        """
        if not self._compile:
            self._compile = True
            for name in self.tensor_data.keys():
                self.tensor_data[name] = torch.stack(self.tensor_data[name], dim=0)
            for name in self.value_data.keys():
                self.value_data[name] = torch.tensor(self.value_data[name])
        return self

    @classmethod
    def merge(cls, trajs):
        """
            Merge a list of compiled trajectories from different batches

            Returns:
                Trajectory: The merged and compiled trajectory object.
        """
        merged_traj = cls()
        for name in trajs[0].tensor_data.keys():
            merged_traj.tensor_data[name] = torch.cat([traj.tensor_data[name] for traj in trajs], dim=1)
        for name in trajs[0].value_data.keys():
            merged_traj.value_data[name] = trajs[0].value_data[name]
        return merged_traj


def resize(y, x, inverse_task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if type(y) in [list, tuple]:
        return x.clone()
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if inverse_task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry


def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)
    return Path(dir)


def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils

def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)


def save_mp4_video(gt, y, x0hat_traj, x0y_traj, xt_traj, output_path, fps=24, sec=15, space=4):
    """
        stack and save trajectory as mp4 video
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    ix, iy = x0hat_traj.shape[-2:]
    reindex = np.linspace(0, len(xt_traj) - 1, sec * fps).astype(int)
    np_x0hat_traj = tensor_to_numpy(x0hat_traj[reindex])
    np_x0y_traj = tensor_to_numpy(x0y_traj[reindex])
    np_xt_traj = tensor_to_numpy(xt_traj[reindex])
    np_y = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]
    for x0hat, x0y, xt in zip(np_x0hat_traj, np_x0y_traj, np_xt_traj):
        canvas = np.ones((ix, 5 * iy + 4 * space, 3), dtype=np.uint8) * 255
        cx = cy = 0
        canvas[cx:cx + ix, cy:cy + iy] = np_y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = np_gt

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0hat

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = xt
        writer.append_data(canvas)
    writer.close()


def log_results(args, sde_trajs, results, images, y, full_samples, table_markdown, total_number):
    # log hyperparameters and configurations
    full_samples = full_samples.flatten(0, 1)
    root = safe_dir(Path(args.save_dir) / f"{args.name}_{args.data.name}_{args.inverse_task.operator.name}")
    with open(str(root / 'config.yaml'), 'w') as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file, default_flow_style=False, allow_unicode=True)

    # log grid results
    resized_y = resize(y, images, args.inverse_task.operator.name)
    stack = torch.cat([images, resized_y, full_samples])
    save_image(stack * 0.5 + 0.5, fp=str(root / 'grid_results.png'), nrow=total_number)

    # log individual sample instances
    if args.save_samples:
        #  Save recovered images
        pil_image_list = tensor_to_pils(full_samples)
        image_dir = safe_dir(root / 'samples')
        cnt = 0
        for run in range(args.num_runs):
            for idx in range(total_number):
                image_path = image_dir / '{:05d}_run{:04d}.png'.format(idx, run)
                pil_image_list[cnt].save(str(image_path))
                cnt += 1
        
        # Save original images
        pil_image_list = tensor_to_pils(images)
        image_dir = safe_dir(root / 'original')
        cnt = 0
        for run in range(args.num_runs):
            for idx in range(total_number):
                image_path = image_dir / '{:05d}_run{:04d}.png'.format(idx, run)
                pil_image_list[cnt].save(str(image_path))
                cnt += 1

    # log sampling trajectory and mp4 video
    if args.save_traj:
        traj_dir = safe_dir(root / 'trajectory')
        print()
        print('save trajectories to mp4 videos...')
        for run, sde_traj in enumerate(sde_trajs):
            if args.save_traj_raw_data:
                # might be SUPER LARGE
                traj_raw_data = safe_dir(traj_dir / 'raw')
                torch.save(sde_traj, str(traj_raw_data / 'trajectory_run{:04d}.pth'.format(run)))
            
            # save mp4 video for trajectories
            if args.sampler.name == 'daps':
                x0hat_traj = sde_traj.tensor_data['x0hat']
                x0y_traj = sde_traj.tensor_data['x0y']
                xt_traj = sde_traj.tensor_data['xt']
                for idx in range(total_number):
                    video_path = str(traj_dir / '{:05d}_run{:04d}.mp4'.format(idx, run))
                    save_mp4_video(images[idx], resized_y[idx], x0hat_traj[:, idx], x0y_traj[:, idx], xt_traj[:, idx], video_path)
            elif args.sampler.name in ["admm", "admm-nesterov"]:
                x_k_traj = sde_traj.tensor_data['x_k']
                z_k_traj = sde_traj.tensor_data['z_k']
                u_k_traj = sde_traj.tensor_data['u_k']
                for idx in range(total_number):
                    video_path = str(traj_dir / '{:05d}_run{:04d}.mp4'.format(idx, run))
                    save_mp4_video(images[idx], resized_y[idx],z_k_traj[:, idx], x_k_traj[:, idx], u_k_traj[:, idx], video_path)
                    
            elif args.sampler.name in ["prox-gd", "prox-hqs"]:
                x_k_traj = sde_traj.tensor_data['x_k']
                z_k_traj = sde_traj.tensor_data['z_k']
                noise_traj = sde_traj.tensor_data['noise']
                for idx in range(total_number):
                    video_path = str(traj_dir / '{:05d}_run{:04d}.mp4'.format(idx, run))
                    save_mp4_video(images[idx], resized_y[idx],z_k_traj[:, idx], x_k_traj[:, idx], noise_traj[:, idx], video_path)

    # log the evaluation metrics
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(table_markdown)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)