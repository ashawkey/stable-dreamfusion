import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, safe_normalize

def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis):
    #                   phis [B,]; thetas: [B,]
    # front = 0         0-90            
    # side (left) = 1   90-180
    # back = 2          180-270
    # side (right) = 3  270-360
    # top = 4                        0-30
    # bottom = 5                     150-180
    res = torch.zeros(phis.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < (np.pi / 2))] = 0
    res[(phis >= (np.pi / 2)) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (3 * np.pi / 2))] = 2
    res[(phis >= (3 * np.pi / 2)) & (phis < (2 * np.pi))] = 3
    # override by thetas
    res[thetas < (np.pi / 6)] = 4
    res[thetas >= (5 * np.pi / 6)] = 5
    return res


def rand_poses(size, device, return_dirs=False, radius_range=[1, 1.5], theta_range=[0, 4 * np.pi / 6], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # jitters
    centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
    targets = torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    up_noise = torch.randn_like(up_vector) * 0.02
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis)
    else:
        dirs = None
    
    return poses, dirs


def circle_poses(device, return_dirs=False, radius=1.25, theta=np.pi/2, phi=0):

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis)
    else:
        dirs = None
    
    return poses, dirs    
    

class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.size = size

        self.training = self.type in ['train', 'all']
        
        self.cx = self.H / 2
        self.cy = self.W / 2

        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, return_dirs=self.opt.dir_text, radius_range=self.radius_range)
        # visualize_poses(poses.detach().cpu().numpy())


    def collate(self, index):

        B = len(index) # always 1

        if self.training:
            # random pose on the fly
            poses, dirs = rand_poses(B, self.device, return_dirs=self.opt.dir_text, radius_range=self.radius_range)

            # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            focal = self.H / (2 * np.tan(np.radians(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])
        else:
            # circle pose
            phi = (index[0] / self.size) * 2 * np.pi
            poses, dirs = circle_poses(self.device, return_dirs=self.opt.dir_text, radius=self.radius_range[1], theta=np.pi/2, phi=phi)

            # fixed focal
            fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
            focal = self.H / (2 * np.tan(np.radians(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])


        # sample a low-resolution but full image for CLIP
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
        }

        return data


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access dataset in trainer.
        return loader