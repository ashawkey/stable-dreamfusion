import taichi as ti
import torch
from taichi.math import vec3
from torch.cuda.amp import custom_fwd

from .utils import NEAR_DISTANCE


@ti.kernel
def simple_ray_aabb_intersec_taichi_forward(
        hits_t: ti.types.ndarray(field_dim=2),
        rays_o: ti.types.ndarray(field_dim=2),
        rays_d: ti.types.ndarray(field_dim=2),
        centers: ti.types.ndarray(field_dim=2),
        half_sizes: ti.types.ndarray(field_dim=2)):

    for r in ti.ndrange(hits_t.shape[0]):
        ray_o = vec3([rays_o[r, 0], rays_o[r, 1], rays_o[r, 2]])
        ray_d = vec3([rays_d[r, 0], rays_d[r, 1], rays_d[r, 2]])
        inv_d = 1.0 / ray_d

        center = vec3([centers[0, 0], centers[0, 1], centers[0, 2]])
        half_size = vec3(
            [half_sizes[0, 0], half_sizes[0, 1], half_sizes[0, 1]])

        t_min = (center - half_size - ray_o) * inv_d
        t_max = (center + half_size - ray_o) * inv_d

        _t1 = ti.min(t_min, t_max)
        _t2 = ti.max(t_min, t_max)
        t1 = _t1.max()
        t2 = _t2.min()

        if t2 > 0.0:
            hits_t[r, 0, 0] = ti.max(t1, NEAR_DISTANCE)
            hits_t[r, 0, 1] = t2


class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_voxels, 3) voxel centers
        half_sizes: (N_voxels, 3) voxel half sizes
        max_hits: maximum number of intersected voxels to keep for one ray
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        hits_t = (torch.zeros(
            rays_o.size(0), 1, 2, device=rays_o.device, dtype=torch.float32) -
                  1).contiguous()

        simple_ray_aabb_intersec_taichi_forward(hits_t, rays_o, rays_d, center,
                                                half_size)

        return None, hits_t, None
