import taichi as ti
import torch
from taichi.math import vec3
from torch.cuda.amp import custom_fwd

from .utils import __morton3D, calc_dt, mip_from_dt, mip_from_pos


@ti.kernel
def raymarching_train(rays_o: ti.types.ndarray(ndim=2),
                      rays_d: ti.types.ndarray(ndim=2),
                      hits_t: ti.types.ndarray(ndim=2),
                      density_bitfield: ti.types.ndarray(ndim=1),
                      noise: ti.types.ndarray(ndim=1),
                      counter: ti.types.ndarray(ndim=1),
                      rays_a: ti.types.ndarray(ndim=2),
                      xyzs: ti.types.ndarray(ndim=2),
                      dirs: ti.types.ndarray(ndim=2),
                      deltas: ti.types.ndarray(ndim=1),
                      ts: ti.types.ndarray(ndim=1), cascades: int,
                      grid_size: int, scale: float, exp_step_factor: float,
                      max_samples: float):

    # ti.loop_config(block_dim=256)
    for r in noise:
        ray_o = vec3(rays_o[r, 0], rays_o[r, 1], rays_o[r, 2])
        ray_d = vec3(rays_d[r, 0], rays_d[r, 1], rays_d[r, 2])
        d_inv = 1.0 / ray_d

        t1, t2 = hits_t[r, 0], hits_t[r, 1]

        grid_size3 = grid_size**3
        grid_size_inv = 1.0 / grid_size

        if t1 >= 0:
            dt = calc_dt(t1, exp_step_factor, grid_size, scale)
            t1 += dt * noise[r]

        t = t1
        N_samples = 0

        while (0 <= t) & (t < t2) & (N_samples < max_samples):
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, exp_step_factor, grid_size, scale)
            mip = ti.max(mip_from_pos(xyz, cascades),
                         mip_from_dt(dt, grid_size, cascades))

            # mip_bound = 0.5
            # mip_bound = ti.min(ti.pow(2., mip - 1), scale)
            mip_bound = scale
            mip_bound_inv = 1 / mip_bound

            nxyz = ti.math.clamp(0.5 * (xyz * mip_bound_inv + 1) * grid_size,
                                 0.0, grid_size - 1.0)
            # nxyz = ti.ceil(nxyz)

            idx = mip * grid_size3 + __morton3D(ti.cast(nxyz, ti.u32))
            occ = density_bitfield[ti.u32(idx // 8)] & (1 << ti.u32(idx % 8))
            # idx = __morton3D(ti.cast(nxyz, ti.uint32))
            # occ = density_bitfield[mip, idx//8] & (1 << ti.cast(idx%8, ti.uint32))

            if occ:
                t += dt
                N_samples += 1
            else:
                # t += dt
                txyz = (((nxyz + 0.5 + 0.5 * ti.math.sign(ray_d)) *
                         grid_size_inv * 2 - 1) * mip_bound - xyz) * d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, exp_step_factor, grid_size, scale)
                while t < t_target:
                    t += calc_dt(t, exp_step_factor, grid_size, scale)

        start_idx = ti.atomic_add(counter[0], N_samples)
        ray_count = ti.atomic_add(counter[1], 1)

        rays_a[ray_count, 0] = r
        rays_a[ray_count, 1] = start_idx
        rays_a[ray_count, 2] = N_samples

        t = t1
        samples = 0

        while (t < t2) & (samples < N_samples):
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, exp_step_factor, grid_size, scale)
            mip = ti.max(mip_from_pos(xyz, cascades),
                         mip_from_dt(dt, grid_size, cascades))

            # mip_bound = 0.5
            # mip_bound = ti.min(ti.pow(2., mip - 1), scale)
            mip_bound = scale
            mip_bound_inv = 1 / mip_bound

            nxyz = ti.math.clamp(0.5 * (xyz * mip_bound_inv + 1) * grid_size,
                                 0.0, grid_size - 1.0)
            # nxyz = ti.ceil(nxyz)

            idx = mip * grid_size3 + __morton3D(ti.cast(nxyz, ti.u32))
            occ = density_bitfield[ti.u32(idx // 8)] & (1 << ti.u32(idx % 8))
            # idx = __morton3D(ti.cast(nxyz, ti.uint32))
            # occ = density_bitfield[mip, idx//8] & (1 << ti.cast(idx%8, ti.uint32))

            if occ:
                s = start_idx + samples
                xyzs[s, 0] = xyz[0]
                xyzs[s, 1] = xyz[1]
                xyzs[s, 2] = xyz[2]
                dirs[s, 0] = ray_d[0]
                dirs[s, 1] = ray_d[1]
                dirs[s, 2] = ray_d[2]
                ts[s] = t
                deltas[s] = dt
                t += dt
                samples += 1
            else:
                # t += dt
                txyz = (((nxyz + 0.5 + 0.5 * ti.math.sign(ray_d)) *
                         grid_size_inv * 2 - 1) * mip_bound - xyz) * d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, exp_step_factor, grid_size, scale)
                while t < t_target:
                    t += calc_dt(t, exp_step_factor, grid_size, scale)


@ti.kernel
def raymarching_train_backword(segments: ti.types.ndarray(ndim=2),
                               ts: ti.types.ndarray(ndim=1),
                               dL_drays_o: ti.types.ndarray(ndim=2),
                               dL_drays_d: ti.types.ndarray(ndim=2),
                               dL_dxyzs: ti.types.ndarray(ndim=2),
                               dL_ddirs: ti.types.ndarray(ndim=2)):

    for s in segments:
        index = segments[s]
        dxyz = dL_dxyzs[index]
        ddir = dL_ddirs[index]

        dL_drays_o[s] = dxyz
        dL_drays_d[s] = dxyz * ts[index] + ddir


class RayMarcherTaichi(torch.nn.Module):

    def __init__(self, batch_size=8192):
        super(RayMarcherTaichi, self).__init__()

        self.register_buffer('rays_a',
                             torch.zeros(batch_size, 3, dtype=torch.int32))
        self.register_buffer(
            'xyzs', torch.zeros(batch_size * 1024, 3, dtype=torch.float32))
        self.register_buffer(
            'dirs', torch.zeros(batch_size * 1024, 3, dtype=torch.float32))
        self.register_buffer(
            'deltas', torch.zeros(batch_size * 1024, dtype=torch.float32))
        self.register_buffer(
            'ts', torch.zeros(batch_size * 1024, dtype=torch.float32))

        # self.register_buffer('dL_drays_o', torch.zeros(batch_size, dtype=torch.float32))
        # self.register_buffer('dL_drays_d', torch.zeros(batch_size, dtype=torch.float32))

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch.float32)
            def forward(ctx, rays_o, rays_d, hits_t, density_bitfield,
                        cascades, scale, exp_step_factor, grid_size,
                        max_samples):
                # noise to perturb the first sample of each ray
                noise = torch.rand_like(rays_o[:, 0])
                counter = torch.zeros(2,
                                      device=rays_o.device,
                                      dtype=torch.int32)

                raymarching_train(\
                    rays_o, rays_d,
                    hits_t.contiguous(),
                    density_bitfield, noise, counter,
                    self.rays_a.contiguous(),
                    self.xyzs.contiguous(),
                    self.dirs.contiguous(),
                    self.deltas.contiguous(),
                    self.ts.contiguous(),
                    cascades, grid_size, scale,
                    exp_step_factor, max_samples)

                # ti.sync()

                total_samples = counter[0]  # total samples for all rays
                # remove redundant output
                xyzs = self.xyzs[:total_samples]
                dirs = self.dirs[:total_samples]
                deltas = self.deltas[:total_samples]
                ts = self.ts[:total_samples]

                return self.rays_a, xyzs, dirs, deltas, ts, total_samples

                # @staticmethod
                # @custom_bwd
                # def backward(ctx, dL_drays_a, dL_dxyzs, dL_ddirs, dL_ddeltas, dL_dts,
                #              dL_dtotal_samples):
                #     rays_a, ts = ctx.saved_tensors
                #     # rays_a = rays_a.contiguous()
                #     ts = ts.contiguous()
                #     segments = torch.cat([rays_a[:, 1], rays_a[-1:, 1] + rays_a[-1:, 2]])
                #     dL_drays_o = torch.zeros_like(rays_a[:, 0])
                #     dL_drays_d = torch.zeros_like(rays_a[:, 0])
                #     raymarching_train_backword(segments.contiguous(), ts, dL_drays_o,
                #                                dL_drays_d, dL_dxyzs, dL_ddirs)
                #     # ti.sync()
                #     # dL_drays_o = segment_csr(dL_dxyzs, segments)
                #     # dL_drays_d = \
                #     #     segment_csr(dL_dxyzs*rearrange(ts, 'n -> n 1')+dL_ddirs, segments)

                #     return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None

        self._module_function = _module_function

    def forward(self, rays_o, rays_d, hits_t, density_bitfield, cascades,
                scale, exp_step_factor, grid_size, max_samples):
        return self._module_function.apply(rays_o, rays_d, hits_t,
                                           density_bitfield, cascades, scale,
                                           exp_step_factor, grid_size,
                                           max_samples)


@ti.kernel
def raymarching_test_kernel(
        rays_o: ti.types.ndarray(ndim=2),
        rays_d: ti.types.ndarray(ndim=2),
        hits_t: ti.types.ndarray(ndim=2),
        alive_indices: ti.types.ndarray(ndim=1),
        density_bitfield: ti.types.ndarray(ndim=1),
        cascades: int,
        grid_size: int,
        scale: float,
        exp_step_factor: float,
        N_samples: int,
        max_samples: int,
        xyzs: ti.types.ndarray(ndim=2),
        dirs: ti.types.ndarray(ndim=2),
        deltas: ti.types.ndarray(ndim=1),
        ts: ti.types.ndarray(ndim=1),
        N_eff_samples: ti.types.ndarray(ndim=1),
):

    for n in alive_indices:
        r = alive_indices[n]
        grid_size3 = grid_size**3
        grid_size_inv = 1.0 / grid_size

        ray_o = vec3(rays_o[r, 0], rays_o[r, 1], rays_o[r, 2])
        ray_d = vec3(rays_d[r, 0], rays_d[r, 1], rays_d[r, 2])
        d_inv = 1.0 / ray_d

        t = hits_t[r, 0]
        t2 = hits_t[r, 1]

        s = 0

        while (0 <= t) & (t < t2) & (s < N_samples):
            xyz = ray_o + t * ray_d
            dt = calc_dt(t, exp_step_factor, grid_size, scale)
            mip = ti.max(mip_from_pos(xyz, cascades),
                         mip_from_dt(dt, grid_size, cascades))

            # mip_bound = 0.5
            # mip_bound = ti.min(ti.pow(2., mip - 1), scale)
            mip_bound = scale
            mip_bound_inv = 1 / mip_bound

            nxyz = ti.math.clamp(0.5 * (xyz * mip_bound_inv + 1) * grid_size,
                                 0.0, grid_size - 1.0)
            # nxyz = ti.ceil(nxyz)

            idx = mip * grid_size3 + __morton3D(ti.cast(nxyz, ti.u32))
            occ = density_bitfield[ti.u32(idx // 8)] & (1 << ti.u32(idx % 8))

            if occ:
                xyzs[n, s, 0] = xyz[0]
                xyzs[n, s, 1] = xyz[1]
                xyzs[n, s, 2] = xyz[2]
                dirs[n, s, 0] = ray_d[0]
                dirs[n, s, 1] = ray_d[1]
                dirs[n, s, 2] = ray_d[2]
                ts[n, s] = t
                deltas[n, s] = dt
                t += dt
                hits_t[r, 0] = t
                s += 1

            else:
                txyz = (((nxyz + 0.5 + 0.5 * ti.math.sign(ray_d)) *
                         grid_size_inv * 2 - 1) * mip_bound - xyz) * d_inv

                t_target = t + ti.max(0, txyz.min())
                t += calc_dt(t, exp_step_factor, grid_size, scale)
                while t < t_target:
                    t += calc_dt(t, exp_step_factor, grid_size, scale)

        N_eff_samples[n] = s


def raymarching_test(rays_o, rays_d, hits_t, alive_indices, density_bitfield,
                     cascades, scale, exp_step_factor, grid_size, max_samples,
                     N_samples):

    N_rays = alive_indices.size(0)
    xyzs = torch.zeros(N_rays,
                       N_samples,
                       3,
                       device=rays_o.device,
                       dtype=rays_o.dtype)
    dirs = torch.zeros(N_rays,
                       N_samples,
                       3,
                       device=rays_o.device,
                       dtype=rays_o.dtype)
    deltas = torch.zeros(N_rays,
                         N_samples,
                         device=rays_o.device,
                         dtype=rays_o.dtype)
    ts = torch.zeros(N_rays,
                     N_samples,
                     device=rays_o.device,
                     dtype=rays_o.dtype)
    N_eff_samples = torch.zeros(N_rays,
                                device=rays_o.device,
                                dtype=torch.int32)

    raymarching_test_kernel(rays_o, rays_d, hits_t, alive_indices,
                            density_bitfield, cascades, grid_size, scale,
                            exp_step_factor, N_samples, max_samples, xyzs,
                            dirs, deltas, ts, N_eff_samples)

    # ti.sync()

    return xyzs, dirs, deltas, ts, N_eff_samples
