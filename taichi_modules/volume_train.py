import taichi as ti
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from .utils import (data_type, ti2torch, ti2torch_grad, torch2ti,
                    torch2ti_grad, torch_type)


@ti.kernel
def composite_train_fw_array(
        sigmas: ti.types.ndarray(),
        rgbs: ti.types.ndarray(),
        deltas: ti.types.ndarray(),
        ts: ti.types.ndarray(),
        rays_a: ti.types.ndarray(),
        T_threshold: float,
        total_samples: ti.types.ndarray(),
        opacity: ti.types.ndarray(),
        depth: ti.types.ndarray(),
        rgb: ti.types.ndarray(),
        ws: ti.types.ndarray(),
):

    for n in opacity:
        ray_idx = rays_a[n, 0]
        start_idx = rays_a[n, 1]
        N_samples = rays_a[n, 2]

        T = 1.0
        samples = 0
        while samples < N_samples:
            s = start_idx + samples
            a = 1.0 - ti.exp(-sigmas[s] * deltas[s])
            w = a * T

            rgb[ray_idx, 0] += w * rgbs[s, 0]
            rgb[ray_idx, 1] += w * rgbs[s, 1]
            rgb[ray_idx, 2] += w * rgbs[s, 2]
            depth[ray_idx] += w * ts[s]
            opacity[ray_idx] += w
            ws[s] = w
            T *= 1.0 - a

            # if T<T_threshold:
            #     break
            samples += 1

        total_samples[ray_idx] = samples


@ti.kernel
def composite_train_fw(sigmas: ti.template(), rgbs: ti.template(),
                       deltas: ti.template(), ts: ti.template(),
                       rays_a: ti.template(), T_threshold: float,
                       T: ti.template(), total_samples: ti.template(),
                       opacity: ti.template(), depth: ti.template(),
                       rgb: ti.template(), ws: ti.template()):

    ti.loop_config(block_dim=256)
    for n in opacity:
        ray_idx = ti.i32(rays_a[n, 0])
        start_idx = ti.i32(rays_a[n, 1])
        N_samples = ti.i32(rays_a[n, 2])

        rgb[ray_idx, 0] = 0.0
        rgb[ray_idx, 1] = 0.0
        rgb[ray_idx, 2] = 0.0
        depth[ray_idx] = 0.0
        opacity[ray_idx] = 0.0
        total_samples[ray_idx] = 0

        T[start_idx] = 1.0
        # T_ = 1.0
        # samples = 0
        # while samples<N_samples:
        for sample_ in range(N_samples):
            # T_ = T[ray_idx, samples]
            s = start_idx + sample_
            T_ = T[s]
            if T_ > T_threshold:
                # s = start_idx + sample_
                a = 1.0 - ti.exp(-sigmas[s] * deltas[s])
                w = a * T_
                rgb[ray_idx, 0] += w * rgbs[s, 0]
                rgb[ray_idx, 1] += w * rgbs[s, 1]
                rgb[ray_idx, 2] += w * rgbs[s, 2]
                depth[ray_idx] += w * ts[s]
                opacity[ray_idx] += w
                ws[s] = w
                # T_ *= (1.0-a)
                T[s + 1] = T_ * (1.0 - a)
                # if T[s+1]>=T_threshold:
                # samples += 1
                total_samples[ray_idx] += 1
            else:
                T[s + 1] = 0.0

        # total_samples[ray_idx] = N_samples


@ti.kernel
def check_value(
        fields: ti.template(),
        array: ti.types.ndarray(),
        checker: ti.types.ndarray(),
):
    for I in ti.grouped(array):
        if fields[I] == array[I]:
            checker[I] = 1


class VolumeRenderer(torch.nn.Module):

    def __init__(self, batch_size=8192, data_type=data_type):
        super(VolumeRenderer, self).__init__()
        # samples level
        self.sigmas_fields = ti.field(dtype=data_type,
                                      shape=(batch_size * 1024, ),
                                      needs_grad=True)
        self.rgbs_fields = ti.field(dtype=data_type,
                                    shape=(batch_size * 1024, 3),
                                    needs_grad=True)
        self.deltas_fields = ti.field(dtype=data_type,
                                      shape=(batch_size * 1024, ),
                                      needs_grad=True)
        self.ts_fields = ti.field(dtype=data_type,
                                  shape=(batch_size * 1024, ),
                                  needs_grad=True)
        self.ws_fields = ti.field(dtype=data_type,
                                  shape=(batch_size * 1024, ),
                                  needs_grad=True)
        self.T = ti.field(dtype=data_type,
                          shape=(batch_size * 1024),
                          needs_grad=True)

        # rays level
        self.rays_a_fields = ti.field(dtype=ti.i64, shape=(batch_size, 3))
        self.total_samples_fields = ti.field(dtype=ti.i64,
                                             shape=(batch_size, ))
        self.opacity_fields = ti.field(dtype=data_type,
                                       shape=(batch_size, ),
                                       needs_grad=True)
        self.depth_fields = ti.field(dtype=data_type,
                                     shape=(batch_size, ),
                                     needs_grad=True)
        self.rgb_fields = ti.field(dtype=data_type,
                                   shape=(batch_size, 3),
                                   needs_grad=True)

        # preallocate tensor
        self.register_buffer('total_samples',
                             torch.zeros(batch_size, dtype=torch.int64))
        self.register_buffer('rgb', torch.zeros(batch_size,
                                                3,
                                                dtype=torch_type))
        self.register_buffer('opacity',
                             torch.zeros(batch_size, dtype=torch_type))
        self.register_buffer('depth', torch.zeros(batch_size,
                                                  dtype=torch_type))
        self.register_buffer('ws',
                             torch.zeros(batch_size * 1024, dtype=torch_type))

        self.register_buffer('sigma_grad',
                             torch.zeros(batch_size * 1024, dtype=torch_type))
        self.register_buffer(
            'rgb_grad', torch.zeros(batch_size * 1024, 3, dtype=torch_type))

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
                # If no output gradient is provided, no need to
                # automatically materialize it as torch.zeros.

                ctx.T_threshold = T_threshold
                ctx.samples_size = sigmas.shape[0]

                ws = self.ws[:sigmas.shape[0]]

                torch2ti(self.sigmas_fields, sigmas.contiguous())
                torch2ti(self.rgbs_fields, rgbs.contiguous())
                torch2ti(self.deltas_fields, deltas.contiguous())
                torch2ti(self.ts_fields, ts.contiguous())
                torch2ti(self.rays_a_fields, rays_a.contiguous())
                composite_train_fw(self.sigmas_fields, self.rgbs_fields,
                                   self.deltas_fields, self.ts_fields,
                                   self.rays_a_fields, T_threshold, self.T,
                                   self.total_samples_fields,
                                   self.opacity_fields, self.depth_fields,
                                   self.rgb_fields, self.ws_fields)
                ti2torch(self.total_samples_fields, self.total_samples)
                ti2torch(self.opacity_fields, self.opacity)
                ti2torch(self.depth_fields, self.depth)
                ti2torch(self.rgb_fields, self.rgb)


                return self.total_samples.sum(
                ), self.opacity, self.depth, self.rgb, ws

            @staticmethod
            @custom_bwd
            def backward(ctx, dL_dtotal_samples, dL_dopacity, dL_ddepth,
                         dL_drgb, dL_dws):

                T_threshold = ctx.T_threshold
                samples_size = ctx.samples_size

                sigma_grad = self.sigma_grad[:samples_size].contiguous()
                rgb_grad = self.rgb_grad[:samples_size].contiguous()

                self.zero_grad()

                torch2ti_grad(self.opacity_fields, dL_dopacity.contiguous())
                torch2ti_grad(self.depth_fields, dL_ddepth.contiguous())
                torch2ti_grad(self.rgb_fields, dL_drgb.contiguous())
                torch2ti_grad(self.ws_fields, dL_dws.contiguous())
                composite_train_fw.grad(self.sigmas_fields, self.rgbs_fields,
                                        self.deltas_fields, self.ts_fields,
                                        self.rays_a_fields, T_threshold,
                                        self.T, self.total_samples_fields,
                                        self.opacity_fields, self.depth_fields,
                                        self.rgb_fields, self.ws_fields)
                ti2torch_grad(self.sigmas_fields, sigma_grad)
                ti2torch_grad(self.rgbs_fields, rgb_grad)

                return sigma_grad, rgb_grad, None, None, None, None

        self._module_function = _module_function

    def zero_grad(self):
        self.sigmas_fields.grad.fill(0.)
        self.rgbs_fields.grad.fill(0.)
        self.T.grad.fill(0.)


    def forward(self, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
        return self._module_function.apply(sigmas, rgbs, deltas, ts, rays_a,
                                           T_threshold)
