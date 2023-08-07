# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import os
import sys
import torch
import torch.utils.cpp_extension

from .bsdf import *
from .loss import *

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/mesh.cu',
        'c_src/loss.cu',
        'c_src/bsdf.cu',
        'c_src/normal.cu',
        'c_src/cubemap.cu',
        'c_src/common.cpp',
        'c_src/torch_bindings.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False), 'lock')
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='renderutils_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)

    # Import, cache, and return the compiled module.
    import renderutils_plugin
    _cached_plugin = renderutils_plugin
    return _cached_plugin

#----------------------------------------------------------------------------
# Internal kernels, just used for testing functionality

class _fresnel_shlick_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, f90, cosTheta):
        out = _get_plugin().fresnel_shlick_fwd(f0, f90, cosTheta, False)
        ctx.save_for_backward(f0, f90, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        f0, f90, cosTheta = ctx.saved_variables
        return _get_plugin().fresnel_shlick_bwd(f0, f90, cosTheta, dout) + (None,)

def _fresnel_shlick(f0, f90, cosTheta, use_python=False):
    if use_python:
        out = bsdf_fresnel_shlick(f0, f90, cosTheta)
    else:
        out = _fresnel_shlick_func.apply(f0, f90, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _fresnel_shlick contains inf or NaN"
    return out


class _ndf_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = _get_plugin().ndf_ggx_fwd(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return _get_plugin().ndf_ggx_bwd(alphaSqr, cosTheta, dout) + (None,)

def _ndf_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_ndf_ggx(alphaSqr, cosTheta)
    else:
        out = _ndf_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _ndf_ggx contains inf or NaN"
    return out

class _lambda_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = _get_plugin().lambda_ggx_fwd(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return _get_plugin().lambda_ggx_bwd(alphaSqr, cosTheta, dout) + (None,)

def _lambda_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_lambda_ggx(alphaSqr, cosTheta)
    else:
        out = _lambda_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _lambda_ggx contains inf or NaN"
    return out

class _masking_smith_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosThetaI, cosThetaO):
        ctx.save_for_backward(alphaSqr, cosThetaI, cosThetaO)
        out = _get_plugin().masking_smith_fwd(alphaSqr, cosThetaI, cosThetaO, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosThetaI, cosThetaO = ctx.saved_variables
        return _get_plugin().masking_smith_bwd(alphaSqr, cosThetaI, cosThetaO, dout) + (None,)

def _masking_smith(alphaSqr, cosThetaI, cosThetaO, use_python=False):
    if use_python:
        out = bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO)
    else:
        out = _masking_smith_func.apply(alphaSqr, cosThetaI, cosThetaO)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _masking_smith contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Shading normal setup (bump mapping + bent normals)

class _prepare_shading_normal_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
        out = _get_plugin().prepare_shading_normal_fwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl, False)
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return out

    @staticmethod
    def backward(ctx, dout):
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
        return _get_plugin().prepare_shading_normal_bwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl) + (None, None, None)

def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''    

    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    if use_python:
        out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    else:
        out = _prepare_shading_normal_func.apply(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# BSDF functions

class _lambert_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi):
        out = _get_plugin().lambert_fwd(nrm, wi, False)
        ctx.save_for_backward(nrm, wi)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi = ctx.saved_variables
        return _get_plugin().lambert_bwd(nrm, wi, dout) + (None,)

def lambert(nrm, wi, use_python=False):
    '''Lambertian bsdf. 
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_lambert(nrm, wi)
    else:
        out = _lambert_func.apply(nrm, wi)
 
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out

class _frostbite_diffuse_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi, wo, linearRoughness):
        out = _get_plugin().frostbite_fwd(nrm, wi, wo, linearRoughness, False)
        ctx.save_for_backward(nrm, wi, wo, linearRoughness)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi, wo, linearRoughness = ctx.saved_variables
        return _get_plugin().frostbite_bwd(nrm, wi, wo, linearRoughness, dout) + (None,)

def frostbite_diffuse(nrm, wi, wo, linearRoughness, use_python=False):
    '''Frostbite, normalized Disney Diffuse bsdf. 
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        wo: World space camera vector.
        linearRoughness: Material roughness
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_frostbite(nrm, wi, wo, linearRoughness)
    else:
        out = _frostbite_diffuse_func.apply(nrm, wi, wo, linearRoughness)
 
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out

class _pbr_specular_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, col, nrm, wo, wi, alpha, min_roughness):
        ctx.save_for_backward(col, nrm, wo, wi, alpha)
        ctx.min_roughness = min_roughness
        out = _get_plugin().pbr_specular_fwd(col, nrm, wo, wi, alpha, min_roughness, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        col, nrm, wo, wi, alpha = ctx.saved_variables
        return _get_plugin().pbr_specular_bwd(col, nrm, wo, wi, alpha, ctx.min_roughness, dout) + (None, None)

def pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08, use_python=False):
    '''Physically-based specular bsdf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        col: Specular lobe color
        nrm: World space shading normal.
        wo: World space camera vector.
        wi: World space light vector
        alpha: Specular roughness parameter with shape [minibatch_size, height, width, 1]
        min_roughness: Scalar roughness clamping threshold

        use_python: Use PyTorch implementation (for validation)
    Returns:
        Shaded specular color
    '''

    if use_python:
        out = bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=min_roughness)
    else:
        out = _pbr_specular_func.apply(col, nrm, wo, wi, alpha, min_roughness)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_specular contains inf or NaN"
    return out

class _pbr_bsdf_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
        ctx.min_roughness = min_roughness
        ctx.BSDF = BSDF
        out = _get_plugin().pbr_bsdf_fwd(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_variables
        return _get_plugin().pbr_bsdf_bwd(kd, arm, pos, nrm, view_pos, light_pos, ctx.min_roughness, ctx.BSDF, dout) + (None, None, None)

def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, bsdf="lambert", use_python=False):
    '''Physically-based bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold
        bsdf: Controls diffuse BSDF, can be either 'lambert' or 'frostbite'

        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded color.
    '''    

    BSDF = 0 
    if bsdf == 'frostbite':
        BSDF = 1

    if use_python:
        out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)
    else:
        out = _pbr_bsdf_func.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_bsdf contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        out = _get_plugin().diffuse_cubemap_fwd(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, = ctx.saved_variables
        cubemap_grad = _get_plugin().diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad, None

def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out

class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        out = _get_plugin().specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = _get_plugin().specular_cubemap_bwd(cubemap, bounds, dout, ctx.roughness, ctx.theta_cutoff)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = _get_plugin().specular_bounds(res, costheta[idx])

    return costheta[idx], bounds
__ndfBoundsDict = {}

def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]

#----------------------------------------------------------------------------
# Fast image loss function

class _image_loss_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, target, loss, tonemapper):
        ctx.loss, ctx.tonemapper = loss, tonemapper
        ctx.save_for_backward(img, target)
        out = _get_plugin().image_loss_fwd(img, target, loss, tonemapper, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        img, target = ctx.saved_variables
        return _get_plugin().image_loss_bwd(img, target, dout, ctx.loss, ctx.tonemapper) + (None, None, None)

def image_loss(img, target, loss='l1', tonemapper='none', use_python=False):
    '''Compute HDR image loss. Combines tonemapping and loss into a single kernel for better perf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        img: Input image.
        target: Target (reference) image. 
        loss: Type of loss. Valid options are ['l1', 'mse', 'smape', 'relmse']
        tonemapper: Tonemapping operations. Valid options are ['none', 'log_srgb']
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Image space loss (scalar value).
    '''
    if use_python:
        out = image_loss_fn(img, target, loss, tonemapper)
    else:
        out = _image_loss_func.apply(img, target, loss, tonemapper)
        out = torch.sum(out) / (img.shape[0]*img.shape[1]*img.shape[2])

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Transform points function

class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return _get_plugin().xfm_fwd(points, matrix, isPoints, False)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        return (_get_plugin().xfm_bwd(points, matrix, dout, ctx.isPoints),) + (None, None, None)

def xfm_points(points, matrix, use_python=False):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    if use_python:
        out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    else:
        out = _xfm_func.apply(points, matrix, True)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def xfm_vectors(vectors, matrix, use_python=False):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    if use_python:
        out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    else:
        out = _xfm_func.apply(vectors, matrix, False)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out



