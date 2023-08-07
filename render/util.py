# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import imageio

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

#----------------------------------------------------------------------------
# sRGB color transforms
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def reinhard(f: torch.Tensor) -> torch.Tensor:
    return f/(1+f)

#-----------------------------------------------------------------------------------
# Metrics (taken from jaxNerf source code, in order to replicate their measurements)
#
# https://github.com/google-research/google-research/blob/301451a62102b046bbeebff49a760ebeec9707b8/jaxnerf/nerf/utils.py#L266
#
#-----------------------------------------------------------------------------------

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * np.log(mse)

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

#----------------------------------------------------------------------------
# Displacement texture lookup
#----------------------------------------------------------------------------

def get_miplevels(texture: np.ndarray) -> float:
    minDim = min(texture.shape[0], texture.shape[1])
    return np.floor(np.log2(minDim))

def tex_2d(tex_map : torch.Tensor, coords : torch.Tensor, filter='nearest') -> torch.Tensor:
    tex_map = tex_map[None, ...]    # Add batch dimension
    tex_map = tex_map.permute(0, 3, 1, 2) # NHWC -> NCHW
    tex = torch.nn.functional.grid_sample(tex_map, coords[None, None, ...] * 2 - 1, mode=filter, align_corners=False)
    tex = tex.permute(0, 2, 3, 1) # NCHW -> NHWC
    return tex[0, 0, ...]

#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                ) # indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            ) # indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

#----------------------------------------------------------------------------
# Image scaling
#----------------------------------------------------------------------------

def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Behaves similar to tf.segment_sum
#----------------------------------------------------------------------------

def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    num_segments = torch.unique_consecutive(segment_ids).shape[0]

    # Repeats ids until same dimension as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], dtype=torch.int64, device='cuda')).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    result = torch.zeros(*shape, dtype=torch.float32, device='cuda')
    result = result.scatter_add(0, segment_ids, data)
    return result

#----------------------------------------------------------------------------
# Matrix helpers.
#----------------------------------------------------------------------------

def fovx_to_fovy(fovx, aspect):
    return np.arctan(np.tan(fovx / 2) / aspect) * 2.0

def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective_offcenter(fovy, fraction, rx, ry, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)

    # Full frustum
    R, L = aspect*y, -aspect*y
    T, B = y, -y

    # Create a randomized sub-frustum
    width  = (R-L)*fraction
    height = (T-B)*fraction
    xstart = (R-L)*rx
    ystart = (T-B)*ry

    l = L + xstart
    r = l + width
    b = B + ystart
    t = b + height
    
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    return torch.tensor([[2/(r-l),        0,  (r+l)/(r-l),              0], 
                         [      0, -2/(t-b),  (t+b)/(t-b),              0], 
                         [      0,        0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [      0,        0,           -1,              0]], dtype=torch.float32, device=device)

def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def scale(s, device=None):
    return torch.tensor([[ s, 0, 0, 0], 
                         [ 0, s, 0, 0], 
                         [ 0, 0, s, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def lookAt(eye, at, up):
    a = eye - at
    w = a / torch.linalg.norm(a)
    u = torch.cross(up, w)
    u = u / torch.linalg.norm(u)
    v = torch.cross(w, u)
    translate = torch.tensor([[1, 0, 0, -eye[0]], 
                              [0, 1, 0, -eye[1]], 
                              [0, 0, 1, -eye[2]], 
                              [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    rotate = torch.tensor([[u[0], u[1], u[2], 0], 
                           [v[0], v[1], v[2], 0], 
                           [w[0], w[1], w[2], 0], 
                           [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    return rotate @ translate

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation(device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.array([0,0,0]).astype(np.float32)
    return torch.tensor(m, dtype=torch.float32, device=device)

#----------------------------------------------------------------------------
# Compute focal points of a set of lines using least squares. 
# handy for poorly centered datasets
#----------------------------------------------------------------------------

def lines_focal(o, d):
    d = safe_normalize(d)
    I = torch.eye(3, dtype=o.dtype, device=o.device)
    S = torch.sum(d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...], dim=0)
    C = torch.sum((d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...]) @ o[..., None], dim=0).squeeze(1)
    return torch.linalg.pinv(S) @ C

#----------------------------------------------------------------------------
# Cosine sample around a vector N
#----------------------------------------------------------------------------
@torch.no_grad()
def cosine_sample(N, size=None):
    # construct local frame
    N = N/torch.linalg.norm(N)

    dx0 = torch.tensor([0, N[2], -N[1]], dtype=N.dtype, device=N.device)
    dx1 = torch.tensor([-N[2], 0, N[0]], dtype=N.dtype, device=N.device)

    dx = torch.where(dot(dx0, dx0) > dot(dx1, dx1), dx0, dx1)
    #dx = dx0 if np.dot(dx0,dx0) > np.dot(dx1,dx1) else dx1
    dx = dx / torch.linalg.norm(dx)
    dy = torch.cross(N,dx)
    dy = dy / torch.linalg.norm(dy)

    # cosine sampling in local frame
    if size is None:
        phi = 2.0 * np.pi * np.random.uniform()
        s = np.random.uniform()
    else:
        phi = 2.0 * np.pi * torch.rand(*size, 1, dtype=N.dtype, device=N.device)
        s = torch.rand(*size, 1, dtype=N.dtype, device=N.device)
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi)*sintheta
    y = np.sin(phi)*sintheta
    z = costheta

    # local to world
    return dx*x + dy*y + N*z

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Bilinear downsample log(spp) steps
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor, spp) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    g = x.shape[-1]
    w = w.expand(g, 1, 4, 4) 
    x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    steps = int(np.log2(spp))
    for _ in range(steps):
        xp = torch.nn.functional.pad(x, (1,1,1,1), mode='replicate')
        x = torch.nn.functional.conv2d(xp, w, padding=0, stride=2, groups=g)
    return x.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Singleton initialize GLFW
#----------------------------------------------------------------------------

_glfw_initialized = False
def init_glfw():
    global _glfw_initialized
    try:
        import glfw
        glfw.ERROR_REPORTING = 'raise'
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        test = glfw.create_window(8, 8, "Test", None, None) # Create a window and see if not initialized yet
    except glfw.GLFWError as e:
        if e.error_code == glfw.NOT_INITIALIZED:
            glfw.init()
            _glfw_initialized = True

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, title=None):
    # Import OpenGL
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image[..., 0:3]) if image.shape[-1] == 4 else np.asarray(image)
    height, width, channels = image.shape

    # Initialize window.
    init_glfw()
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.default_window_hints()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)

def save_image_raw(fn, x : np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)


def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255

#----------------------------------------------------------------------------

def time_to_text(x):
    if x > 3600:
        return "%.2f h" % (x / 3600)
    elif x > 60:
        return "%.2f m" % (x / 60)
    else:
        return "%.2f s" % x

#----------------------------------------------------------------------------

def checkerboard(res, checker_size) -> np.ndarray:
    tiles_y = (res[0] + (checker_size*2) - 1) // (checker_size*2)
    tiles_x = (res[1] + (checker_size*2) - 1) // (checker_size*2)
    check = np.kron([[1, 0] * tiles_x, [0, 1] * tiles_x] * tiles_y, np.ones((checker_size, checker_size)))*0.33 + 0.33
    check = check[:res[0], :res[1]]
    return np.stack((check, check, check), axis=-1)

