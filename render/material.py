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

from . import util
from . import texture

######################################################################################
# Wrapper to make materials behave like a python dict, but register textures as 
# torch.nn.Module parameters.
######################################################################################
class Material(torch.nn.Module):
    def __init__(self, mat_dict):
        super(Material, self).__init__()
        self.mat_keys = set()
        for key in mat_dict.keys():
            self.mat_keys.add(key)
            self[key] = mat_dict[key]

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        self.mat_keys.add(key)
        setattr(self, key, val)

    def __delitem__(self, key):
        self.mat_keys.remove(key)
        delattr(self, key)

    def keys(self):
        return self.mat_keys

######################################################################################
# .mtl material format loading / storing
######################################################################################
@torch.no_grad()
def load_mtl(fn, clear_ks=True):
    import re
    mtl_path = os.path.dirname(fn)

    # Read file
    with open(fn, 'r') as f:
        lines = f.readlines()

    # Parse materials
    materials = []
    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]
        if 'newmtl' in prefix:
            material = Material({'name' : data[0]})
            materials += [material]
        elif materials:
            if 'bsdf' in prefix or 'map_kd' in prefix or 'map_ks' in prefix or 'bump' in prefix:
                material[prefix] = data[0]
            else:
                material[prefix] = torch.tensor(tuple(float(d) for d in data), dtype=torch.float32, device='cuda')

    # Convert everything to textures. Our code expects 'kd' and 'ks' to be texture maps. So replace constants with 1x1 maps
    for mat in materials:
        if not 'bsdf' in mat:
            mat['bsdf'] = 'pbr'

        if 'map_kd' in mat:
            mat['kd'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_kd']))
        else:
            mat['kd'] = texture.Texture2D(mat['kd'])
        
        if 'map_ks' in mat:
            mat['ks'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_ks']), channels=3)
        else:
            mat['ks'] = texture.Texture2D(mat['ks'])

        if 'bump' in mat:
            mat['normal'] = texture.load_texture2D(os.path.join(mtl_path, mat['bump']), lambda_fn=lambda x: x * 2 - 1, channels=3)

        # Convert Kd from sRGB to linear RGB
        mat['kd'] = texture.srgb_to_rgb(mat['kd'])

        if clear_ks:
            # Override ORM occlusion (red) channel by zeros. We hijack this channel
            for mip in mat['ks'].getMips():
                mip[..., 0] = 0.0 

    return materials

@torch.no_grad()
def save_mtl(fn, material):
    folder = os.path.dirname(fn)
    with open(fn, "w") as f:
        f.write('newmtl defaultMat\n')
        if material is not None:
            f.write('bsdf   %s\n' % material['bsdf'])
            if 'kd' in material.keys():
                f.write('map_kd texture_kd.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_kd.png'), texture.rgb_to_srgb(material['kd']))
            if 'ks' in material.keys():
                f.write('map_ks texture_ks.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_ks.png'), material['ks'])
            if 'normal' in material.keys():
                f.write('bump texture_n.png\n')
                texture.save_texture2D(os.path.join(folder, 'texture_n.png'), material['normal'], lambda_fn=lambda x:(util.safe_normalize(x)+1)*0.5)
        else:
            f.write('Kd 1 1 1\n')
            f.write('Ks 0 0 0\n')
            f.write('Ka 0 0 0\n')
            f.write('Tf 1 1 1\n')
            f.write('Ni 1\n')
            f.write('Ns 0\n')

######################################################################################
# Merge multiple materials into a single uber-material
######################################################################################

def _upscale_replicate(x, full_res):
    x = x.permute(0, 3, 1, 2)
    x = torch.nn.functional.pad(x, (0, full_res[1] - x.shape[3], 0, full_res[0] - x.shape[2]), 'replicate')
    return x.permute(0, 2, 3, 1).contiguous()

def merge_materials(materials, texcoords, tfaces, mfaces):
    assert len(materials) > 0
    for mat in materials:
        assert mat['bsdf'] == materials[0]['bsdf'], "All materials must have the same BSDF (uber shader)"
        assert ('normal' in mat) is ('normal' in materials[0]), "All materials must have either normal map enabled or disabled"

    uber_material = Material({
        'name' : 'uber_material',
        'bsdf' : materials[0]['bsdf'],
    })

    textures = ['kd', 'ks', 'normal']

    # Find maximum texture resolution across all materials and textures
    max_res = None
    for mat in materials:
        for tex in textures:
            tex_res = np.array(mat[tex].getRes()) if tex in mat else np.array([1, 1])
            max_res = np.maximum(max_res, tex_res) if max_res is not None else tex_res
    
    # Compute size of compund texture and round up to nearest PoT
    full_res = 2**np.ceil(np.log2(max_res * np.array([1, len(materials)]))).astype(np.int)

    # Normalize texture resolution across all materials & combine into a single large texture
    for tex in textures:
        if tex in materials[0]:
            tex_data = torch.cat(tuple(util.scale_img_nhwc(mat[tex].data, tuple(max_res)) for mat in materials), dim=2) # Lay out all textures horizontally, NHWC so dim2 is x
            tex_data = _upscale_replicate(tex_data, full_res)
            uber_material[tex] = texture.Texture2D(tex_data)

    # Compute scaling values for used / unused texture area
    s_coeff = [full_res[0] / max_res[0], full_res[1] / max_res[1]]

    # Recompute texture coordinates to cooincide with new composite texture
    new_tverts = {}
    new_tverts_data = []
    for fi in range(len(tfaces)):
        matIdx = mfaces[fi]
        for vi in range(3):
            ti = tfaces[fi][vi]
            if not (ti in new_tverts):
                new_tverts[ti] = {}
            if not (matIdx in new_tverts[ti]): # create new vertex
                new_tverts_data.append([(matIdx + texcoords[ti][0]) / s_coeff[1], texcoords[ti][1] / s_coeff[0]]) # Offset texture coodrinate (x direction) by material id & scale to local space. Note, texcoords are (u,v) but texture is stored (w,h) so the indexes swap here
                new_tverts[ti][matIdx] = len(new_tverts_data) - 1
            tfaces[fi][vi] = new_tverts[ti][matIdx] # reindex vertex

    return uber_material, new_tverts_data, tfaces

