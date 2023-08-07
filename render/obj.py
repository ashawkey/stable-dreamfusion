# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import torch

from . import texture
from . import mesh
from . import material

######################################################################################
# Utility functions
######################################################################################

def _find_mat(materials, name):
    for mat in materials:
        if mat['name'] == name:
            return mat
    return materials[0] # Materials 0 is the default

######################################################################################
# Create mesh object from objfile
######################################################################################

def load_obj(filename, clear_ks=True, mtl_override=None):
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load materials
    all_materials = [
        {
            'name' : '_default_mat',
            'bsdf' : 'pbr',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }
    ]
    if mtl_override is None: 
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'mtllib':
                all_materials += material.load_mtl(os.path.join(obj_path, line.split()[1]), clear_ks) # Read in entire material library
    else:
        all_materials += material.load_mtl(mtl_override)

    # load vertices
    vertices, texcoords, normals  = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'usemtl': # Track used materials
            mat = _find_mat(all_materials, line.split()[1])
            if not mat in used_materials:
                used_materials.append(mat)
            activeMatIdx = used_materials.index(mat)
        elif prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if vv[2] != "" else -1
            for i in range(nv - 2): # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)

    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    else:
        uber_material = used_materials[0]

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda') if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device='cuda') if len(normals) > 0 else None
    
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None

    print(f'[load_obj] vertices: {vertices.shape}, faces: {faces.shape}')
    print(f'[load_obj] texcoords: {texcoords.shape if texcoords is not None else "None"}')

    return mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, material=uber_material)

######################################################################################
# Save mesh object to objfile
######################################################################################

def write_obj(folder, mesh, save_material=True):
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    if save_material:
        mtl_file = os.path.join(folder, 'mesh.mtl')
        print("Writing material: ", mtl_file)
        material.save_mtl(mtl_file, mesh.material)

    print("Done exporting mesh")
