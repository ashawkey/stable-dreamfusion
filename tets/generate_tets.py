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


'''
This code segment shows how to use Quartet: https://github.com/crawforddoran/quartet, 
to generate a tet grid 
1) Download, compile and run Quartet as described in the link above. Example usage `quartet meshes/cube.obj 0.5 cube_5.tet`
2) Run the function below to generate a file `cube_32_tet.tet`
'''

def generate_tetrahedron_grid_file(res=32, root='..'):
    frac = 1.0 / res
    command = 'cd %s/quartet; ' % (root) + \
                './quartet_release meshes/cube.obj %f meshes/cube_%f_tet.tet -s meshes/cube_boundary_%f.obj' % (frac, res, res)
    os.system(command)


'''
This code segment shows how to convert from a quartet .tet file to compressed npz file
'''
def convert_from_quartet_to_npz(quartetfile = 'cube_32_tet.tet', npzfile = '32_tets.npz'):

    file1 = open(quartetfile, 'r')
    header = file1.readline()
    numvertices = int(header.split(" ")[1])
    numtets     = int(header.split(" ")[2])
    print(numvertices, numtets)

    # load vertices
    vertices = np.loadtxt(quartetfile, skiprows=1, max_rows=numvertices)
    vertices = vertices - 0.5
    print(vertices.shape, vertices.min(), vertices.max())

    # load indices
    indices = np.loadtxt(quartetfile, dtype=int, skiprows=1+numvertices, max_rows=numtets)
    print(indices.shape)

    np.savez_compressed(npzfile, vertices=vertices, indices=indices)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=32)
    parser.add_argument('--root', type=str, default='..')
    args = parser.parse_args()

    generate_tetrahedron_grid_file(res=args.res, root=args.root)
    convert_from_quartet_to_npz(quartetfile=os.path.join(args.root, 'quartet', 'meshes', f'cube_{args.res}.000000_tet.tet'), npzfile=os.path.join('./tets', f'{args.res}_tets.npz'))