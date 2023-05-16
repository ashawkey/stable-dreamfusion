import os
import numpy as np
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm
import pyvista as pv

def render_video(anim_mesh):
    center = anim_mesh.center_mass
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(anim_mesh)

    radius = 10
    n_frames = 360  
    angle_step = 2 * np.pi / n_frames  
    for i in tqdm(range(n_frames)):
        camera_pos = [center[0] + radius * np.cos(i*angle_step),center[1] + radius *np.sin(i*angle_step),center[2]]
        plotter.camera_position = (camera_pos, center, (0, 0, 1))
        plotter.show(screenshot=f'frame_{i}.png', auto_close=False)
    plotter.close()
    os.system('ffmpeg -r 30 -f image2 -s 1920x1080 -i "result/frame_%d.png" -vcodec libx264 -crf 25  -pix_fmt yuv420p result/output.mp4')



def generate_mesh(obj1,obj2,transform_vector):

    # Read 2 objects
    filename1 = obj1 # Central Object
    filename2 = obj2 # Surrounding Object
    mesh1 = trimesh.load_mesh(filename1)
    mesh2 = trimesh.load_mesh(filename2)

    extents1 = mesh1.extents
    extents2 = mesh1.extents
    
    radius1 = sum(extents1) / 3.0
    radius2 = sum(extents2) / 3.0

    center1 = mesh1.center_mass
    center2 = mesh2.center_mass

    # Move
    T1 = -center1
    new =[]
    for i in transform_vector:
        try:
            new.append(float(i))*radius1
        except:
            pass
    transform_vector = new
    print(T1, transform_vector, radius1)
    T2 = -center2 + transform_vector

    # Transform
    mesh1.apply_translation(T1)
    mesh2.apply_translation(T2)

    # merge mesh
    merged_mesh = trimesh.util.concatenate((mesh1, mesh2))

    # save mesh
    merged_mesh.export('merged_mesh.obj')
    print("----> merge mesh done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate rotating mesh animation.')
    parser.add_argument('--center_obj', type=str, help='Input OBJ1 file.')
    parser.add_argument('--surround_obj', type=str, help='Input OBJ2 file.')
    parser.add_argument('--transform_vector', help='Transform_vector.')
    parser.add_argument('--output_file', type=str, default="result/Demo.mp4", help='Output MP4 file.')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames to render.')
    args = parser.parse_args()
    
    #mesh = obj.Obj("wr.obj")
    generate_mesh(args.center_obj,args.surround_obj,args.transform_vector)

    input_file = Path("merged_mesh.obj")
    output_file = Path(args.output_file)

    out_dir = output_file.parent.joinpath('frames')
    out_dir.mkdir(parents=True, exist_ok=True)

    anim_mesh = trimesh.load_mesh(str(input_file))

    render_video(anim_mesh)

