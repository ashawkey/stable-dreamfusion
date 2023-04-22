import numpy as np
import pymesh
import argparse

from pathlib import Path
from tqdm import tqdm


def generate_frames(anim_mesh, outfile_path, num_frames=100):
    """
    生成每个帧的 mesh 对象
    """
    print('Generating frames ...')
    for frame_num in tqdm(range(num_frames)):
        angle_deg = frame_num * 2 * np.pi / 100
        R = np.array([
            [np.cos(angle_deg), -np.sin(angle_deg), 0],
            [np.sin(angle_deg), np.cos(angle_deg), 0],
            [0, 0, 1]
        ])

        rotated_mesh = anim_mesh.copy()
        rotated_mesh.vertices = np.dot(rotated_mesh.vertices, R.T)

        # 生成文件名并进行保存
        outfile = outfile_path.joinpath(f'frame_{frame_num:05d}.obj')
        pymesh.save_mesh(str(outfile), rotated_mesh)


def render_video(in_path, out_path):
    """
    将 obj 文件转换为 mp4 视频
    """
    print('Rendering video ...')
    cmd = f"ffmpeg -i {in_path} -vcodec libx264 -crf 25 {out_path}"
    os.system(cmd)
    print(f'Video saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate rotating mesh animation.')
    parser.add_argument('input_file', type=str, help='Input OBJ file.')
    parser.add_argument('output_file', type=str, help='Output MP4 file.')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames to render.')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    out_dir = output_file.parent.joinpath('frames')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 mesh 文件
    anim_mesh = pymesh.load_mesh(str(input_file))

    # 生成每一帧的 mesh 文件
    generate_frames(anim_mesh, out_dir, num_frames=args.num_frames)

    # 将 obj 文件转换为 mp4 视频
    render_video(str(out_dir.joinpath('frame_%05d.obj')), str(output_file))
