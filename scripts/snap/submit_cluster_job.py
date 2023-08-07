
import yaml
import os
import argparse


def generate_yaml(
                  gpu_memory=40, # 40G or 80G?
                  gpu_num=1, 
                  cpu_num_per_gpu=6,
                  memory_per_gpu=30,
                  replicas=1,
                  project_name='magic123',
                  project_support_alias='img2mesh',
                  pre_run_event='mkdir -p /fsx/code && ln -s /nfs/code/gqian /fsx/code/ && cd /fsx/code/gqian/img2mesh', 
                  command="while :; do sleep 1000; done", 
                  job_name='debug',  
                  force_node=False,
                  **kwargs
                  ):
    data = {
        'docker_image': '440036398022.dkr.ecr.us-west-2.amazonaws.com/facecraft-ml:efa',
        'project_name': project_name,
        'project_support_alias': project_support_alias,
        'team': 'creative_vision',
        #'fsx': 'fs-0b933bba2f17fe699', # 100T genai filesystem
        'fsx': 'fs-056caaa56fa5cc5f3', # 2T personal filesystem of gqian
        'gpu_type': 'nvidia-tesla-a100',
        'gpu_num': gpu_num,
        'cpu_num': str(int(cpu_num_per_gpu * gpu_num)),
        'memory': str(int(memory_per_gpu * gpu_num)),
        'gpu_memory': str(int(gpu_memory)),
        'pytorchjob': {
            'replicas': replicas
        },
        'efa': True,
        'script': {
            'pre_run_event': str(pre_run_event),
            'command': str(command),
            'jobs': [
                {'name': job_name}
            ]
        }
    }
    if gpu_num == 1 and not force_node:
        gpu_yaml = {
            'custom_node_labels': {
            'use_case': 'p4d_debug',
        },
        }
    else:
        gpu_yaml = {
            'custom_node_labels': {
            'snap.com/spine': 'unknown',
            'snap.com/region': 'us-west-2c', 
        },
        }
    data.update(gpu_yaml) 
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate yaml file for AWS Pytorch Job")
    parser.add_argument('--yaml_folder', type=str, default='./', help='path to save the yaml folder')
    parser.add_argument('--gpu_memory', type=int, default=40, help='GPU memory (in GB)')
    parser.add_argument('--gpu_num', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--cpu_num_per_gpu', type=int, default=6, help='Number of CPUs per GPU')
    parser.add_argument('--memory_per_gpu', type=float, default=30.0, help='Memory per GPU')
    parser.add_argument('--replicas', type=int, default=1, help='Number of replicas')
    parser.add_argument('--project_name', type=str, default='magic123', help='Project name')
    parser.add_argument('--project_support_alias', type=str, default='img2mesh', help='Project support alias')
    #parser.add_argument('--pre_run_event', type=str, default='export PATH=/nfs/code/gqian/miniconda3/bin:$PATH && conda init bash && source ~/.bashrc && cd /nfs/code/gqian/img2mesh ', help='Pre-run event command')
    parser.add_argument('--pre_run_event', type=str, default='cd /nfs/code/gqian/img2mesh', help='Pre-run event command')
    parser.add_argument('--command', type=str, default='while :; do sleep 1000; done', help='Command')
    parser.add_argument('--job_name', type=str, default='debug', help='Job name')
    parser.add_argument('--force_node', action='store_true',
                         help="use normal cluster not debug cluster")
    args, unknown = parser.parse_known_args()
    args.job_name = args.job_name.replace('_', '-').replace('.', '-')[:51] # do not support _ and . in job name, and max length is limited (around 70)

    # "bash scripts/magic123/run_single_bothpriors.sh 0 r256 data/nerf4/drums rgba.png --h 300 --w 300"
    data = generate_yaml(**vars(args))
    yaml_str = yaml.safe_dump(data)

    # Write the YAML content to a file
    os.makedirs(args.yaml_folder, exist_ok=True)
    yaml_path = os.path.join(args.yaml_folder, f'{args.job_name}.yaml')
    with open(yaml_path, 'w') as file:
        file.write(yaml_str)
    print(f'YAML file saved to {yaml_path}')
    
    # launch the job using snap_rutls
    os.system(f'yes yes | snap_rutils cluster run {yaml_path} -s')
    
    # show the job status
    os.system(f'kubectl get pods | grep {args.job_name} ')

    # show the job logs
    os.system(f'kubectl logs {args.job_name}-worker-0')
