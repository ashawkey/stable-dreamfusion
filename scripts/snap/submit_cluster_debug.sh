jobname=$1
timestamp=$(date +'%Y%m%d')
[ -d "$i" ] && echo "$i exists."
python scripts/snap/submit_cluster_job.py --yaml_folder scripts/snap/yamls \
        --gpu_memory 40 --gpu_num 1 --force_node --cpu_num_per_gpu 6 --memory_per_gpu 30.0 --replicas 1 \
        --project_name magic123 --project_support_alias img2mesh \
        --job_name gqian-$timestamp-$1 \
        --command "while :; do sleep 1000; done"
