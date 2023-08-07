script_name=$1
runid=$2
runid2=$3
i=$4
imagename=$5
run1=$6
run2=$7
arguments="${@:8}"

timestamp=$(date +'%Y%m%d')
[ -d "$i" ] && echo "$i exists."
example=$(basename $i)
echo ${@:8}
python scripts/snap/submit_cluster_job.py --yaml_folder scripts/snap/yamls \
        --gpu_memory 40 --gpu_num 1 --force_node --cpu_num_per_gpu 6 --memory_per_gpu 30.0 --replicas 1 \
        --project_name magic123 --project_support_alias img2mesh \
        --job_name gqian-$timestamp-$runid-$runid2-$example \
        --command "bash $script_name 0 $runid $runid2 $i $imagename $run1 $run2 $arguments "
