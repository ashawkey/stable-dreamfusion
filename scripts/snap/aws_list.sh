script_name=$1
runid=$2
runid2=$3
imagename=$4
run1=$5
run2=$6
arguments="${@:7}"


examples=(
    'data/realfusion15/two_donuts/'
    'data/realfusion15/watercolor_horse/'
)


timestamp=$(date +'%Y%m%d')
for i in "${examples[@]}"; do
    echo "$i"
    [ -d "$i" ] && echo "$i exists."
    example=$(basename $i)
    echo ${@:8}
    python scripts/snap/submit_cluster_job.py --yaml_folder scripts/snap/yamls \
        --gpu_memory 40 --gpu_num 1 --force_node --cpu_num_per_gpu 6 --memory_per_gpu 30.0 --replicas 1 \
        --project_name magic123 --project_support_alias img2mesh \
        --job_name gqian-$timestamp-$runid-$runid2-$example \
        --command "bash $script_name 0 $runid $runid2 $i $imagename $run1 $run2 $arguments "
done
