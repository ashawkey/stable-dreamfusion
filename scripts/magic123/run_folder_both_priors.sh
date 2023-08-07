device=$1
runid=$2 # jobname for the first stage
runid2=$3 # jobname for the second stage
topdir=$4   # path to the directory containing the images, e.g. data/nerf4
imagename=$5
step1=$6
step2=$7

for i in $topdir/*; do
    echo "$i"
    [ -d "$i" ] && echo "$i exists."
    bash scripts/magic123/run_both_priors.sh $device $runid "$i" $imagename $step1 $step2 ${@:8}
done
