device=$1
runid=$2 # jobname for the first stage
runid2=$3 # jobname for the second stage
imagename=$4
step1=$5
step2=$6

examples=(
    'data/realfusion15/teddy_bear/'
    'data/realfusion15/mental_dragon_statue/'
    'data/realfusion15/colorful_teapot/'
    'data/realfusion15/fish_real_nemo/'
    'data/realfusion15/two_cherries' 
    'data/realfusion15/watercolor_horse/'
    'data/nerf4/chair'
    'data/nerf4/drums'
    'data/nerf4/ficus'
    'data/nerf4/mic'
)

for i in "${examples[@]}"; do
    echo "$i"
    [ -d "$i" ] && echo "$i exists."
    bash scripts/magic123/run_both_priors.sh $device $runid "$i" $imagename $step1 $step2 ${@:7}
done
