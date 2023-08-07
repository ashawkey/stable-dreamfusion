# python all_metrics/metric_utils.py --datasets  nerf4 realfusion15 --pred_pattern "all_outputs/magic123/magic123-2d1-3d30-dmtet" --results_folder "results/images"

# 2d only
python all_metrics/metric_utils.py --datasets  nerf4 realfusion15 --pred_pattern "all_outputs/magic123-2d/magic123-2d1*" --results_folder "results/images"

# 3d only
python all_metrics/metric_utils.py --datasets  nerf4 realfusion15 --pred_pattern "all_outputs/magic123-3d/zero123-z40*" --results_folder "results/images"

# python all_metrics/metric_utils.py --datasets nerf4 realfusion15 --pred_pattern "all_outputs/3d_fuse" --results_folder "color" --rgb_name "" --first_str "*_0_*"
# python all_metrics/metric_utils.py --datasets nerf4 realfusion15 --pred_pattern "all_outputs/neural_lift" --results_folder "albedo" --rgb_name "albedo" --first_str "*0000*"
# python all_metrics/metric_utils.py --datasets nerf4 realfusion15 --pred_pattern "all_outputs/real_fusion" --results_folder "colors" --rgb_name ""
# python all_metrics/metric_utils.py --datasets nerf4 realfusion15 --pred_pattern "all_outputs/shape" --results_folder "" --rgb_name "" --first_str "0."
# python all_metrics/metric_utils.py --datasets realfusion15 --pred_pattern "all_outputs/pointe-r100" --results_folder "" --rgb_name "" --first_str "0."