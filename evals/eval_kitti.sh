#!/bin/bash

abs_dir="/home/dominic/Documents"
submap_size=${1:-75} # Default to 75 if not provided
dataset_path="/home/dominic/Downloads/data_odometry_color/dataset/sequences"
gt_path="/home/dominic/Downloads/data_odometry_poses/dataset/poses"
log_path="$(pwd)/logs/kitti_results_w${submap_size}.txt"

mkdir -p "$(pwd)/logs"

datasets=(
    # 00
    # 01
    # 02
    # 03
    # 04
    05
    # 06
    # 07
    # 08
    # 09
    # 10
)

# Number of full runs
n=1  # <-- change as needed

# If file doesn't exist, write header
if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE" > "$log_path"
fi

for run in $(seq 1 $n); do
    echo "==== Starting Run $run ===="

    total_mean=0
    count=0

    for dataset in "${datasets[@]}"; do
        echo "Running main.py on $dataset (Run $run)"
        dataset_name="${dataset_path}/${dataset}/image_2"
        python main.py --image_folder "$dataset_name" --max_loops 1 --min_disparity 25 --vis_map --conf_threshold 50 --lc_thres 0.95 --submap_size "$submap_size" --log_results --skip_dense_log --log_path "$(pwd)/logs/${dataset}_run${run}_w${submap_size}.txt"
    done

    for dataset in "${datasets[@]}"; do
        echo "Evaluating $dataset (Run $run)"
        est_path="$(pwd)/logs/${dataset}_run${run}_w${submap_size}.txt"
        est_path_mod="$(pwd)/logs/${dataset}_run${run}_w${submap_size}c.txt"
        gt_file="${gt_path}${dataset}/groundtruth.txt"

        python evals/convert_kitti_keyframes.py --index_file $est_path --source_file "$gt_path/${dataset}.txt" --output_lines_file "$gt_path/${dataset}c.txt" --cleaned_index_file $est_path_mod

        ape_result=$(evo_ape kitti "$gt_path/${dataset}c.txt" $est_path_mod -as)
        mean=$(echo "$ape_result" | grep "mean" | head -1 | sed -E 's/.*mean[^0-9]*([0-9.]+).*/\1/')
        mean=${mean:-0}

        echo "$run,$dataset,$mean" >> "$log_path"

        total_mean=$(echo "$total_mean + $mean" | bc -l)
        count=$((count + 1))
    done

    avg_mean=$(echo "$total_mean / $count" | bc -l)
    echo "$run,Average,$avg_mean" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE for run $run: $avg_mean"
done
