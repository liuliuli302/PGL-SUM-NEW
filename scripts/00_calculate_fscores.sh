dataset_root_path="$HOME/workspace/PGL-SUM-NEW/data/datasets"
dataset_name="TVSum"
frame_interval=16

python evaluation/compute_fscores_for_shell.py \
    --root_path $dataset_root_path \
    --dataset_name $dataset_name \
    --frame_interval $frame_interval \
    --eval_method "avg" \

echo "Finished calculating fscores scores for $dataset_name."
echo "The results are saved in $dataset_root_path/$dataset_name/result.json"