mkdir -p /mnt/sagemaker-nvme/hf-cache
export HF_HOME=/mnt/sagemaker-nvme/hf-cache
export TRANSFORMERS_CACHE=/mnt/sagemaker-nvme/hf-cache
export HF_HUB_CACHE=/mnt/sagemaker-nvme/hf-cache
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

model_name=deepseek_agent
# task_type=web_caption,webqa,heading_ocr,element_ocr,element_ground,action_prediction,action_ground
task_type=webqa

python $DEBUG_MODE run_agent.py \
    --model_name $model_name \
    --dataset_name_or_path visualwebbench/VisualWebBench \
    --task_type $task_type \
    --gpus 0

