export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/dreamshaper_8.safetensors"
export DATASET_CONFIG="/data/code/sd-scripts/experiments/config_ft.toml"
export OUTPUT_DIR="/data/train-output/style/50k_noperson_filtered"
export OUTPUT_NAME="fashion-styles_20k_b48_vqatags"

accelerate launch --num_cpu_threads_per_process 1 fine_tune.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL \
    --output_dir=$OUTPUT_DIR \
    --output_name=$OUTPUT_NAME \
    --logging_dir=$OUTPUT_DIR/logs \
    --log_with="tensorboard" \
    --dataset_config=$DATASET_CONFIG \
    --save_precision="float" \
    --save_model_as=safetensors \
    --save_every_n_epochs=4 \
    --learning_rate=5e-6 \
    --max_train_steps=20000 \
    --gradient_accumulation_steps=8 \
    --use_8bit_adam \
    --xformers \
    --gradient_checkpointing \
    --mixed_precision=fp16
