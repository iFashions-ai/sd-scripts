# export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/dreamshaper_8.safetensors"
export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v51VAE.safetensors"
export DATASET_CONFIG="/data/code/sd-scripts/experiments/config_ft.toml"
export OUTPUT_DIR="/data/train-output/style/100k_noperson_filtered"
export OUTPUT_NAME="fashion-styles_100kdata_40ksteps_b24_filtered"

accelerate launch --num_cpu_threads_per_process 1 fine_tune.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL \
    --output_dir=$OUTPUT_DIR \
    --output_name=$OUTPUT_NAME \
    --logging_dir=$OUTPUT_DIR/logs \
    --log_with="tensorboard" \
    --dataset_config=$DATASET_CONFIG \
    --save_model_as=safetensors \
    --save_every_n_epochs=2 \
    --learning_rate=5e-6 \
    --max_train_steps=40000 \
    --gradient_accumulation_steps=4 \
    --use_8bit_adam \
    --xformers \
    --gradient_checkpointing \
    --mixed_precision=fp16 \
    --cache_latents
