# export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/dreamshaper_8.safetensors"
export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/realisticVisionV51_v51VAE.safetensors"
export DATASET_CONFIG="/data/code/sd-scripts/experiments/config_ft.toml"
export OUTPUT_DIR="/data/train-output/style/280kdata_20ksteps_b28_filtered"
export OUTPUT_NAME="fashion-styles_280kdata_20ksteps_b28_filtered"

accelerate launch --num_cpu_threads_per_process 1 fine_tune.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL \
    --output_dir=$OUTPUT_DIR \
    --output_name=$OUTPUT_NAME \
    --logging_dir=$OUTPUT_DIR/logs \
    --log_with="tensorboard" \
    --dataset_config=$DATASET_CONFIG \
    --save_model_as=safetensors \
    --save_every_n_steps=10000 \
    --learning_rate=5e-6 \
    --max_train_steps=20000 \
    --gradient_accumulation_steps=7 \
    --use_8bit_adam \
    --xformers \
    --gradient_checkpointing \
    --mixed_precision=fp16 \
    --cache_latents \
    --cache_latents_to_disk

# # export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/dreamshaper_8.safetensors"
# export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/SDXL/sd_xl_base_1.0.safetensors"
# export DATASET_CONFIG="/data/code/sd-scripts/experiments/config_ft.toml"
# export OUTPUT_DIR="/data/train-output/style/SDXL_100k_FT"
# export OUTPUT_NAME="fashion-styles_100kdata_b4"

# accelerate launch --num_cpu_threads_per_process 1 sdxl_train.py \
#     --pretrained_model_name_or_path=$PRETRAINED_MODEL \
#     --output_dir=$OUTPUT_DIR \
#     --output_name=$OUTPUT_NAME \
#     --logging_dir=$OUTPUT_DIR/logs \
#     --log_with="tensorboard" \
#     --dataset_config=$DATASET_CONFIG \
#     --save_model_as=safetensors \
#     --save_every_n_steps=1000 \
#     --learning_rate=4e-7 \
#     --max_train_steps=25000 \
#     --gradient_accumulation_steps=4 \
#     --optimizer_type="Adafactor" \
#     --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" \
#     --lr_scheduler="constant_with_warmup" \
#     --lr_warmup_steps=100 \
#     --xformers \
#     --gradient_checkpointing \
#     --mixed_precision=fp16 \
#     --no_half_vae \
#     --cache_latents \
#     --cache_latents_to_disk \
#     --cache_text_encoder_outputs \
#     --cache_text_encoder_outputs_to_disk
