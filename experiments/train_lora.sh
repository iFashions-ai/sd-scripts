export PRETRAINED_MODEL="/data/code/stable-diffusion-webui/models/Stable-diffusion/SDXL/sd_xl_base_1.0.safetensors"
export DATASET_CONFIG="/data/code/sd-scripts/experiments/config_lora.toml"
export OUTPUT_DIR="/data/train-output/style/SDXL_310k_LoRA_dim8"
export OUTPUT_NAME="fashion-styles_310kdata_b16_lora"

accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL \
    --dataset_config=$DATASET_CONFIG \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$OUTPUT_DIR/logs \
    --log_with="tensorboard" \
    --output_name=$OUTPUT_NAME \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=5000 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --optimizer_type="Adafactor" \
    --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=100 \
    --xformers \
    --mixed_precision="fp16" \
    --gradient_checkpointing \
    --save_every_n_steps=500 \
    --network_module=networks.lora \
    --network_dim=8 \
    --network_alpha=1 \
    --network_train_unet_only \
    --no_half_vae \
    --cache_latents \
    --cache_latents_to_disk \


# --learning_rate=1e-4 \
# --network_args "conv_dim=4" "conv_alpha=1" \
