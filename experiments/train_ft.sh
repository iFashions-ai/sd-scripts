accelerate launch --num_cpu_threads_per_process 1 fine_tune.py \
    --pretrained_model_name_or_path="/data/code/stable-diffusion-webui/models/Stable-diffusion/dreamshaper_8.safetensors" \
    --output_dir="/data/train-output/style" \
    --output_name="fashion-styles_20k_b16_vqatags.ft" \
    --dataset_config="/data/code/sd-scripts/experiments/config_ft.toml" \
    --save_model_as=safetensors \
    --learning_rate=5e-6 \
    --max_train_steps=20000 \
    --gradient_accumulation_steps=4 \
    --use_8bit_adam \
    --xformers \
    --gradient_checkpointing \
    --mixed_precision=fp16 \

