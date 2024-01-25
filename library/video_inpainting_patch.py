from typing import Any, Dict, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from PIL import Image

from library.sdxl_lpw_stable_diffusion import (
    SdxlStableDiffusionLongPromptWeightingPipeline,
    preprocess_image,
)


class VideoInpaintingPatch(nn.Module):
    def __init__(self, vae_scale_factor: float, output_channel: int = 320):
        super().__init__()
        self.vae_scale_factor = vae_scale_factor
        self.output_channel = output_channel

        self.head_curr = nn.Conv2d(
            5, self.output_channel, 3, padding=1, padding_mode="replicate", bias=False
        )
        self.head_prev = nn.Conv2d(
            5, self.output_channel, 3, padding=1, padding_mode="replicate", bias=False
        )
        self.reset_parameters()

    def forward(self, vae: AutoencoderKL, image, mask, prev_image, prev_mask):
        # remove the image[mask] and prev_image[~prev_mask] parts
        mask = (mask > 0).to(image.dtype)[:, None]
        latent, latent_mask = self.encode_vae(vae, image, mask, apply_mask=True)

        prev_mask = (prev_mask <= 0).to(prev_image.dtype)[:, None]
        prev_latent, prev_latent_mask = self.encode_vae(
            vae, prev_image, prev_mask, apply_mask=True
        )

        prev_feat = self.head_prev(torch.cat([prev_latent, prev_latent_mask], dim=1))
        curr_feat = self.head_curr(torch.cat([latent, latent_mask], dim=1))
        return curr_feat + prev_feat

    def encode_vae(self, vae: AutoencoderKL, image, mask, apply_mask: bool):
        vae_dtype = next(vae.parameters()).dtype
        with torch.no_grad():
            if apply_mask:
                image = image * (1 - mask) + 0.5 * mask
            latent = vae.encode(image.to(vae_dtype)).latent_dist.sample()
            if torch.any(torch.isnan(latent)):
                latent = torch.nan_to_num(latent, 0, out=latent)
            latent *= self.vae_scale_factor

            H, W = latent.shape[-2:]
            latent_mask = nn.functional.interpolate(
                mask, size=(H * 8, W * 8), mode="bilinear"
            ).round()
            latent_mask = nn.functional.max_pool2d(latent_mask, (8, 8)).round()
        return latent, latent_mask

    def reset_parameters(self):
        # zero initialization
        nn.init.zeros_(self.head_curr.weight)
        nn.init.zeros_(self.head_prev.weight)


class UnetPatched:
    def __init__(self, unet, input_block_addons: Dict[int, torch.Tensor]):
        self.unet = unet
        self.dtype = self.unet.dtype
        self.in_channels = self.unet.in_channels

        self.input_block_addons = input_block_addons

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.unet(*args, **kwds, input_block_addons=self.input_block_addons)


class VideoInpaintingPatchPipeline(SdxlStableDiffusionLongPromptWeightingPipeline):
    def __init__(self, *args, inpainting_head: VideoInpaintingPatch, **kwargs):
        super().__init__(*args, **kwargs)
        self.inpainting_head = inpainting_head
        self.original_unet = self.unet

    def __call__(
        self,
        *args,
        image: Union[torch.FloatTensor, Image.Image],
        mask_image: Union[torch.FloatTensor, Image.Image],
        prev_image: Union[torch.FloatTensor, Image.Image],
        prev_mask: Union[torch.FloatTensor, Image.Image],
        **kwargs
    ):
        dtype = self.unet.dtype

        def preprocess_mask(mask: Image.Image, target_size):
            mask = mask.resize(target_size, resample=Image.NEAREST)
            mask = np.array(mask.convert("L"))
            mask = (mask > 0).astype(np.uint8) * 255

            # enlarge mask
            def enlarge_mask(mask, kscale=0.05):
                if kscale <= 0:
                    return mask
                a = max(int(np.sqrt(mask.shape[0] * mask.shape[1]) * kscale), 11)
                return cv2.dilate(mask, np.ones((a, a), np.uint8))

            mask = enlarge_mask(mask, kscale=0)

            return torch.from_numpy(mask[None])

        def convert_mask_for_pipeline(mask: torch.Tensor):
            mask = nn.functional.max_pool2d(
                mask, (self.vae_scale_factor, self.vae_scale_factor)
            )
            mask = (mask[0] > 0).to(mask.dtype)
            mask = torch.tile(mask, (4, 1, 1))[None]
            mask = 1 - mask  # repaint white, keep black
            return mask

        # preprocess image and mask
        if isinstance(image, Image.Image):
            image = preprocess_image(image)
        if isinstance(prev_image, Image.Image):
            prev_image = preprocess_image(prev_image)
        image_h, image_w = image.shape[-2:]
        image_wh = (image_w, image_h)

        if isinstance(mask_image, Image.Image):
            mask_image = preprocess_mask(mask_image, target_size=image_wh)
        if isinstance(prev_mask, Image.Image):
            prev_mask = preprocess_mask(prev_mask, target_size=image_wh)

        image = image.to(device=self.device, dtype=dtype)
        prev_image = prev_image.to(device=self.device, dtype=dtype)
        mask_image = mask_image.to(device=self.device, dtype=dtype)
        prev_mask = prev_mask.to(device=self.device, dtype=dtype)

        x_addons = self.inpainting_head(
            self.vae, image, mask_image, prev_image, prev_mask
        )

        try:
            self.unet = UnetPatched(
                self.original_unet, input_block_addons={0: x_addons}
            )
            latents = super().__call__(
                *args,
                **kwargs,
                image=image,
                mask_image=convert_mask_for_pipeline(mask_image)
            )
        finally:
            self.unet = self.original_unet
        return latents
