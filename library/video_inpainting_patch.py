import torch
import torch.nn as nn
from diffusers import AutoencoderKL


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
        latent, latent_mask = self.encode_vae(vae, image, mask)

        prev_mask = (prev_mask <= 0).to(prev_image.dtype)[:, None]
        prev_latent, prev_latent_mask = self.encode_vae(vae, prev_image, prev_mask)

        prev_feat = self.head_prev(torch.cat([prev_latent, prev_latent_mask], dim=1))
        curr_feat = self.head_curr(torch.cat([latent, latent_mask], dim=1))
        return curr_feat, prev_feat

    def encode_vae(self, vae: AutoencoderKL, image, mask):
        vae_dtype = next(vae.parameters()).dtype
        with torch.no_grad():
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
