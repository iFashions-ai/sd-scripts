import os
import cv2
from PIL import Image

import numpy as np
import torch
from library.train_util import BaseDataset, FineTuningSubset, ImageInfo, BucketManager
from dataclasses import dataclass
from typing import Tuple, Sequence
import json


@dataclass
class DataUnit:
    image: str
    caption: str
    mask: Tuple[str, int]  # (image, index), where mask = image == index
    prev_image: str
    prev_mask: Tuple[str, int]

    def __post_init__(self):
        self.image = str(self.image)
        self.mask = (str(self.mask[0]), self.mask[1])
        self.prev_image = str(self.prev_image)
        self.prev_mask = (str(self.prev_mask[0]), self.prev_mask[1])


class VideoInpaintingDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        self.batch_size = batch_size

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                print(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1"
                )
                continue

            if subset in self.subsets:
                print(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one"
                )
                continue

            # Load metadata
            if os.path.exists(subset.metadata_file):
                print(f"loading existing metadata: {subset.metadata_file}")
                with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                    metadata = [DataUnit(**json.loads(line)) for line in f]
            else:
                raise ValueError(f"no metadata: {subset.metadata_file}")

            if len(metadata) < 1:
                print(
                    f"ignore subset with '{subset.metadata_file}': no image entries found"
                )
                continue

            tags_list = []
            self.key2metadata = {}
            for md in metadata:
                image_key = md.image
                self.key2metadata[image_key] = md
                abs_path = None
                if os.path.exists(image_key):
                    abs_path = image_key
                else:
                    abs_path = os.path.join(subset.image_dir, image_key)
                assert abs_path is not None, f"no image: {image_key}"

                caption = md.caption or ""
                image_info = ImageInfo(
                    image_key, subset.num_repeats, caption, False, abs_path
                )
                image_info.image_size = None

                if not subset.color_aug and not subset.random_crop:
                    # if npz exists, use them
                    (
                        image_info.latents_npz,
                        image_info.latents_npz_flipped,
                    ) = self.image_key_to_npz_file(subset, image_key)

                self.register_image(image_info, subset)

            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)

        # check existence of all npz files
        use_npz_latents = False

        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            if image_info.image_size is None:
                sizes = None  # not calculated
                break
            sizes.add(image_info.image_size[0])
            sizes.add(image_info.image_size[1])
            resos.add(tuple(image_info.image_size))

        if sizes is None:
            if use_npz_latents:
                use_npz_latents = False
                print(
                    "npz files exist, but no bucket info in metadata. ignore npz files"
                )

            assert (
                resolution is not None
            ), "if metadata doesn't have bucket info, resolution is required"

            self.enable_bucket = enable_bucket
            if self.enable_bucket:
                self.min_bucket_reso = min_bucket_reso
                self.max_bucket_reso = max_bucket_reso
                self.bucket_reso_steps = bucket_reso_steps
                self.bucket_no_upscale = bucket_no_upscale
        else:
            if not enable_bucket:
                print("metadata has bucket info, enable bucketing")
            print("using bucket info in metadata")
            self.enable_bucket = True

            assert (
                not bucket_no_upscale
            ), "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used"

            # bucket情報を初期化しておく、make_bucketsで再作成しない
            self.bucket_manager = BucketManager(False, None, None, None, None)
            self.bucket_manager.set_predefined_resos(resos)

        # npz情報をきれいにしておく
        if not use_npz_latents:
            for image_info in self.image_data.values():
                image_info.latents_npz = image_info.latents_npz_flipped = None

    def image_key_to_npz_file(self, subset: FineTuningSubset, image_key):
        base_name = os.path.splitext(image_key)[0]
        npz_file_norm = base_name + ".npz"

        if os.path.exists(npz_file_norm):
            # image_key is full path
            npz_file_flip = base_name + "_flip.npz"
            if not os.path.exists(npz_file_flip):
                npz_file_flip = None
            return npz_file_norm, npz_file_flip

        # if not full path, check image_dir. if image_dir is None, return None
        if subset.image_dir is None:
            return None, None

        # image_key is relative path
        npz_file_norm = os.path.join(subset.image_dir, image_key + ".npz")
        npz_file_flip = os.path.join(subset.image_dir, image_key + "_flip.npz")

        if not os.path.exists(npz_file_norm):
            npz_file_norm = None
            npz_file_flip = None
        elif not os.path.exists(npz_file_flip):
            npz_file_flip = None

        return npz_file_norm, npz_file_flip

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        # add mask, prev_image, prev_mask
        masks = []
        prev_images = []
        prev_masks = []
        for i, image_key in enumerate(sample["image_keys"]):
            image_info = self.image_data[image_key]
            metadata: DataUnit = self.key2metadata[image_key]
            crop_top_left = sample["crop_top_lefts"][i]
            target_size = sample["images"][i].shape[2:0:-1]
            flipped = sample["flippeds"][i]

            # mask
            mask = self.read_mask(*metadata.mask)
            mask_crop = self.resize_and_crop(
                mask, flipped, image_info.resized_size, crop_top_left, target_size
            )
            masks.append(torch.from_numpy(mask_crop))

            # prev_image
            prev_image = np.array(Image.open(metadata.prev_image).convert("RGB"))
            prev_image_crop = self.resize_and_crop(
                prev_image, flipped, image_info.resized_size, crop_top_left, target_size
            )
            prev_image_crop = self.image_transforms(prev_image_crop)
            prev_images.append(prev_image_crop)

            # prev_mask
            prev_mask = self.read_mask(*metadata.prev_mask)
            prev_mask_crop = self.resize_and_crop(
                prev_mask, flipped, image_info.resized_size, crop_top_left, target_size
            )
            prev_masks.append(torch.from_numpy(prev_mask_crop))

        sample["masks"] = torch.stack(masks)
        sample["prev_images"] = (
            torch.stack(prev_images).to(memory_format=torch.contiguous_format).float()
        )
        sample["prev_masks"] = torch.stack(prev_masks)
        return sample

    @staticmethod
    def resize_and_crop(
        image: np.ndarray, flipped: bool, resized_size, crop_top_left, target_size
    ):
        original_size = image.shape[1::-1]
        if original_size != resized_size:
            image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
        if flipped:
            image = image[:, ::-1]

        image = image[
            crop_top_left[1] : crop_top_left[1] + target_size[1],
            crop_top_left[0] : crop_top_left[0] + target_size[0],
        ]
        return np.ascontiguousarray(image)

    @staticmethod
    def read_mask(filename, index):
        mask = np.array(Image.open(filename).convert("L"))
        mask = (mask == index).astype(np.uint8) * 255

        # TODO: expand mask
        return mask
