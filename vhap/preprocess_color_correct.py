from pathlib import Path
import json
from typing import List, Optional

from PIL import Image
import pillow_avif  # noqa
import imageio.v3 as iio
import numpy as np
import torch
import tyro

from vhap.util.color import color_correct_srgb
from vhap.util.image import scale_img_hwc


def main(input_folder: Path, ccm_path: Path, downsample_scales: List[int] = []):
    frame_folders = sorted([p for p in input_folder.glob("frame_*") if p.is_dir()])
    print(f"Found {len(frame_folders)} frames")

    with open(ccm_path, "r") as f:
        ccm_dict = json.load(f)

    ccm_dict = {cid: torch.tensor(ccm["ccm"]).cuda() for cid, ccm in ccm_dict.items()}
    print(f"Loaded {len(ccm_dict)} CCMs")

    for frame_dir in frame_folders:
        # Make output folder
        out_folder = frame_dir / "img_cc"
        out_folder.mkdir(exist_ok=True)

        # Make downsampled folder
        for ds_scale in downsample_scales:
            out_folder_ds = frame_dir / f"img_cc_{ds_scale}"
            out_folder_ds.mkdir(exist_ok=True)

        for cid in ccm_dict.keys():
            # Load image
            img_path = frame_dir / "images" / f"cam_{cid}.png"
            img = torch.as_tensor(iio.imread(img_path), dtype=torch.float32).cuda()[..., :3] / 255.0  # [H, W, 3]

            # Color correct
            img_cc = color_correct_srgb(img, ccm_dict[cid])

            # Save
            iio.imwrite(
                out_folder / f"cam_{cid}.avif",
                (img_cc * 255).cpu().numpy().astype(np.uint8),
            )

            # Optionally downsample and save
            for ds_scale in downsample_scales:
                out_path_ds = frame_dir / f"img_cc_{ds_scale}"
                out_size = (
                    img_cc.shape[0] // ds_scale,
                    img_cc.shape[1] // ds_scale,
                )  # H, W
                img_cc_ds = scale_img_hwc(img_cc, size=out_size)
                iio.imwrite(
                    out_path_ds / f"cam_{cid}.avif",
                    (img_cc_ds * 255).cpu().numpy().astype(np.uint8),
                )


if __name__ == "__main__":
    tyro.cli(main)
