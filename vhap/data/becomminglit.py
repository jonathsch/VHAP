import json
from copy import deepcopy
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dreifus.matrix import CameraCoordinateConvention, Pose, PoseType
from PIL import Image
import pillow_avif  # noqa
from torch.utils.data import default_collate
from torchvision.transforms.functional import to_tensor

from vhap.config.becomminglit import BecommingLitDataConfig
from vhap.util.log import get_logger

logger = get_logger(__name__)


class BecommingLitDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: BecommingLitDataConfig, img_to_tensor: bool = False, batchify_all_views: bool = False):
        super().__init__()
        self.config = cfg
        self.img_to_tensor = img_to_tensor
        self.batchify_all_views = batchify_all_views

        # Define paths and camera parameters
        self.properties = self.define_properties()
        self.camera_params = self.load_camera_params()
        self.camera_ids = sorted(list(self.camera_params.keys()))

        # Find tracking frames
        self.timestep_ids = self.load_timesteps(tracking_frames_only=True)
        self.timestep_indices = list(range(len(self.timestep_ids)))

        logger.info(f"Found {len(self.timestep_ids)} tracking frames")

        self.items: List[Dict[str, Any]] = []
        for frame_idx, frame_id in enumerate(self.timestep_ids):
            for cam_idx, cam_id in enumerate(self.camera_params.keys()):
                self.items.append(
                    {
                        "timestep_index": frame_idx,
                        "timestep_id": frame_id,
                        "camera_index": cam_idx,
                        "camera_id": cam_id,
                    }
                )

    @property
    def num_timesteps(self):
        return len(self.timestep_indices)

    @property
    def num_cameras(self):
        return len(self.camera_ids)

    @property
    def seq_folder(self):
        return self.config.root_folder / self.config.subject / self.config.sequence

    def __len__(self):
        if self.batchify_all_views:
            return self.num_timesteps
        else:
            return len(self.items)

    def __getitem__(self, idx: int):
        if self.batchify_all_views:
            return self.getitem_by_timestep(idx)
        else:
            return self.getitem_single_image(idx)

    def getitem_single_image(self, item_idx: int):
        item = deepcopy(self.items[item_idx])

        timestep_id = item["timestep_id"]
        cam_id = item["camera_id"]
        cam_prefix = self.properties["rgb"]["cam_id_prefix"]
        cam_suffix = self.properties["rgb"]["suffix"]
        folder = self.properties["rgb"]["folder"]
        rgb_path = (
            self.seq_folder / "timesteps" / f"frame_{timestep_id:05d}" / folder / f"{cam_prefix}{cam_id}.{cam_suffix}"
        )
        item["rgb"] = np.array(Image.open(rgb_path))

        cam_param = self.camera_params[item["camera_id"]]
        item["intrinsic"] = cam_param["intrinsic"].clone()
        item["extrinsic"] = cam_param["extrinsic"].clone()

        # NOTE: Skip alpha map for now

        if self.config.use_landmark:
            if self.config.landmark_source == "face-alignment":
                folder = self.properties["landmark2d/face-alignment"]["folder"]
                lmk_path = self.seq_folder / folder / f"{cam_id}.npz"
            elif self.config.landmark_source == "star":
                folder = self.properties["landmark2d/STAR"]["folder"]
                prefix = self.properties["landmark2d/STAR"]["cam_id_prefix"]
                lmk_path = self.seq_folder / folder / f"{prefix}{cam_id}.npz"
            else:
                raise NotImplementedError(f"Landmark source {self.config.landmark_source} not implemented")

            landmark_npz = np.load(lmk_path)
            timestep_idx = item["timestep_index"]
            item["lmk2d"] = landmark_npz["face_landmark_2d"][timestep_idx]  # [N, 3]
            if (item["lmk2d"][:, :2] == -1).sum() > 0:
                item["lmk2d"][:, 2:] = 0.0
            else:
                item["lmk2d"][:, 2:] = 1.0

        self.apply_scale_factor(item)
        if self.img_to_tensor:
            item["rgb"] = to_tensor(item["rgb"])

        return item

    def getitem_by_timestep(self, timestep_idx: int):
        begin = timestep_idx * self.num_cameras
        indices = range(begin, begin + self.num_cameras)
        item = default_collate([self.getitem_single_image(idx) for idx in indices])

        item["num_cameras"] = self.num_cameras
        return item

    def apply_scale_factor(self, item: Dict[str, Any]):
        h, w = item["rgb"].shape[:2]

        # properties that are defined based on image size
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= w
            item["lmk2d"][..., 1] *= h

        if "lmk2d_iris" in item:
            item["lmk2d_iris"][..., 0] *= w
            item["lmk2d_iris"][..., 1] *= h

        if "bbox_2d" in item:
            item["bbox_2d"][[0, 2]] *= w
            item["bbox_2d"][[1, 3]] *= h

    def define_properties(self):
        return {
            "rgb": {
                "folder": f"img_cc_{self.config.n_downsample_rgb}" if self.config.n_downsample_rgb else "img_cc",
                "cam_id_prefix": "cam_",
                "per_timestep": True,
                "suffix": "avif",
            },
            # "alpha_map": {
            #     "folder": "alpha_maps",
            #     "cam_id_prefix": "cam",
            #     "per_timestep": True,
            #     "suffix": "jpg",
            # },
            "landmark2d/face-alignment": {
                "folder": "landmark2d/face-alignment",
                "per_timestep": False,
                "suffix": "npz",
            },
            "landmark2d/STAR": {
                "folder": "landmark2d/STAR",
                "cam_id_prefix": "cam_",
                "per_timestep": False,
                "suffix": "npz",
            },
        }

    def get_property_path(
        self,
        name,
        index: Optional[int] = None,
        timestep_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        p = self.properties[name]
        folder = p["folder"] if "folder" in p else None
        per_timestep = p["per_timestep"]
        suffix = p["suffix"]

        path = self.seq_folder
        if folder is not None:
            path = path / folder

        if self.num_cameras > 1:
            if camera_id is None:
                assert index is not None, "index is required when camera_id is not provided."
                camera_id = self.items[index]["camera_id"]
            if "cam_id_prefix" in p:
                camera_id = p["cam_id_prefix"] + camera_id
        else:
            camera_id = ""

        if per_timestep:
            if timestep_id is None:
                assert index is not None, "index is required when timestep_id is not provided."
                timestep_id = self.items[index]["timestep_id"]
            if len(camera_id) > 0:
                path /= f"{camera_id}_{timestep_id}.{suffix}"
            else:
                path /= f"{timestep_id}.{suffix}"
        else:
            if len(camera_id) > 0:
                path /= f"{camera_id}.{suffix}"
            else:
                path = Path(str(path) + f".{suffix}")

        return path

    def load_camera_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        cam_params_path = self.config.root_folder / self.config.subject / "camera_calibration.json"
        with open(cam_params_path, mode="r") as f:
            cam_calib = json.load(f)

        # Intrinsics
        cam_params = cam_calib["cam_data"]
        fx = cam_params["fx"] / self.config.n_downsample_rgb if self.config.n_downsample_rgb else cam_params["fx"]
        fy = cam_params["fy"] / self.config.n_downsample_rgb if self.config.n_downsample_rgb else cam_params["fy"]
        cx = cam_params["cx"] / self.config.n_downsample_rgb if self.config.n_downsample_rgb else cam_params["cx"]
        cy = cam_params["cy"] / self.config.n_downsample_rgb if self.config.n_downsample_rgb else cam_params["cy"]
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Extrinsics
        world_to_cams = {cid: np.array(w2c) for cid, w2c in cam_calib["world_to_cam"].items()}

        # Convert from OpenCV to OpenGL convention
        if self.config.camera_coord_conversion == "opencv->opengl":
            world_to_cams = {
                cid: Pose(
                    w2c, pose_type=PoseType.WORLD_2_CAM, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV
                )
                .change_pose_type(PoseType.CAM_2_WORLD)
                .change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
                .change_pose_type(PoseType.WORLD_2_CAM)
                .numpy()
                for cid, w2c in world_to_cams.items()
            }

        return {
            cid: {"intrinsic": K, "extrinsic": torch.as_tensor(w2c, dtype=torch.float32)}
            for cid, w2c in world_to_cams.items()
        }

    def load_timesteps(self, tracking_frames_only: bool = True) -> List[int]:
        csv_path = self.config.root_folder / self.config.subject / self.config.sequence / "frames.csv"
        frame_df = pd.read_csv(csv_path)

        if tracking_frames_only:
            frame_ids = frame_df[frame_df["light_pattern"] == 0]["frame_id"].values
        else:
            frame_ids = frame_df["frame_id"].values

        return frame_ids.tolist()


if __name__ == "__main__":
    config = BecommingLitDataConfig(
        root_folder=Path("/cluster/pegasus/jschmidt/"),
        subject="9997",
        sequence="sequence_0003",
        use_landmark=False,
    )

    dataset = BecommingLitDataset(config, batchify_all_views=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    batch = dataset.getitem_single_image(0)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
