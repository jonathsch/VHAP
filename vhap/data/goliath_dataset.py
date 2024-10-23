import json
import numpy as np
import torch
from typing import Optional
from pathlib import Path
from vhap.data.video_dataset import VideoDataset
from vhap.config.goliath import GoliathDataConfig
from vhap.util import camera
from vhap.util.log import get_logger
import trimesh
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
import zipfile

SCALE_FACTOR = 0.001


logger = get_logger(__name__)


class GoliathDataset(VideoDataset):
    def __init__(
        self,
        cfg: GoliathDataConfig,
        img_to_tensor: bool = False,
        batchify_all_views: bool = False,
    ):
        self.cfg = cfg

        super().__init__(
            cfg=cfg,
            img_to_tensor=img_to_tensor,
            batchify_all_views=batchify_all_views,
        )

        # Find tracking frames (3D keypoints are available)
        self.timestep_ids = sorted([
            p.stem
            for p in (
                self.cfg.root_folder / self.cfg.sequence / "keypoints_3d"
            ).iterdir()
            if p.suffix == ".json"
        ])

        # Limit to first 100 frames for debugging
        self.timestep_ids = self.timestep_ids[:100]
        print(*self.timestep_ids[:10], sep="\n")

        self.timestep_indices = list(range(len(self.timestep_ids)))

        self.items = []
        for fi, timestep_index in enumerate(self.timestep_indices):
            for ci, camera_id in enumerate(self.camera_ids):
                self.items.append(
                    {
                        "timestep_index": fi,  # new index after filtering
                        "timestep_index_original": timestep_index,  # original index
                        "timestep_id": self.timestep_ids[timestep_index],
                        "camera_index": ci,
                        "camera_id": camera_id,
                    }
                )

    def match_sequences(self):
        logger.info(f"Sequence: {self.cfg.sequence}")
        return [self.cfg.root_folder / self.cfg.sequence]

    def define_properties(self):
        self.properties = {
            "rgb": {
                "folder": "image",
                "per_timestep": True,
                "suffix": "avif",
            },
            "landmark2d/face-alignment": {
                "folder": "landmark2d/face-alignment",
                "per_timestep": False,
                "suffix": "npz",
            },
            "landmark2d/STAR": {
                "folder": "landmark2d/STAR",
                "per_timestep": False,
                "suffix": "npz",
            },
        }

    def load_camera_params(self):
        load_path = self.cfg.root_folder / self.cfg.sequence / "camera_calibration.json"
        assert load_path.exists(), f"{load_path} does not exist"

        with open(load_path, "r") as f:
            goliath_calib = json.load(f)["KRT"]
        camera_params = {str(c["cameraId"]): c for c in goliath_calib}

        # Save camera ids
        self.camera_ids = list(camera_params.keys())

        # Read camera intrinsics
        Ks = {
            cam_id: np.array(krt["K"]).T
            for cam_id, krt in camera_params.items()
        }

        # Read camera poses
        Ts = {
            cam_id: np.array(krt["T"]).T
            for cam_id, krt in camera_params.items()
        }

        # Convert camera poses to head-related poses wrt to first frame
        head_pose_path = (
            self.cfg.root_folder / self.cfg.sequence / "head_pose" / "head_pose.zip"
        )
        with zipfile.ZipFile(head_pose_path, "r") as zip_file:
            first_head_pose = sorted([p for p in zip_file.namelist()])[0]
            with zip_file.open(first_head_pose) as f:
                head_pose = np.eye(4)
                head_pose[:3, :4] = np.loadtxt(f)

        Ts = {cam_id: T @ head_pose for cam_id, T in Ts.items()}
        Ts = np.stack([T for T in Ts.values()])  # (N, 4, 4)

        # Convert from mm to meters
        Ts[:, :3, 3] *= SCALE_FACTOR

        # Ks /= 2  # Fix wrong scaling in the dataset
        if self.cfg.n_downsample_rgb:
            Ks /= self.cfg.n_downsample_rgb

        # Convert to OpenGL convention
        extrinsic = [
            Pose(
                extrinsics,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                pose_type=PoseType.WORLD_2_CAM,
            )
            .change_pose_type(PoseType.CAM_2_WORLD)
            .change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
            .change_pose_type(PoseType.WORLD_2_CAM)
            .numpy()
            for extrinsics in Ts
        ]
        extrinsic = np.stack(extrinsic, axis=0)
        extrinsic = torch.as_tensor(extrinsic, dtype=torch.float32)

        # orientation = R# .transpose(-1, -2)  # (N, 3, 3)
        # location = R.transpose(-1, -2) @ -t[..., None]  # (N, 3, 1)
        # location *= 0.001  # convert to meters

        # trimesh.PointCloud(location[..., 0].detach().cpu().numpy() / 1000).export("campos_goliath.ply")

        # # adjust how cameras distribute in the space with a global rotation
        # if self.cfg.align_cameras_to_axes:
        #     orientation, location = camera.align_cameras_to_axes(
        #         orientation, location, target_convention="opengl"
        #     )

        # # modify the local orientation of cameras to fit in different camera conventions
        # if self.cfg.camera_coord_conversion is not None:
        #     orientation = camera.change_camera_coord_convention(
        #         self.cfg.camera_coord_conversion, orientation
        #     )

        # c2w = torch.cat(
        #     [orientation, location], dim=-1
        # )  # camera-to-world transformation

        # if self.cfg.target_extrinsic_type == "w2c":
        #     R = orientation.transpose(-1, -2)
        #     T = orientation.transpose(-1, -2) @ -location
        #     w2c = torch.cat([R, T], dim=-1)  # world-to-camera transformation
        #     extrinsic = w2c
        # elif self.cfg.target_extrinsic_type == "c2w":
        #     extrinsic = c2w
        # else:
        #     raise NotImplementedError(
        #         f"Unknown extrinsic type: {self.cfg.target_extrinsic_type}"
        #     )
        # extrinsic = w2c

        # Scale extrinsics to better match the flame mesh
        # extrinsic[:, :3, 3] *= 0.001

        self.camera_params = {}
        for i, camera_id in enumerate(camera_params.keys()):
            self.camera_params[camera_id] = {
                "intrinsic": torch.from_numpy(Ks[camera_id]),
                "extrinsic": extrinsic[i],
            }

    def filter_division(self, division):
        CAM_SUBSET = {
            "401643",
            "401645" "401646",
            "401650",
            "401652",
            "401653",
            "401655",
            "401659",
            "401949",
            "401951",
            "401958",
        }
        # CAM_SUBSET = {
        #     "401645"
        # }
        self.camera_ids = [cam_id for cam_id in self.camera_ids if cam_id in CAM_SUBSET]
        pass

    def apply_transforms(self, item):
        # NOTE: Skip for now
        super().apply_transforms(item)
        return item

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

        path = self.sequence_path
        if folder is not None:
            path = path / folder

        if self.num_cameras > 1:
            if camera_id is None:
                assert (
                    index is not None
                ), "index is required when camera_id is not provided."
                camera_id = self.items[index]["camera_id"]
            if "cam_id_prefix" in p:
                camera_id = p["cam_id_prefix"] + camera_id
        else:
            camera_id = ""

        if per_timestep:
            if timestep_id is None:
                assert (
                    index is not None
                ), "index is required when timestep_id is not provided."
                timestep_id = self.items[index]["timestep_id"]
            if len(camera_id) > 0:
                path = path / f"cam{camera_id}" / f"{timestep_id}.{suffix}"
            else:
                path /= f"{timestep_id}.{suffix}"
        else:
            if len(camera_id) > 0:
                path /= f"{camera_id}.{suffix}"
            else:
                path = Path(str(path) + f".{suffix}")

        return path

    def get_property_path_list(self, name):
        paths = []
        for i in range(len(self.items)):
            img_path = self.get_property_path(name, i)
            paths.append(img_path)
        return paths


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    import tyro
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from vhap.config.base import import_module

    cfg = tyro.cli(
        GoliathDataConfig,
        default=GoliathDataConfig(
            root_folder=Path("/mnt/cluster/pegasus/jschmidt/goliath"),
            sequence="m--20230306--0707--AXE977--pilot--ProjectGoliath--Head",
        ),
    )
    cfg.use_landmark = False
    dataset = import_module(cfg._target)(
        cfg=cfg,
        img_to_tensor=False,
        batchify_all_views=True,
    )

    print(dataset.num_timesteps)
    print(dataset.num_cameras)
    print(len(dataset))

    # print(dataset.camera_params["222200037"])
    # intrinsics = dataset.camera_params["222200037"]["intrinsic"]
    # extrinsics = dataset.camera_params["222200037"]["extrinsic"]
    # np.save("intrinsics.npy", intrinsics.detach().cpu().numpy())
    # np.save("extrinsics.npy", extrinsics.detach().cpu().numpy())
    # quit()

    # sample = dataset[0]
    # print(sample.keys())
    # print(sample["rgb"].shape)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    batch = next(iter(dataloader))
    print(batch["camera_id"][0])
    print(batch["intrinsic"][0])
    print(batch["extrinsic"][0])
    # print(batch["intrinsic"][0])
    # print(batch["extrinsic"][0])
