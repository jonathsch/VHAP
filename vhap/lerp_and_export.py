from pathlib import Path
import yaml

import numpy as np
import trimesh
import torch
import tyro
from tqdm import tqdm
from pytorch3d.transforms import so3_exp_map

from vhap.util.log import get_logger
from vhap.config.becomminglit import BecommingLitDataConfig, NersembleTrackingConfig
from vhap.data.becomminglit import BecommingLitDataset
from vhap.model.flame import FlameHead

logger = get_logger(__name__, root=True)


def main(config_path: Path):
    device = torch.device("cuda:0")

    with open(config_path, mode="r") as f:
        cfg: NersembleTrackingConfig = yaml.unsafe_load(f)

    dataset = BecommingLitDataset(cfg.data)
    timestep_ids = dataset.timestep_ids
    timestep_indices = dataset.timestep_indices

    output_folder = cfg.exp.output_folder / "vertices"
    output_folder.mkdir(exist_ok=True, parents=True)

    flame_model = FlameHead(
        cfg.model.n_shape,
        cfg.model.n_expr,
        add_teeth=cfg.model.add_teeth,
        remove_lip_inside=cfg.model.remove_lip_inside,
        face_clusters=cfg.model.tex_clusters,
    ).to(device)

    # Load the FLAME paramters
    param_folder = sorted(cfg.exp.output_folder.glob("20*"))[-1]  # Get the latest folder
    flame_params = np.load(param_folder / "tracked_flame_params_30.npz")

    # Shape and static offset do not change over time
    shape = torch.as_tensor(flame_params["shape"][None, ...], dtype=torch.float32, device=device)
    static_offset = torch.as_tensor(flame_params["static_offset"], dtype=torch.float32, device=device)

    vert_list = []
    head_pose_list = []

    id_rot = torch.zeros(1, 3, device=device, dtype=torch.float32)
    id_transl = torch.zeros(1, 3, device=device, dtype=torch.float32)

    for idx in tqdm(range(len(timestep_ids) - 1)):
        # Tracking frame
        tid = timestep_ids[idx]
        params = {
            k: torch.as_tensor(v[idx : idx + 1], dtype=torch.float32, device=device)
            for k, v in flame_params.items()
            if isinstance(v, np.ndarray) and k not in {"shape", "static_offset"} and v.ndim > 1
        }

        head_pose = torch.eye(4)
        head_pose[:3, :3] = so3_exp_map(params["rotation"]).squeeze(0) # [3, 3]
        head_pose[:3, 3] = params["translation"].view(3)
        head_pose_list.append(head_pose.clone())

        verts, verts_cano, lmks = flame_model(
            shape,
            params["expr"],
            # params["rotation"],
            id_rot,
            params["neck_pose"],
            params["jaw_pose"],
            params["eyes_pose"],
            # params["translation"],
            id_transl,
            static_offset=static_offset,
            return_verts_cano=True,
        )

        vert_list.append(verts[0].detach().cpu().numpy())
        trimesh.PointCloud(verts[0].detach().cpu().numpy()).export(output_folder / f"frame_{tid}.ply")

        # Interpolate first frame
        tid_1 = tid + 1
        params = {
            k: torch.lerp(
                torch.as_tensor(v[idx : idx + 1], dtype=torch.float32, device=device),
                torch.as_tensor(v[idx + 1 : idx + 2], dtype=torch.float32, device=device),
                1 / 3,
            )
            for k, v in flame_params.items()
            if isinstance(v, np.ndarray) and k not in {"shape", "static_offset"} and v.ndim > 1
        }

        head_pose = torch.eye(4)
        head_pose[:3, :3] = so3_exp_map(params["rotation"]).squeeze(0) # [3, 3]
        head_pose[:3, 3] = params["translation"].view(3)
        head_pose_list.append(head_pose.clone())

        # print(
        #     torch.as_tensor(flame_params["static_offset"][idx : idx + 1], dtype=torch.float32, device=device)
        #     .clone()
        #     .shape
        # )
        # print(flame_params["static_offset"].shape)
        # print(torch.as_tensor(flame_params["static_offset"][idx + 1], dtype=torch.float32, device=device).clone().shape)
        # quit()

        verts, verts_cano, lmks = flame_model(
            shape,
            params["expr"],
            params["rotation"],
            params["neck_pose"],
            params["jaw_pose"],
            params["eyes_pose"],
            params["translation"],
            static_offset=static_offset,
            return_verts_cano=True,
        )

        vert_list.append(verts[0].detach().cpu().numpy())
        trimesh.PointCloud(verts[0].detach().cpu().numpy()).export(output_folder / f"frame_{tid_1}.ply")

        # Interpolate second frame
        tid_2 = tid + 2
        params = {
            k: torch.lerp(
                torch.as_tensor(v[idx : idx + 1], dtype=torch.float32, device=device).clone(),
                torch.as_tensor(v[idx + 1 : idx + 2], dtype=torch.float32, device=device).clone(),
                2 / 3,
            )
            for k, v in flame_params.items()
            if isinstance(v, np.ndarray) and k not in {"shape", "static_offset"} and v.ndim > 1
        }

        head_pose = torch.eye(4)
        head_pose[:3, :3] = so3_exp_map(params["rotation"]).squeeze(0) # [3, 3]
        head_pose[:3, 3] = params["translation"].view(3)
        head_pose_list.append(head_pose.clone())

        verts, verts_cano, lmks = flame_model(
            shape,
            params["expr"],
            params["rotation"],
            params["neck_pose"],
            params["jaw_pose"],
            params["eyes_pose"],
            params["translation"],
            static_offset=static_offset,
            return_verts_cano=True,
        )

        vert_list.append(verts[0].detach().cpu().numpy())
        trimesh.PointCloud(verts[0].detach().cpu().numpy()).export(output_folder / f"frame_{tid_2}.ply")

    # Last frame (must be tracking frame)
    tid = timestep_ids[-1]
    params = {
        k: torch.as_tensor(v[-1:], dtype=torch.float32, device=device)
        for k, v in flame_params.items()
        if isinstance(v, np.ndarray) and k not in {"shape", "static_offset"} and v.ndim > 1
    }

    head_pose = torch.eye(4)
    head_pose[:3, :3] = so3_exp_map(params["rotation"]).squeeze(0) # [3, 3]
    head_pose[:3, 3] = params["translation"].view(3)
    head_pose_list.append(head_pose.clone())

    verts, verts_cano, lmks = flame_model(
        shape,
        params["expr"],
        params["rotation"],
        params["neck_pose"],
        params["jaw_pose"],
        params["eyes_pose"],
        params["translation"],
        static_offset=static_offset,
        return_verts_cano=True,
    )

    vert_list.append(verts[0].detach().cpu().numpy())
    trimesh.PointCloud(verts[0].detach().cpu().numpy()).export(output_folder / f"frame_{tid}.ply")

    vert_stack = np.stack(vert_list) # [N, V, 3]
    vert_mean = np.mean(vert_stack, axis=0) # [V, 3]
    vert_var = np.var(vert_stack) # [1]

    np.save(output_folder / "vertices_mean.npy", vert_mean)
    np.savetxt(output_folder / "vertices_var.txt", vert_var)


if __name__ == "__main__":
    tyro.cli(main)
