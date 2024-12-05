from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class BecommingLitDataConfig:
    # The root folder for the dataset.
    root_folder: Path
    # The subject name
    subject: str
    # The sequence name
    sequence: str
    # Whether to align the camera poses to the axes
    align_cameras_to_axes: bool = False
    # How to convert the camera coordinates
    camera_coord_conversion: str = "opencv->opengl"
    # Pose Type of the camera calibration
    target_extrinsic_type: Literal["w2c", "c2w"] = "w2c"
    # Which downsampled scale to use
    n_downsample_rgb: Optional[int] = 4
    # Which background color to use
    background_color: Optional[Literal["white", "black"]] = None
    # Whether to use the background matting
    use_alpha_map: bool = False
    # Whether to use the landmarks
    use_landmark: bool = True
    # Landmark detection backbone
    landmark_source: Optional[Literal["face-alignment", "star"]] = "star"
