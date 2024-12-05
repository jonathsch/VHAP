from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from vhap.config.base import (
    LossWeightConfig,
    PipelineConfig,
    StageRgbGlobalTrackingConfig,
    StageRgbSequentialTrackingConfig,
    BaseTrackingConfig,
    DataConfig,
)


@dataclass
class BecommingLitDataConfig:
    # The root folder for the dataset.
    root_folder: Path
    # The subject name
    subject: str
    # The sequence name
    sequence: str
    # The target class
    _target: str = "vhap.data.becomminglit.BecommingLitDataset"
    # Whether to align the camera poses to the axes
    align_cameras_to_axes: bool = False
    # How to convert the camera coordinates
    camera_coord_conversion: str = "opencv->opengl"
    calibrated: bool = True
    # Pose Type of the camera calibration
    target_extrinsic_type: Literal["w2c", "c2w"] = "w2c"
    # Which downsampled scale to use
    n_downsample_rgb: Optional[int] = 4
    # DO NOT CHANGE THIS
    scale_factor: float = 1.0
    # Which background color to use
    background_color: Optional[Literal["white", "black"]] = None
    # Whether to use the background matting
    use_alpha_map: bool = False
    # Whether to use the landmarks
    use_landmark: bool = True
    # Landmark detection backbone
    landmark_source: Optional[Literal["face-alignment", "star"]] = "star"


@dataclass()
class NersembleLossWeightConfig(LossWeightConfig):
    landmark: Optional[float] = 3.0  # should not be lower to avoid collapse
    always_enable_jawline_landmarks: bool = False  # allow disable_jawline_landmarks in StageConfig to work
    reg_expr: float = 1e-2  # for best expressivness
    reg_tex_tv: Optional[float] = 1e5  # 10x of the base value


@dataclass()
class NersembleStageRgbSequentialTrackingConfig(StageRgbSequentialTrackingConfig):
    optimizable_params: tuple[str, ...] = ("pose", "joints", "expr", "dynamic_offset")

    align_texture_except: tuple[str, ...] = ("boundary",)
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Due to the limited flexibility in the lower neck region of FLAME, we relax the 
    alignment constraints for better alignment in the face region.
    """


@dataclass()
class NersembleStageRgbGlobalTrackingConfig(StageRgbGlobalTrackingConfig):
    align_texture_except: tuple[str, ...] = ("boundary",)
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Due to the limited flexibility in the lower neck region of FLAME, we relax the 
    alignment constraints for better alignment in the face region.
    """


@dataclass()
class NersemblePipelineConfig(PipelineConfig):
    rgb_sequential_tracking: NersembleStageRgbSequentialTrackingConfig
    rgb_global_tracking: NersembleStageRgbGlobalTrackingConfig


@dataclass()
class NersembleTrackingConfig(BaseTrackingConfig):
    data: BecommingLitDataConfig
    w: NersembleLossWeightConfig
    pipeline: NersemblePipelineConfig

    def get_occluded(self):
        occluded_table = {
            "018": ("neck_lower",),
            "218": ("neck_lower",),
            "251": ("neck_lower", "boundary"),
            "253": ("neck_lower",),
        }
        if self.data.subject in occluded_table:
            logger.info(f"Automatically setting cfg.model.occluded to {occluded_table[self.data.subject]}")
            self.model.occluded = occluded_table[self.data.subject]
