#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#


from typing import Optional, Literal, List
from dataclasses import dataclass
import tyro

from vhap.config.base import (
    StageRgbSequentialTrackingConfig,
    StageRgbGlobalTrackingConfig,
    PipelineConfig,
    DataConfig,
    LossWeightConfig,
    BaseTrackingConfig,
)
from vhap.util.log import get_logger

logger = get_logger(__name__)


@dataclass()
class GoliathDataConfig(DataConfig):
    _target: str = "vhap.data.goliath_dataset.GoliathDataset"
    calibrated: bool = True
    background_color: Optional[Literal["white", "black"]] = "black"
    landmark_source: Optional[Literal["face-alignment", "star"]] = "star"

    use_alpha_map: bool = True
    use_color_correction: bool = True
    """Whether to use color correction to harmonize the color of the input images."""
    camera_subset: Optional[List[str]] = None
    """Subset of cameras to use. If None, all cameras are used."""


@dataclass()
class GoliathLossWeightConfig(LossWeightConfig):
    landmark: Optional[float] = 3.0  # should not be lower to avoid collapse
    always_enable_jawline_landmarks: bool = (
        False  # allow disable_jawline_landmarks in StageConfig to work
    )
    reg_expr: float = 1e-2  # for best expressivness
    reg_tex_tv: Optional[float] = 1e5  # 10x of the base value


@dataclass()
class GoliathStageRgbSequentialTrackingConfig(StageRgbSequentialTrackingConfig):
    optimizable_params: tuple[str, ...] = ("pose", "joints", "expr", "dynamic_offset")

    align_texture_except: tuple[str, ...] = ("boundary",)
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Due to the limited flexibility in the lower neck region of FLAME, we relax the 
    alignment constraints for better alignment in the face region.
    """


@dataclass()
class GoliathStageRgbGlobalTrackingConfig(StageRgbGlobalTrackingConfig):
    align_texture_except: tuple[str, ...] = ("boundary",)
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Due to the limited flexibility in the lower neck region of FLAME, we relax the 
    alignment constraints for better alignment in the face region.
    """


@dataclass()
class GoliathPipelineConfig(PipelineConfig):
    rgb_sequential_tracking: GoliathStageRgbSequentialTrackingConfig
    rgb_global_tracking: GoliathStageRgbGlobalTrackingConfig


@dataclass()
class GoliathTrackingConfig(BaseTrackingConfig):
    data: GoliathDataConfig
    w: GoliathLossWeightConfig
    pipeline: GoliathPipelineConfig

    def get_occluded(self):
        occluded_table = {
            "018": ("neck_lower",),
            "218": ("neck_lower",),
            "251": ("neck_lower", "boundary"),
            "253": ("neck_lower",),
        }
        # if self.data.SID in occluded_table:
        #     logger.info(
        #         f"Automatically setting cfg.model.occluded to {occluded_table[self.data.subject]}"
        #     )
        #     self.model.occluded = occluded_table[self.data.subject]


if __name__ == "__main__":
    config = tyro.cli(GoliathTrackingConfig)
    print(tyro.to_yaml(config))
