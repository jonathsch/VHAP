import tyro

from vhap.config.becomminglit import NersembleTrackingConfig
from vhap.model.tracker import GlobalTracker


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    cfg = tyro.cli(NersembleTrackingConfig)

    tracker = GlobalTracker(cfg)
    tracker.optimize()
