import tyro

from vhap.config.goliath import GoliathTrackingConfig
from vhap.model.tracker import GlobalTracker


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    cfg = tyro.cli(GoliathTrackingConfig)

    tracker = GlobalTracker(cfg)
    tracker.optimize()
