import numpy as np
from pathlib import Path
import tyro
from tqdm import tqdm
from shutil import move, rmtree

def main(folder: Path):
    frame_folders = [p for p in folder.iterdir() if p.is_dir() and p.name.startswith("frame")]
    
    for frame_folder in tqdm(frame_folders):
        img_path = frame_folder / "images" / "cam_222200037.png"
        img_tgt_path = folder / f"{int(frame_folder.name.split('_')[-1]):05d}.png"
        move(img_path, img_tgt_path)
        rmtree(frame_folder)

if __name__ == "__main__":
    tyro.cli(main)