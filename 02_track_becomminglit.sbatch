#!/bin/bash 
 
#SBATCH --job-name=vhap_tracking
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j_%N.log 

#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:1

#SBATCH -p submit
 
source /rhome/jschmidt/.bashrc
eval "$(conda shell.bash hook)"
source activate VHAP

nvidia-smi

DATA_ROOT="/cluster/pegasus/jschmidt/"
SUBJECT="9997"
SEQUENCE="FREE"

OUTPUT_FOLDER="${DATA_ROOT}/${SUBJECT}/${SEQUENCE}/flame_tracking"

srun python vhap/track_becomminglit.py \
--data.root_folder "${DATA_ROOT}" \
--data.subject ${SUBJECT} \
--data.sequence ${SEQUENCE} \
--exp.output_folder ${OUTPUT_FOLDER} \
--data.n_downsample_rgb 4
