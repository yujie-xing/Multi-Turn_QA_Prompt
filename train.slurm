#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="news_train"
#SBATCH --output=news_train.out

module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.3.1
source news/bin/activate
python3 train.py --train_path "dataset/coqa-train-prompted.json" --dev_path "dataset/coqa-dev-prompted.json" --output_dir "prompt_QA_coqa" --num_train_epochs 10 --load_best_model_at_end True --save_strategy "epoch" --evaluation_strategy "epoch"

deactivate
