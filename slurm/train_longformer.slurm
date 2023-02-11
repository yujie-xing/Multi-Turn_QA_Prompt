#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100m80:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64000
#SBATCH --job-name="news_train_longformer"
#SBATCH --output=news_train_longformer.out

module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.3.1
source news/bin/activate
python3 train_longformer.py --only_qa --train_path "dataset/quac-train-prompted.json" --dev_path "dataset/quac-dev-prompted.json" --model_path "mrm8488/longformer-base-4096-finetuned-squadv2" --tokenizer_path "mrm8488/longformer-base-4096-finetuned-squadv2" --output_dir "prompt_QA_longformer" --batch_size 8 --gradient_accumulation_steps 80 --disable_tqdm True --num_train_epochs 10 --load_best_model_at_end True --save_strategy "epoch" --evaluation_strategy "epoch" --learning_rate 3e-5 --weight_decay 0.01

deactivate
