#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100m80:1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32000            # 64GB - CPU memory must be large enough for gpt2-xl.
#SBATCH --job-name="news_train"
#SBATCH --output=news_train.out

module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.3.1
source news/bin/activate
python3 train.py --train_path "dataset/quac-train-prompted.json" --dev_path "dataset/quac-dev-prompted.json" --model_path "gpt2" --tokenizer_path "gpt2" --output_dir "prompt_QA_orig_quac" --batch_size 16 --gradient_accumulation_steps 40 --disable_tqdm True --num_train_epochs 10 --load_best_model_at_end True --save_strategy "epoch" --evaluation_strategy "epoch" --learning_rate 3e-5 --weight_decay 0.01

deactivate
