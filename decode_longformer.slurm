#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64000
#SBATCH --job-name="news_decode_longformer"
#SBATCH --output=news_decode_longformer.out

module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.3.1
source news/bin/activate
python3 decode.py --test_path "dataset/quac-dev.json" --output_file "answers_decode.json" --tokenizer_path "prompt_QA_longformer/tokenizer" --model_path "prompt_QA_longformer" --search_size 50 --max_answer_length 20 --decode --output_dir "output" --evaluation_strategy "epoch"

deactivate
