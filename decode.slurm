#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100m40
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64000            # 64GB - CPU memory must be large enough for gpt2-xl.
#SBATCH --job-name="news_decode"
#SBATCH --output=news_decode.out

module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.3.1
source news/bin/activate
python3 decode.py --test_path "dataset/coqa-dev.json" --output_file "answers_decode.json" --tokenizer_path "prompt_QA_coqa/tokenizer" --model_path "prompt_QA_coqa" --search_size 50 --max_answer_length 64 --decode --output_dir "prompt_QA_coqa/output" --fp16 --evaluation_strategy "epoch"

deactivate
