#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64000            # 64GB - CPU memory must be large enough for gpt2-xl.
#SBATCH --job-name="news_eval_gen_prompt"
#SBATCH --output=news_eval_gen_prompt.out

module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.3.1
source news/bin/activate
python3 decode.py --evaluate --label_names=["target_ids"] --test_path "dataset/coqa-dev-prompted.json" --output_file "answers_eval_gen_prompt" --tokenizer_path "prompt_gen_prompt/tokenizer" --model_path "prompt_gen_prompt" --disable_tqdm True --search_size 50 --max_answer_length 64 --output_dir "output" --evaluation_strategy "epoch"

deactivate
