import json
from add_prompt import qa_add_prompt

paths = ["dataset/coqa-train.json", "dataset/coqa-dev.json"]
output_paths = ["dataset/coqa-train-prompted.json", "dataset/coqa-dev-prompted.json"]

for i in range(2):
	prompts = qa_add_prompt(paths[i])
	prompts.write_prompted_data(output_paths[i])
print("done!")