import json
from add_prompt import qa_add_prompt

paths = ["dataset/quac-train.json", "dataset/quac-dev.json"]
output_paths = ["dataset/quac-train-prompted.json", "dataset/quac-dev-prompted.json"]

for i in range(2):
	prompts = qa_add_prompt(paths[i])
	prompts.write_prompted_data(output_paths[i])
print("done!")