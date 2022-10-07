from GPT2forQA import *
from transformers import AutoModel, AutoConfig

GPT2Config = AutoConfig.from_pretrained("gpt2")

model = GPT2forQA(GPT2Config)
model.transformer = AutoModel.from_pretrained("gpt2")

for param in model.transformer.parameters():
    param.requires_grad = False




if __name__ == "__main__":

	import torch
	from torch.optim import AdamW

	input = torch.empty((3,5), dtype=torch.long).random_(100)
	start_labels = torch.empty(3, dtype=torch.long).random_(5)
	end_labels = torch.empty(3, dtype=torch.long).random_(5)

	optimizer = AdamW(model.parameters(), lr=5e-5)

	for param in model.transformer.parameters():
		print(param)
		break
	for param in model.classifier.parameters():
		print(param)
		break

	loss = model(input_ids=input,token_type_ids=None,position_ids=None,start_labels=start_labels,end_labels=end_labels).loss
	loss.backward()
	optimizer.step()


	for param in model.transformer.parameters():
		print(param)
		break
	for param in model.classifier.parameters():
		print(param)
		break
