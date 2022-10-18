from GPT2forQA import *
from data import *
import torch
import numpy as np


class decode_QA():

	def __init__(self, data_path, model, output_path, search_size = 10):

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.data = qa_data(data_path)
		self.model = model.to(self.device)
		self.output = output_path
		self.search_size = search_size
		

	def decode(self):

		qa_dicts = self.data.qa_dicts

		for qa_list in qa_dicts:

			qa_dict = qa_list[0]
			input_ids, token_type_ids = preprocess_decode(qa_dict)  ### Need to be implemented.
			with torch.no_grad():
				input_ids = input_ids.to(self.device)
				token_type_ids = token_type_ids.to(self.device)
				start_logits,end_logits = model(input_ids=input_ids,token_type_ids=token_type_ids).logits
			start_values, start_indices = start_logits.topk(self.search_size,dim=-1)
			end_values, end_indices = end_logits.topk(self.search_size,dim=-1)
			predicted_start, predicted_end = self.search(start_values, start_indices, end_values, end_indices)
			predicted_span = id_to_span(predicted_start, predicted_end) ### Need to be implemented.
			previous_qa_dict = qa_dict

			self.write(qa_dict, predicted_span)


			for qa_dict in qa_list[1:]:
				qa_dict = self.data.add_prompt_decode(qa_dict, predicted_span, previous_qa_dict)
				input_ids, token_type_ids, prompt_positions = preprocess_decode(qa_dict)  ### Need to be implemented.
				with torch.no_grad():
					input_ids = input_ids.to(self.device)
					token_type_ids = token_type_ids.to(self.device)		
					start_logits,end_logits = model(input_ids=input_ids,token_type_ids=token_type_ids).logits
				for prompt in prompt_positions:
					start_logits[prompt] = float("-Inf")
					end_logits[prompt] = float("-Inf")
				start_values, start_indices = start_logits.topk(self.search_size,dim=-1)
				end_values, end_indices = end_logits.topk(self.search_size,dim=-1)
				predicted_start, predicted_end = self.search(start_values, start_indices, end_values, end_indices)
				predicted_span = id_to_span(predicted_start, predicted_end) ### Need to be implemented.
				previous_qa_dict = qa_dict

				self.write(qa_dict, predicted_span)


	def search(self, start_value, start_indices, end_values, end_indices):

		start_values = start_values.cpu().numpy()
		start_indices = start_indices.cpu().numpy()
		end_values = end_value.cpu().numpy()
		end_indices = end_indices.cpu().numpy()

		predicted_start = 0
		predicted_end = 0
		predicted_value = 0

		for i in range(start_values.shape[0]):
			start_value = start_values[i]
			start_indice = start_indices[i]
			for j in range(end_values.shape[0]):
				end_value = end_values[j]
				end_indice = end_indices[j]
				value = start_value + end_value
				if start_indice < end_indice and value > predicted_value:
					predicted_start = start_indice
					predicted_end = end_indice
					predicted_value = value

		return predicted_start, predicted_end





	def write(self, qa_dict, predicted_span):
		with open(self.output, 'w') as output:
			output.write(qa_dict['id'])
			output.write(qa_dict['turn_id'])
			output.write(qa_dict['question'])
			output.write(predicted_span[0], predicted_span[1])
			output.write(qa_dict['context'][predicted_span[0]:predicted_span[1]])
			output.write(qa_dict['answer_span'][0], qa_dict['answer_span'][1])
			output.write(qa_dict['answer'])
			output.write()




if __name__ == "__main__":

	import torch
	# from torch.optim import AdamW
	from transformers import AutoModel, AutoConfig

	GPT2Config = AutoConfig.from_pretrained("gpt2")

	model = GPT2forQA(GPT2Config)
	# model.resize_token_embeddings(len(tokenizer))
	model.transformer = AutoModel.from_pretrained("gpt2")

	# for param in model.transformer.parameters():
	#     param.requires_grad = False

	# input_ids = torch.empty((3,5), dtype=torch.long).random_(100)
	# attention_mask = torch.empty((3,5), dtype=torch.long).random_(1)

	# start_labels = torch.empty(3, dtype=torch.long).random_(5)
	# end_labels = torch.empty(3, dtype=torch.long).random_(5)

	# optimizer = AdamW(model.parameters(), lr=5e-5)

	# loss = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=None,start_labels=start_labels,end_labels=end_labels).loss
	# loss.backward()
	# optimizer.step()


	# for param in model.classifier.parameters():
	# 	print(param)
	# 	break
