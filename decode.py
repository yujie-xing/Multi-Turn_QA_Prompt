from GPT2forQA import *
from data import *
import torch


class decode_QA():

	def __init__(self, data_path, model, output_path):

		self.data = qa_data(data_path)
		self.model = model
		self.output = output_path



	def decode(self):

		qa_dicts = self.data.qa_dicts

		for qa_list in qa_dicts:

			qa_dict = qa_list[0]
			input_ids, token_type_ids = preprocess_decode(qa_dict)  ### Need to be implemented.
			with torch.no_grad():
				start_logits,end_logits = model(input_ids=input_ids,token_type_ids=token_type_ids).logits
			predicted_start = start_logits.argmax().item()
			predicted_end = end_logits.argmax().item()
			predicted_span = id_to_span(predicted_start, predicted_end) ### Need to be implemented.
			previous_qa_dict = qa_dict

			self.write(qa_dict, predicted_span)


			for qa_dict in qa_list[1:]:
				qa_dict = self.data.add_prompt_decode(qa_dict, predicted_span, previous_qa_dict)
				input_ids, token_type_ids, prompt_positions = preprocess_decode(qa_dict)  ### Need to be implemented.
				with torch.no_grad():
					start_logits,end_logits = model(input_ids=input_ids,token_type_ids=token_type_ids).logits
				for prompt in prompt_positions:
					start_logits[prompt] = float("-Inf")
					end_logits[prompt] = float("-Inf")
				predicted_start = start_logits.argmax(dim=-1).item()
				predicted_end = end_logits.argmax(dim=-1).item()
				predicted_span = id_to_span(predicted_start, predicted_end) ### Need to be implemented.
				previous_qa_dict = qa_dict

				self.write(qa_dict, predicted_span)




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
	model.transformer = AutoModel.from_pretrained("gpt2")

	# for param in model.transformer.parameters():
	#     param.requires_grad = False

	# input = torch.empty((3,5), dtype=torch.long).random_(100)
	# start_labels = torch.empty(3, dtype=torch.long).random_(5)
	# end_labels = torch.empty(3, dtype=torch.long).random_(5)

	# optimizer = AdamW(model.parameters(), lr=5e-5)

	# loss = model(input_ids=input,token_type_ids=None,start_labels=start_labels,end_labels=end_labels).loss
	# loss.backward()
	# optimizer.step()


	# for param in model.classifier.parameters():
	# 	print(param)
	# 	break
