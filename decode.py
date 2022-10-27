from os import path
import torch
from transformers import AutoTokenizer, Trainer
from GPT2forQA import *
from data import DataArguments, decode_data



class generate_QA():

	def __init__(self, args, dataargs):

		self.args = args
		self.dataargs = dataargs

		# Initialize tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(dataargs.tokenizer_path)
		# Initialize model
		model = GPT2forQA.from_pretrained(dataargs.model_path)
		self.predictor = QATrainer(model, self.args)

		with open(path.join(self.args.output_dir,'output.txt'), 'w') as output:
			output.write("")

		

	def decode(self):

		decode_data_processor = decode_data()

		qa_dicts = decode_data_processor.data_to_dicts(self.dataargs.test_path)

		for qa_list in qa_dicts:

			predicted_span = None
			previous_qa_dict = None

			original_context = qa_list[0]['context']

			for i, qa_dict in enumerate(qa_list):
				qa_dict = decode_data_processor.add_prompt_decode(qa_dict, predicted_span, previous_qa_dict)
				tokenized_qa_dict = decode_data_processor.preprocess([qa_dict], self.tokenizer, self.dataargs.max_length, self.dataargs.doc_stride)
				start_logits,end_logits = self.predictor.predict(tokenized_qa_dict).predictions
				predicted_span, predicted_score = decode_data_processor.postprocess([qa_dict], tokenized_qa_dict, start_logits, end_logits, self.dataargs.search_size, self.dataargs.max_answer_length)
				predicted_span_original = decode_data_processor.calc_original_span_positions(qa_dict['prompt_positions_original'],predicted_span)
				previous_qa_dict = qa_dict

				self.write(qa_dict, predicted_span, predicted_score, predicted_span_original, original_context)




	def evaluate(self):   ## For evaluation of prompted test dataset.

		eval_data_processor = decode_data()

		test_dataset = eval_data_processor.load(self.dataargs.test_path)

		# Tokenize dataset & prepared labels
		tokenized_test_dataset = eval_data_processor.preprocess(test_dataset, self.tokenizer, self.dataargs.max_length, self.dataargs.doc_stride)

		start_logits, end_logits = self.predictor.predict(tokenized_test_dataset).predictions
		predicted_spans, predicted_scores = eval_data_processor.postprocess(test_dataset, tokenized_test_dataset, start_logits, end_logits, self.dataargs.search_size, self.dataargs.max_answer_length)

		predicted_spans_original = list()
		for i, predicted_span in enumerate(predicted_spans):
			predicted_span_original = eval_data_processor.calc_original_span_positions(test_dataset[i]['prompt_positions_original'],predicted_span)
			predicted_spans_original.append(predicted_span_original)
				
		for i, qa_dict in enumerate(test_dataset):
			self.write(qa_dict, predicted_spans[i], predicted_scores[i], predicted_spans_original[i], qa_dict['original_context'])



	def write(self, qa_dict, predicted_span, predicted_score, predicted_span_original, original_context):
		with open(path.join(self.args.output_dir,'output.txt'), 'a') as output:
			output.write(qa_dict['id'])
			output.write("\n")
			output.write(str(qa_dict['turn_id']))
			output.write("\n")
			output.write(qa_dict['question'])
			output.write("\n")
			output.write(str(predicted_span[0]))
			output.write(",")
			output.write(str(predicted_span[1]))
			output.write("\n")
			output.write(qa_dict['context'][predicted_span[0]:predicted_span[1]])
			output.write("\n")
			output.write(str(predicted_span_original[0]))
			output.write(",")
			output.write(str(predicted_span_original[1]))
			output.write("\n")
			output.write(original_context[predicted_span_original[0]:predicted_span_original[1]])
			output.write("\n")
			output.write(str(qa_dict['answer_span'][0]))
			output.write(",")
			output.write(str(qa_dict['answer_span'][1]))
			output.write("\n")
			output.write(qa_dict['answer'])
			output.write("\n")
			output.write(qa_dict['original_answer'])
			output.write("\n")
			output.write("=============================\n")




if __name__ == "__main__":

	from transformers import HfArgumentParser, TrainingArguments

	#===============================
	# Parser -> args
	try:
		parser = HfArgumentParser((DataArguments,TrainingArguments))
		dataargs, args = parser.parse_args_into_dataclasses()
		if dataargs.batch_size != 0:
			args.per_device_eval_batch_size = dataargs.batch_size
		try:
			mkdir(args.output_dir)
		except:
			pass
		print(args)
		print(dataargs)
	except:  ## Only for test
		print("\n======================\n")
		args = TrainingArguments(output_dir="test_output")
		try:
			mkdir("test_output")
		except:
			pass
		dataargs = DataArguments(test_path='dataset/coqa-dev-prompted.json', tokenizer_path='test/tokenizer', model_path='test')
		print("Test Mode")
		print(args)
		print(dataargs)


	#============================
	QA_model = generate_QA(args, dataargs)
	QA_model.decode()
