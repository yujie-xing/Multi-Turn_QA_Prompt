import json
import torch
from typing import Optional
from dataclasses import dataclass, field
from numpy import array, argsort
from datasets import Dataset
from add_prompt import *


@dataclass
class DataArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	train_path: str = field(
		default=None,
		metadata={"help": "The input training data file (json)."}
	)
	dev_path: str = field(
		default=None,
		metadata={"help": "The input dev data file (json)."})
	test_path: str = field(
		default=None,
		metadata={"help": "The input test data file (json)."})
	output_file: str = field(
		default="output.txt",
		metadata={"help": "The name of the output file."})
	batch_size: Optional[int] = field(
		default=0,
		metadata={"help": "a simpler way to change both train and eval batch size."})
	tokenizer_path: str = field(
		default='gpt2',
		metadata={"help": "The Pretrained tokenizer's name or path."})
	model_path: str = field(
		default='gpt2',
		metadata={"help": "The Pretrained model's name or path."})
	max_length: Optional[int] = field(
		default=1023,
		metadata={"help": "The max length for question and context. Defaulted to model max length -1."})
	doc_stride: Optional[int] = field(
		default=128,
		metadata={"help": "The overlaping tokens for overflowing tokens. Defaulted to 128."})
	search_size: Optional[int] = field(
		default=20,
		metadata={"help": "The size of searching for start and end labels."})
	max_answer_length: Optional[int] = field(
		default=None,
		metadata={"help": "The max length for question and context. Defaulted to model max length -1."})
	evaluate: bool = field(
		default=False,
		metadata={"help": "Run evaluation with a prompted dataset"})
	decode: bool = field(
		default=False,
		metadata={"help": "Decode on a test dataset"})
	only_lm: bool = field(
		default=False,
		metadata={"help": "Do not process QA"})
	only_qa: bool = field(
		default=False,
		metadata={"help": "Do not process lm"})
	fine_tune: bool = field(
		default=False,
		metadata={"help": "Turn on when fine tuning an existing model"})



class train_data():

	def load(self, path) -> dict:
		with open(path) as file:
			data = json.load(file)

		return data

	def preprocess(self, dataset, tokenizer, max_length, doc_stride, sharp_id, space_sharp_id):

		# Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		
		eos_id = tokenizer.eos_token_id
		tokenized_examples = {"original_input_ids":list(),"input_ids":list(),"target_ids":list(),"attention_mask":list(),"token_type_ids":list(),"start_positions":list(),"end_positions":list()}
		
		for example in dataset:
			
			tokenized_example = tokenizer(
				example["question"],
				example["context"],
				max_length=max_length,
				truncation='only_second',
				return_overflowing_tokens=True,
				return_offsets_mapping=True,
				stride=doc_stride,
				padding="max_length",
			)

			tokenized_target = tokenizer(
				example["question"],
				example["human_answer"],
				max_length=max_length,
				truncation='only_second',
				return_overflowing_tokens=True,
				return_offsets_mapping=True,
				stride=doc_stride,
				padding="max_length",
			)
			

			# The offset mappings will give us a map from token to character position in the original context. This will
			# help us compute the start_positions and end_positions.
			offset_mapping = tokenized_example.pop("offset_mapping")
		
			for i, offsets in enumerate(offset_mapping):
				
				sequence_ids = tokenized_example.sequence_ids(i)
				sequence_ids_target = tokenized_target.sequence_ids(i)
				
				start_index = 0
				while sequence_ids[start_index] != 1:
					start_index += 1

				eos_index = start_index

				end_index = len(sequence_ids) - 1
				while sequence_ids[end_index] != 1:
					end_index -= 1

				target_end_index = len(sequence_ids_target) - 1
				while sequence_ids[target_end_index] != 1:
					target_end_index -= 1
				
				input_ids = tokenized_example['input_ids'][i][:eos_index] + [eos_id] + tokenized_example['input_ids'][i][eos_index:]
				target_ids = tokenized_target['input_ids'][i][:target_end_index+1] + [eos_id] + tokenized_target['input_ids'][i][target_end_index+1:]
				# input_ids = tokenized_example['input_ids'][i]
				attention_mask = tokenized_example['attention_mask'][i][:eos_index] + [1] + tokenized_example['attention_mask'][i][eos_index:]
				# attention_mask = tokenized_example['attention_mask'][i]
				sequence_ids = sequence_ids[:eos_index] + [1] + sequence_ids[eos_index:]
				# offsets = offsets[:eos_index] + [(-1,-1)] + offsets[eos_index:end_index + 1]
				offsets = [None]*(eos_index) + [(-1,-1)] + offsets[eos_index:end_index+1]
				
				
				tokenized_examples["original_input_ids"].append(input_ids)
				tokenized_examples["input_ids"].append([space_sharp_id if x==sharp_id else x for x in input_ids])
				tokenized_examples["target_ids"].append(target_ids)
				tokenized_examples["attention_mask"].append(attention_mask)
				tokenized_examples["token_type_ids"].append([id if id is not None else 1 for id in sequence_ids])

				start_index += 1
				end_index += 1

				assert eos_index == start_index-1
				
				# We will label impossible answers with the eos_index.

				answer_span = example["answer_span"]
				# If no answers are given, set the cls_index as answer.
				if answer_span[0] == -1:
		#             raise Exception("The answer span does not exist.")
					tokenized_examples["start_positions"].append(eos_index)
					tokenized_examples["end_positions"].append(eos_index)
				else:
					# Start/end character index of the answer in the text.
					start_char = answer_span[0]
					end_char = answer_span[1]

					# Detect if the answer is out of the span (in which case this feature is labeled with the last token of the question (start_index - 1)).
					if not (offsets[start_index][0] <= start_char and offsets[end_index][1] >= end_char):
		#                 raise Exception("The answer span does not fall in the context.")
						tokenized_examples["start_positions"].append(eos_index)
						tokenized_examples["end_positions"].append(eos_index)
					else:
						# Otherwise move the start_index and end_index to the two ends of the answer.
						# Note: we could go after the last offset if the answer is the last word (edge case).
						start_token, end_token = self.char_to_token(start_char,end_char,offsets,start_index,end_index)
						tokenized_examples["start_positions"].append(start_token)
						tokenized_examples["end_positions"].append(end_token)

		return Dataset.from_dict(tokenized_examples)

	def char_to_token(self, start_char, end_char, offsets, start_index, end_index):

		while start_index < len(offsets) and offsets[start_index][0] <= start_char:
			start_index += 1

		while offsets[end_index][1] >= end_char:
			end_index -= 1

		return start_index - 1, end_index + 1


class decode_data(train_data):

	def data_to_dicts_coqa(self, path):

		qa_dicts = list()

		# turn data to a dict that contain necessary informations. Need to write your own.
		# taking coqa as an example.

		with open(path) as file:
			data = json.load(file)
			data = data["data"]
			
		for article in data:

			qa_list = list()

			id = article['id']
			context = article['story']
			questions = article['questions']
			answers = article['answers']

			for num in range(len(questions)):
				question = questions[num]['input_text']
				answer = answers[num]['span_text']
				answer_start = answers[num]['span_start']
				answer_end = answers[num]['span_end']
				answer_end = answers[num]['span_end'] # context[start : end] = answer.
				human_answer = answers[num]['input_text']

				turn_id_q = questions[num]['turn_id']
				turn_id_a = answers[num]['turn_id']
				if turn_id_q != turn_id_a:
					raise Exception("Turn id does not match")
				else:
					turn_id = turn_id_q

				qa_dict = {'id': id, 'context': context, 'turn_id': turn_id, 'question': question, 'answer' : answer, 'answer_span': (answer_start,answer_end), 'human_answer': human_answer}
				qa_list.append(qa_dict)
			qa_dicts.append(qa_list)

		return qa_dicts


	def data_to_dicts_quac(self, path):

		qa_dicts = list()

		# turn data to a dict that contain necessary informations. Need to write your own.
		# taking coqa as an example.

		with open(path) as file:
			data = json.load(file)
			data = data["data"]
			
		for article in data:
			for paragraph in article['paragraphs']:

				qa_list = list()

				id = paragraph['id']
				context = paragraph['context']
				qas = paragraph['qas']

				for turn_id in range(len(qas)):
					question = qas[turn_id]['question']
					answer = qas[turn_id]['orig_answer']['text']
					if answer == 'CANNOTANSWER':
						answer_start = -1
						answer_end = -1
					else:
						answer_start = int(qas[turn_id]['orig_answer']['answer_start'])
						answer_end = answer_start + len(answer)  # context[start : end] = answer.
					human_answer = answer

					qa_dict = {'id': id, 'context': context, 'turn_id': turn_id+1, 'question': question, 'answer' : answer, 'answer_span': (answer_start,answer_end), 'human_answer': human_answer}
					qa_list.append(qa_dict)
				qa_dicts.append(qa_list)

		return qa_dicts


	def add_prompt_decode(self, qa_dict, predicted_span, previous_qa_dict):

		if predicted_span == None and previous_qa_dict == None:
			qa_dict['prompt_positions_original'] = list()
			qa_dict['prompt_positions'] = list()
			qa_dict['original_answer'] = qa_dict['answer']
			return qa_dict

		prompted_context = previous_qa_dict['context']
		prompt = "<" + str(previous_qa_dict['turn_id']) + ">"

		if predicted_span[0] != -1:
			qa_dict['context'] = prompted_context[:predicted_span[0]] + prompt + " " + prompted_context[predicted_span[0]:predicted_span[1]] + " " + prompt + prompted_context[predicted_span[1]:]
		else:
			qa_dict['context'] = prompted_context


		# Update the answer span in accordance to the prompted context.

		prompt_positions_original = previous_qa_dict['prompt_positions_original']
		prompt_positions = previous_qa_dict['prompt_positions']
		if predicted_span[0] != -1:
			predicted_span_original = self.calc_original_span_positions(prompt_positions_original, predicted_span)
			prompt_positions_original.append((predicted_span_original[0],len(prompt)+1,'start'))
			prompt_positions_original.append((predicted_span_original[1],len(prompt)+1,'end'))
			prompt_positions = self.updated_prompt_positions(prompt_positions,predicted_span,prompt)
		
		qa_dict['prompt_positions_original'] = prompt_positions_original
		qa_dict['prompt_positions'] = prompt_positions

		span = qa_dict['answer_span']
		start_diff = 0
		end_diff = 0
		for prompt_position_original in prompt_positions_original:
			if prompt_position_original[0] <= span[0]:  ## <= makes <1> <2> <3> instead of <3> <2> <1>
				start_diff += prompt_position_original[1]

			if prompt_position_original[0] < span[1]:  ## < makes <3> <2> <1> instead of <1> <2> <3>
				end_diff += prompt_position_original[1]
		
		new_span = (span[0]+start_diff, span[1]+end_diff)
		qa_dict['answer_span'] = new_span
		qa_dict['original_answer'] = qa_dict['answer']
		qa_dict['answer'] = qa_dict['context'][new_span[0]:new_span[1]]

		return qa_dict


	def calc_original_span_positions(self, prompt_positions_original, predicted_span):

		# Recalculate predicted span based on the original context.

		if predicted_span[0] == -1:
			return predicted_span

		prompt_positions_original.sort(key = lambda x: x[0])
		start_diff = 0
		end_diff = 0

		for num in range(len(prompt_positions_original)):

			lengths_inserted_front = sum(list(p[1] for p in prompt_positions_original[:num]))
			prompt_position = prompt_positions_original[num][0] + lengths_inserted_front
			prompt_length = prompt_positions_original[num][1]

			if prompt_position < predicted_span[0]:
				start_diff += prompt_length

			if prompt_position < predicted_span[1]:
				end_diff += prompt_length

		return (predicted_span[0]-start_diff, predicted_span[1]-end_diff)



	def updated_prompt_positions(self, prompt_positions, predicted_span, prompt):

		new_prompt_positions = list()

		start_prompt = (predicted_span[0],predicted_span[0]+len(prompt)+1,"start")
		end_prompt = (predicted_span[1]+len(prompt)+1,predicted_span[1]+2*(len(prompt)+1),"end")

		for p in prompt_positions:
			if p[0]>predicted_span[1]:
				p = (p[0]+2*(len(prompt)+1),p[1]+2*(len(prompt)+1),p[2])
			elif p[0]>predicted_span[0]:
				p = (p[0]+len(prompt)+1,p[1]+len(prompt)+1,p[2])
			new_prompt_positions.append(p)

		new_prompt_positions.append(start_prompt)
		new_prompt_positions.append(end_prompt)

		return new_prompt_positions


	def preprocess(self, dataset, tokenizer, max_length, doc_stride, sharp_id, space_sharp_id) -> Dataset:

		# Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		
		# eos_id = tokenizer.eos_token_id
		tokenized_examples = {"original_input_ids":list(),"input_ids":list(),"attention_mask":list(),"token_type_ids":list(),"ids":list(),"turn_ids":list(),"offset_mappings":list()}
		
		for example in dataset:
			
			tokenized_example = tokenizer(
				example["question"],
				example["context"],
				max_length=max_length,
				truncation='only_second',
				return_overflowing_tokens=True,
				return_offsets_mapping=True,
				stride=doc_stride,
				padding="max_length",
			)

			# The offset mappings will give us a map from token to character position in the original context. This will
			# help us compute the start_positions and end_positions.
			offset_mapping = tokenized_example.pop("offset_mapping")

			for i, offsets in enumerate(offset_mapping):

				tokenized_examples["ids"].append(example["id"])
				tokenized_examples["turn_ids"].append(example["turn_id"])
				
				sequence_ids = tokenized_example.sequence_ids(i)
				
				start_index = 0
				while sequence_ids[start_index] != 1:
					start_index += 1

				end_index = len(sequence_ids) - 1
				while sequence_ids[end_index] != 1:
					end_index -= 1
					
				# input_ids = tokenized_example['input_ids'][i][:eos_index] + [eos_id] + tokenized_example['input_ids'][i][eos_index:]
				input_ids = tokenized_example['input_ids'][i]
				# attention_mask = tokenized_example['attention_mask'][i][:eos_index] + [1] + tokenized_example['attention_mask'][i][eos_index:]
				attention_mask = tokenized_example['attention_mask'][i]
				# sequence_ids = sequence_ids[:eos_index] + [1] + sequence_ids[eos_index:]
				# offsets = [None]*eos_index + [(-1,-1)] + offsets[eos_index:end_index+1]
				offsets = [None]*(start_index-1) + [(-1,-1)] + offsets[start_index:end_index+1]
				
				
				tokenized_examples["original_input_ids"].append(input_ids)
				tokenized_examples["input_ids"].append([space_sharp_id if x==sharp_id else x for x in input_ids])
				tokenized_examples["attention_mask"].append(attention_mask)
				tokenized_examples["token_type_ids"].append([id if id is not None else 1 for id in sequence_ids])
				tokenized_examples["offset_mappings"].append(offsets)

		return Dataset.from_dict(tokenized_examples)

	def postprocess(self, dataset, tokenized_dataset, start_logits, end_logits, search_size = 20, max_answer_length = None):

		# Build a map example to its corresponding features.
		dataset_id_to_index = {qa_dict['id']+str(qa_dict['turn_id']): i for i, qa_dict in enumerate(dataset)}
		overflow_mapping = {i: list() for i in range(len(dataset))}
		for i, tokenized_data in enumerate(tokenized_dataset):
			overflow_mapping[dataset_id_to_index[tokenized_data["ids"]+str(tokenized_data["turn_ids"])]].append(i)

		# The dictionaries we have to fill.
		spans = list()
		scores = list()

		# Let's loop over all the examples!
		for dataset_index, qa_dict in enumerate(dataset):
			# Those are the indices of the features associated to the current example.

			tokenized_indices = overflow_mapping[dataset_index]

			predicted_spans = []
			predicted_scores = []
			prompt_chars = list()

			prompt_positions = qa_dict['prompt_positions']
			for prompt in prompt_positions:
				if prompt[2] == 'start':
					for prompt_char in range(prompt[0],prompt[1]):
						prompt_chars.append(prompt_char)
				elif prompt[2] == 'end':
					for prompt_char in range(prompt[0],prompt[1]):
						prompt_chars.append(prompt_char)

			# Looping through all the features associated to the current example.
			for tokenized_index in tokenized_indices:
				# We grab the predictions of the model for this feature.
				start_logit = start_logits[tokenized_index]
				end_logit = end_logits[tokenized_index]
				# This is what will allow us to map the positions in our logits to span of texts in the original
				# context.
				offset_mapping = tokenized_dataset[tokenized_index]["offset_mappings"]

				# Update minimum null prediction.
				# eos_index = offset_mapping.index([-1,-1])
				null_index = offset_mapping.index([-1,-1])
				assert null_index == 0  # for longformer
				null_score = start_logit[null_index] + end_logit[null_index]

				best_score = null_score
				best_answer = (-1,-1)

				# Go through all possibilities for the `n_best_size` greater start and end logits.
				start_indices = argsort(start_logit)[::-1][:search_size].tolist()
				end_indices = argsort(end_logit)[::-1][:search_size].tolist()

				print(start_indices)

				for start_index in start_indices:
					for end_index in end_indices:
						# Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
						# to part of the input_ids that are not in the context.
						if (end_index < start_index
							or start_index >= len(offset_mapping)
							or end_index >= len(offset_mapping)
							or offset_mapping[start_index] is None
							or offset_mapping[end_index] is None
						):
							continue
						# Don't consider answers with a length that is either < 0 or > max_answer_length.
						if max_answer_length is not None:
							if end_index - start_index + 1 > max_answer_length:
								continue

						start_char = offset_mapping[start_index][0]
						end_char = offset_mapping[end_index][1]

						while start_char in prompt_chars:
							start_char += 1
						while end_char in prompt_chars:
							end_char -= 1

						predicted_score = start_logit[start_index] + end_logit[end_index]

						if predicted_score > best_score:
							best_score = predicted_score
							best_answer = (start_char, end_char)
							print(start_index)
				print('done')

				predicted_spans.append(best_answer)
				predicted_scores.append(best_score)
			
			if len(predicted_spans) == 1:
				span = predicted_spans[0]
				score = predicted_scores[0]
			else:
				span = (-1,-1)
				score = max(predicted_scores)
				predicted_scores = array(predicted_scores)
				best_score_index = argsort(predicted_scores)
				for score_index in best_score_index:
					if predicted_spans[score_index] != (-1,-1):
						span = predicted_spans[score_index]
						score = predicted_scores[score_index]

			if len(dataset) == 1:
				return span, score
			else:
				spans.append(span)
				scores.append(score)

		return spans, scores


class train_data_longformer(train_data):

	def preprocess(self, dataset, tokenizer):

		# Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		
		# eos_id = tokenizer.eos_token_id
		tokenized_examples = {"input_ids":list(),"attention_mask":list(),"start_positions":list(),"end_positions":list()}

		tokenized = tokenizer(
			[example['question'] for example in dataset],
			[example["context"] for example in dataset],
			max_length=2560,
			truncation='only_second',
			return_offsets_mapping=True,
			padding="max_length",
		)


		tokenized_examples["input_ids"] = tokenized["input_ids"]
		tokenized_examples["attention_mask"] = tokenized["attention_mask"]

		for num in range(len(tokenized['offset_mapping'])):
			
			# The offset mappings will give us a map from token to character position in the original context. This will
			# help us compute the start_positions and end_positions.
			offsets = tokenized["offset_mapping"][num]
				
			sequence_ids = tokenized.sequence_ids(num)
				
			start_index = 0
			while sequence_ids[start_index] != 1:
				start_index += 1

			end_index = len(sequence_ids) - 1
			while sequence_ids[end_index] != 1:
				end_index -= 1
				
			offsets = [None]*(start_index-1) + [(-1,-1)] + offsets[start_index:end_index+1]
		
			# We will label impossible answers with the cls_label (token index = 0).

			answer_span = dataset[num]["answer_span"]
			# If no answers are given, set the cls_index as answer.
			if answer_span[0] == -1:
	#             raise Exception("The answer span does not exist.")
				tokenized_examples["start_positions"].append(0)
				tokenized_examples["end_positions"].append(0)
			else:
				# Start/end character index of the answer in the text.
				start_char = answer_span[0]
				end_char = answer_span[1]

				# Detect if the answer is out of the span (in which case this feature is labeled with the last token of the question (start_index - 1)).
				if not (offsets[start_index][0] <= start_char and offsets[end_index][1] >= end_char):
	#                 raise Exception("The answer span does not fall in the context.")
					tokenized_examples["start_positions"].append(0)
					tokenized_examples["end_positions"].append(0)
				else:
					# Otherwise move the start_index and end_index to the two ends of the answer.
					# Note: we could go after the last offset if the answer is the last word (edge case).
					start_token, end_token = self.char_to_token(start_char,end_char,offsets,start_index,end_index)
					tokenized_examples["start_positions"].append(start_token)
					tokenized_examples["end_positions"].append(end_token)

		return Dataset.from_dict(tokenized_examples)

class decode_data_longformer(decode_data):

	def preprocess(self, dataset, tokenizer, input1=1, input2=2, input3=3, input4=4) -> Dataset:

		# Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		
		# eos_id = tokenizer.eos_token_id
		tokenized_examples = {"input_ids":list(),"attention_mask":list(),"ids":list(),"turn_ids":list(),"offset_mappings":list()}

		tokenized = tokenizer(
			[example['question'] for example in dataset],
			[example['context'] for example in dataset],
			max_length=2560,
			truncation='only_second',
			return_offsets_mapping=True,
			padding="max_length",
		)

		tokenized_examples["input_ids"] = tokenized["input_ids"]
		tokenized_examples["attention_mask"] = tokenized["attention_mask"]

		for num in range(len(tokenized["offset_mapping"])):

			# The offset mappings will give us a map from token to character position in the original context. This will
			# help us compute the start_positions and end_positions.
			offsets = tokenized["offset_mapping"][num]

			tokenized_examples["ids"].append(dataset[num]["id"])
			tokenized_examples["turn_ids"].append(dataset[num]["turn_id"])
				
			sequence_ids = tokenized.sequence_ids(num)
				
			start_index = 0
			while sequence_ids[start_index] != 1:
				start_index += 1

			end_index = len(sequence_ids) - 1
			while sequence_ids[end_index] != 1:
				end_index -= 1
					
			offsets = [(-1,-1)] + [None]*(start_index-1) + offsets[start_index:end_index+1]
			
			tokenized_examples["offset_mappings"].append(offsets)

		return Dataset.from_dict(tokenized_examples)


def train_dev_test(coqa_path):
	data_processor = train_data_longformer()
	test_data_processor = decode_data_longformer()
	train_dataset = data_processor.load(coqa_path)

	# Initialize tokenizer
	# tokenizer = AutoTokenizer.from_pretrained('gpt2')
	# tokenizer.pad_token = tokenizer.eos_token
	# special_tokens_dict = {'pad_token': '<|paddingtokencustomized|>'}
	# tokenizer.add_special_tokens(special_tokens_dict)
	tokenizer = AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")
	sharp_id = tokenizer.vocab["<"]
	space_sharp_id = tokenizer.vocab["Ġ<"]

	# Tokenize dataset & prepared labels
	# tokenized_train_dataset = data_processor.preprocess(train_dataset[:], tokenizer, 1020, 128, sharp_id, space_sharp_id)
	tokenized_train_dataset = data_processor.preprocess(train_dataset[:1], tokenizer)
	tokenized_test_dataset = test_data_processor.preprocess(train_dataset[:1], tokenizer, 1020, 128, sharp_id, space_sharp_id)

	num = 0

	print(tokenized_train_dataset["input_ids"][num]==tokenized_test_dataset["input_ids"][num])

	print(train_dataset[num]['context'][train_dataset[num]['answer_span'][0]:train_dataset[num]['answer_span'][1]])

	print(len(tokenized_train_dataset['input_ids']))
	print(tokenizer.decode(tokenized_train_dataset[num]['input_ids'][tokenized_train_dataset[num]['start_positions']:tokenized_train_dataset[num]['end_positions']+1]))




def decode_test(quac_path):

	data = decode_data_longformer()

	tokenizer = AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")
	# tokenizer = AutoTokenizer.from_pretrained('gpt2')
	# tokenizer.pad_token = tokenizer.eos_token
	# special_tokens_dict = {'pad_token': '<|paddingtokencustomized|>'}
	# tokenizer.add_special_tokens(special_tokens_dict)
	sharp_id = tokenizer.vocab["<"]
	space_sharp_id = tokenizer.vocab["Ġ<"]

	qa_dicts = data.data_to_dicts_quac(quac_path)

	qa_list = qa_dicts[2409]
	print(len(qa_list))

	qa_dict = qa_list[0]
	previous_qa_dict = None
	predicted_span = None
	original_context = qa_dict['context']
	print(qa_dict)
	print()
	
	for i, qa_dict in enumerate(qa_list):
		print(qa_dict['turn_id'])
		qa_dict = data.add_prompt_decode(qa_dict, predicted_span, previous_qa_dict)

		tokenized_qa_dict = data.preprocess([qa_dict], tokenizer, 1020, 128, sharp_id, space_sharp_id)

		span = qa_dict['answer_span']  # simulating predicted span
		original_span = data.calc_original_span_positions(qa_dict['prompt_positions_original'],span)
		prompt_positions = qa_dict['prompt_positions']
		print(qa_dict['original_answer'])
		print(original_context[original_span[0]:original_span[1]])
		print(qa_dict['context'][span[0]:span[1]])
		print(span)
		# print(qa_dict['context'])
		for prompt in prompt_positions:
			print(qa_dict['context'][prompt[0]:prompt[1]])
		print()
		previous_qa_dict = qa_dict
		predicted_span = previous_qa_dict['answer_span']





if __name__ == "__main__":

	from transformers import AutoTokenizer

	coqa_train_path = 'dataset/coqa-train.json'
	coqa_train_prompted_path = 'dataset/coqa-train-prompted.json'
	coqa_dev_path = 'dataset/coqa-dev.json'
	quac_train_path = 'dataset/quac-train.json'
	quac_dev_path = 'dataset/quac-dev.json'
	quac_train_prompted_path = 'dataset/quac-train-prompted.json'

	# train_dev_test(quac_train_prompted_path)

	decode_test(quac_train_path)
