import json
from add_prompt import *


def load_train_dev_data(path):
	with open(path) as file:
		data = json.load(file)

	return data


def prepare_train_features(dataset, tokenizer, max_length, doc_stride):

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    
    eos_id = tokenizer.eos_token_id
    tokenized_examples = {"input_ids":list(),"attention_mask":list(),"token_type_ids":list(),"start_labels":list(),"end_labels":list()}
    
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
            
            sequence_ids = tokenized_example.sequence_ids(i)
            
            # Add en eos token in the middle.
            
            eos_index = 0
            while sequence_ids[eos_index] != 1:
                eos_index += 1
                
            input_ids = tokenized_example['input_ids'][i][:eos_index] + [eos_id] + tokenized_example['input_ids'][i][eos_index:]
            attention_mask = tokenized_example['attention_mask'][i][:eos_index] + [1] + tokenized_example['attention_mask'][i][eos_index:]
            sequence_ids = sequence_ids[:eos_index] + [1] + sequence_ids[eos_index:]
            offsets = offsets[:eos_index] + [(-1,-1)] + offsets[eos_index:]
            
            
            tokenized_examples["input_ids"].append(input_ids)
            tokenized_examples["attention_mask"].append(attention_mask)
            tokenized_examples["token_type_ids"].append([id if id is not None else 1 for id in sequence_ids])
            
            # We will label impossible answers with the index of the eos token.

            assert eos_index == input_ids.index(eos_id)

            answer_span = example["answer_span"]
            # If no answers are given, set the cls_index as answer.
            if answer_span[0] == -1:
    #             raise Exception("The answer span does not exist.")
                tokenized_examples["start_labels"].append(eos_index)
                tokenized_examples["end_labels"].append(eos_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answer_span[0]
                end_char = answer_span[1]

                # Start token index of the current span in the text.
                token_start_index = eos_index + 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the eos index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
    #                 raise Exception("The answer span does not fall in the context.")
                    tokenized_examples["start_labels"].append(eos_index)
                    tokenized_examples["end_labels"].append(eos_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_labels"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_labels"].append(token_end_index + 1)

    return tokenized_examples



class qa_data(qa_add_prompt):

	def __init__(self, path):

		self.qa_dicts = self.data_to_dicts(path)


	# def data_to_dicts(self, path):

		# pass

	def add_prompt_decode(self, qa_dict, predicted_span, previous_qa_dict):

		prompted_context = previous_qa_dict['context']
		prompt = "<" + str(previous_qa_dict['turn_id']) + ">"

		qa_dict['context'] = prompted_context[:predicted_span[0]] + prompt + " " + prompted_context[predicted_span[0]:predicted_span[1]] + " " + prompt + prompted_context[predicted_span[1]:]


		# Update the answer span in accordiance to the prompted context.

		try:
			prompt_positions_original = previous_qa_dict['prompt_positions_original']
			predicted_span_original = self.calc_original_span_positions(prompt_positions_original, predicted_span)
			prompt_positions_original.append((predicted_span_original[0],len(prompt)+1,'start'))
			prompt_positions_original.append((predicted_span_original[1],len(prompt)+1,'end'))

			prompt_positions = previous_qa_dict['prompt_positions']
			prompt_positions = self.updated_prompt_positions(prompt_positions,predicted_span,prompt)

			
		except KeyError:  # When turn_id = 1, the predicted span is based on the original context.
			prompt_positions_original = [(predicted_span[0],len(prompt)+1,'start'),(predicted_span[1],len(prompt)+1,'end')]
			prompt_positions = [(predicted_span[0],predicted_span[0]+4),(predicted_span[1]+4,predicted_span[1]+8)]

		
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

		prompt_positions_original.sort(key = lambda x: x[0])
		start_position = 0
		end_position = 0


		for num in range(len(prompt_positions_original)):

			lengths_inserted_front = sum(list(p[1] for p in prompt_positions_original[:num]))
			prompt_position = prompt_positions_original[num][0] + lengths_inserted_front
			prompt_length = prompt_positions_original[num][1]

			if prompt_position < predicted_span[0]:
				predicted_span = (predicted_span[0] - prompt_length, predicted_span[1])
			if prompt_position < predicted_span[1]:
				predicted_span = (predicted_span[0], predicted_span[1] - prompt_length)

		return predicted_span



	def updated_prompt_positions(self, prompt_positions,predicted_span,prompt):

		new_prompt_positions = list()

		start_prompt = (predicted_span[0],predicted_span[0]+len(prompt)+1)
		end_prompt = (predicted_span[1]+len(prompt)+1,predicted_span[1]+2*(len(prompt)+1))

		for p in prompt_positions:
			if p[0]>predicted_span[1]:
				p = (p[0]+2*(len(prompt)+1),p[1]+2*(len(prompt)+1))
			elif p[0]>predicted_span[0]:
				p = (p[0]+len(prompt)+1,p[1]+len(prompt)+1)
			new_prompt_positions.append(p)

		new_prompt_positions.append(start_prompt)
		new_prompt_positions.append(end_prompt)

		return new_prompt_positions




if __name__ == "__main__":

	coqa_train_path = 'dataset/coqa-train.json'
	coqa_dev_path = 'dataset/coqa-dev.json'
	quac_train_path = 'dataset/quac-train.json'
	quac_dev_path = 'dataset/quac-dev.json'

	data = qa_data(coqa_dev_path)

	qa_dicts = data.qa_dicts

	for qa_list in qa_dicts:

		qa_dict = qa_list[0]
		previous_qa_dict = qa_dict
		print(qa_dict)
		print()
		
		for qa_dict in qa_list[1:]:
			predicted_span = previous_qa_dict['answer_span']
			qa_dict = data.add_prompt_decode(qa_dict, predicted_span, previous_qa_dict)
			span = qa_dict['answer_span']
			prompt_positions = qa_dict['prompt_positions']
			print(qa_dict['turn_id'])
			print(qa_dict['original_answer'])
			print(qa_dict['context'][span[0]:span[1]])
			print(span)
			print(qa_dict['context'])
			for prompt in prompt_positions:
				print(qa_dict['context'][prompt[0]:prompt[1]])
			print()
			previous_qa_dict = qa_dict

		break