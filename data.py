from add_prompt import *


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