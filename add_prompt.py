import json


class qa_add_prompt():

	def __init__(self, path):

		self.prompted_dicts = self.add_prompt(self.data_to_dicts(path))

	def data_to_dicts(self, path):

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


	def add_prompt(self, qa_dicts):

		prompted_dicts = list()

		for article in qa_dicts:

			prompt_positions = list()
			
			for num in range(len(article)):
				qa_dict = article[num]
				span = qa_dict['answer_span']
				turn_id = qa_dict['turn_id']
				prompt = "<" + str(turn_id) + ">"
				
				if num == 0:
					context = qa_dict['context']
					qa_dict['original_answer'] = qa_dict['answer']
					qa_dict['prompt_positions'] = list() # Record where the start of a prompt is in the prompted context
					
					prompted_dicts.append(qa_dict)

					if span[0] != -1:
						prompted_context = context[:span[0]] + prompt + " " + context[span[0]:span[1]] + " " + prompt + context[span[1]:]
						prompt_positions.append((span[0],len(prompt)+1,'start'))
						prompt_positions.append((span[1],len(prompt)+1,'end')) # -> (start, prompt_length) and (end, prompt_length)
					else:
						prompted_context = context


				else:
					qa_dict['context'] = prompted_context[:]
					qa_dict['original_answer'] = qa_dict['answer']
					## context[prompt_start, prompt_end] =  prompt
					qa_dict['prompt_positions'] = self.calc_new_prompt_positions(prompt_positions)

					if span[0] != -1:

						start_diff = 0
						end_diff = 0
						for prompt_position in prompt_positions:
							if prompt_position[0] <= span[0]:  ## <= makes <1> <2> <3> instead of <3> <2> <1>
								start_diff += prompt_position[1]

							if prompt_position[0] < span[1]:  ## < makes <3> <2> <1> instead of <1> <2> <3>
								end_diff += prompt_position[1]
						
						new_span = (span[0]+start_diff, span[1]+end_diff)
						qa_dict['answer_span'] = new_span
						qa_dict['answer'] = prompted_context[new_span[0]:new_span[1]]

						prompted_dicts.append(qa_dict)

						prompted_context = prompted_context[:new_span[0]] + prompt + " " + prompted_context[new_span[0]:new_span[1]] + " " + prompt + prompted_context[new_span[1]:]
						prompt_positions.append((span[0],len(prompt)+1,'start'))
						prompt_positions.append((span[1],len(prompt)+1,'end')) # -> (start, prompt_length) and (end, prompt_length) of the original context
					
					else:
						prompted_dicts.append(qa_dict)

		return prompted_dicts



	def calc_new_prompt_positions(self, prompt_positions):

		new_prompt_positions = list()

		if len(prompt_positions) == 0:
			return new_prompt_positions

		prompt_positions.sort(key = lambda x: x[0])
		prompt = prompt_positions[0]
		# if prompt[2] == 'end':
		# 	prompt = (prompt[0]+1, prompt[0]+prompt[1], prompt[2])
		# elif prompt[2] == 'start':
		prompt = (prompt[0], prompt[0]+prompt[1], prompt[2])
		new_prompt_positions.append(prompt)


		for num in range(1,len(prompt_positions)):
			prompt =  prompt_positions[num]
			for previous_prompt in prompt_positions[:num]:
				prompt = (prompt[0]+previous_prompt[1], prompt[1],prompt[2])
			
			# if prompt[2] == 'end':
			# 	prompt = (prompt[0]+1, prompt[0]+prompt[1], prompt[2])
			# elif prompt[2] == 'start':
			prompt = (prompt[0], prompt[0]+prompt[1], prompt[2])
			new_prompt_positions.append(prompt)

		return new_prompt_positions




	def prompted_data(self):
		return self.prompted_dicts

	def write_prompted_data(self, output_path):
		with open(output_path,'w') as file:
			json.dump(self.prompted_dicts, file)


if __name__ == "__main__":
	path = "dataset/coqa-dev.json"
	prompts = qa_add_prompt(path)
	prompted_data = prompts.prompted_data()
	
	id = prompted_data[0]['id']
	for qa_dict in prompted_data:
		if qa_dict['id'] == id:
			span = qa_dict['answer_span']
			print("Turn id is: {}".format(qa_dict['turn_id']))
			print()
			print("Use prompt positions to print prompts:")
			for prompt_position in qa_dict['prompt_positions']:
				print(qa_dict['context'][prompt_position[0]:prompt_position[1]])
			print()
			print("The original answer is: {}".format(qa_dict['original_answer']))
			print("The answer is: {}".format(qa_dict['answer']))
			print("Use the answer span to print the answer: {}".format(qa_dict['context'][span[0]:span[1]]))
			print("The answer span is {} to {}".format(span[0],span[1]))
			print()
			print(qa_dict['context'])
			print()
			print()