from typing import Optional, Tuple, Union
from os import path
from collections import defaultdict
import json
import pickle
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2ForTokenClassification
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoTokenizer, GPT2Model, AutoModelForQuestionAnswering
from transformers import Trainer
from data import decode_data, decode_data_longformer, DataArguments



GPT2_INPUTS_DOCSTRING = r"""
	Args:
		input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
			`input_ids_length` = `sequence_length` if `past_key_values` is `None` else
			`past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
			sequence tokens in the vocabulary.

			If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
			`input_ids`.

			Indices can be obtained using [`GPT2Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.

			[What are input IDs?](../glossary#input-ids)
		past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
			Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
			`past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
			their past given to this model should not be passed as `input_ids` as they have already been computed.
		attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.

			If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
			`past_key_values`. In other words, the `attention_mask` always has to have the length:
			`len(past_key_values) + len(input_ids)`

			[What are attention masks?](../glossary#attention-mask)
		token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
			Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
			1]`:

			- 0 corresponds to a *sentence A* token,
			- 1 corresponds to a *sentence B* token.

			[What are token type IDs?](../glossary#token-type-ids)
		position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
			config.max_position_embeddings - 1]`.

			[What are position IDs?](../glossary#position-ids)
		head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
			Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

			- 1 indicates the head is **not masked**,
			- 0 indicates the head is **masked**.

		inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
			is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
			model's internal embedding lookup matrix.

			If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
			`past_key_values`).
		use_cache (`bool`, *optional*):
			If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
			`past_key_values`).
		output_attentions (`bool`, *optional*):
			Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
			tensors for more detail.
		output_hidden_states (`bool`, *optional*):
			Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
			more detail.
		return_dict (`bool`, *optional*):
			Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class GPT2forQA(GPT2ForTokenClassification):

	def __init__(self, config):
		super().__init__(config)
		### self.num_labels = config.num_labels
		self.num_labels = 2  # Only need 2 labels: start and end

		self.transformer = GPT2Model(config)
#		if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
#			classifier_dropout = config.classifier_dropout
#		elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
#			classifier_dropout = config.hidden_dropout
#		else:
#			classifier_dropout = 0.1
#		self.dropout = nn.Dropout(classifier_dropout)
		
		### self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		self.classifier = nn.Linear(config.hidden_size, self.num_labels)  # Only need 2 labels: start and end
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.pad_token_id = config.pad_token_id

		# Model parallel
		self.model_parallel = False
		self.device_map = None

		# Initialize weights and apply final processing
		self.post_init()

	@add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
	# fmt: off
	## The following seems to be for sample codes. Too lazy to copy paste all the docs so just commented them out
	# @add_code_sample_docstrings(
	# 	processor_class=_TOKENIZER_FOR_DOC,
	# 	checkpoint="brad1141/gpt2-finetuned-comp2",
	# 	output_type=TokenClassifierOutput,
	# 	config_class=_CONFIG_FOR_DOC,
	# 	expected_loss=0.25,
	# 	expected_output=["Lead", "Lead", "Lead", "Position", "Lead", "Lead", "Lead", "Lead", "Lead", "Lead", "Lead", "Lead"],
	# )
	# # fmt: on

	
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		target_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		start_positions: Optional[torch.LongTensor] = None,
		end_positions: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, TokenClassifierOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		transformer_outputs = self.transformer(
			input_ids,
			past_key_values=past_key_values,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = transformer_outputs[0]
		hidden_states = self.dropout(hidden_states)
		qa_logits = None
		lm_logits = None
		loss = None

		if target_ids is not None:
			lm_logits = self.lm_head(hidden_states)
			loss_fct_lm = CrossEntropyLoss(ignore_index=self.pad_token_id)
			lm_loss = loss_fct_lm(lm_logits.view(-1, lm_logits.size(-1)), target_ids.view(-1))

		if start_positions is not None and end_positions is not None:
			qa_logits = self.classifier(hidden_states)
			### Split start and end logits. 0 for start and 1 for end
			start_logits = qa_logits[:,:,0]
			end_logits = qa_logits[:,:,1]
			loss_fct = CrossEntropyLoss()
			### loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			qa_loss = (start_loss + end_loss) / 2

		if qa_logits is not None and lm_logits is not None:
			loss = (lm_loss + qa_loss) / 2
		elif lm_logits is not None and qa_logits is None:
			loss = lm_loss
		elif lm_logits is None and qa_logits is not None:
			loss = qa_loss

		# For decoding:
		if target_ids is None and start_positions is None and end_positions is None:
			lm_logits = self.lm_head(hidden_states)
			qa_logits = self.classifier(hidden_states)

		if not return_dict:
			output = (logits,) + transformer_outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return TokenClassifierOutput(
			loss=loss,
			## logits=logits
			logits=(qa_logits,lm_logits),
			hidden_states=transformer_outputs.hidden_states,
			attentions=transformer_outputs.attentions,
		)
		

class QATrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		# forward pass
		outputs = model(**inputs)
		loss = outputs.get("loss")
		# compute custom loss (suppose one has 3 labels with different weights)
		return (loss, outputs) if return_outputs else loss

class QALongformerTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		# forward pass
		outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], start_positions=inputs['start_positions'], end_positions=inputs['end_positions'])
		loss = outputs.get("loss")
		# compute custom loss (suppose one has 3 labels with different weights)
		return (loss, outputs) if return_outputs else loss


class generate_QA():

	def __init__(self, args, dataargs):

		self.args = args
		self.dataargs = dataargs
		with open(path.join(self.dataargs.model_path,self.dataargs.dataargs_file),'rb') as model_dataargs:
			self.model_dataargs = pickle.load(model_dataargs)
		self.data_processor = decode_data()
		print(self.model_dataargs)

		# Initialize tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(dataargs.tokenizer_path)
		# self.sharp_id = self.tokenizer.vocab["<"]
		# self.space_sharp_id = self.tokenizer.vocab["Ġ<"]
		# Initialize model
		model = GPT2forQA.from_pretrained(dataargs.model_path)
		self.predictor = QATrainer(model, self.args)
		self.output_path = path.join(self.args.output_dir,self.dataargs.output_file)

		# with open(self.output_path, 'w') as output:
		# 	output.write("")
			
		print("Output to {}.\n".format(self.output_path))


	def decode(self):

		# Handles models with QA support. Use evaluate() for models without QA support.

		if "coqa" in self.dataargs.test_path:
			qa_dicts = self.data_processor.data_to_dicts_coqa(self.dataargs.test_path)
		elif "quac" in self.dataargs.test_path:
			qa_dicts = self.data_processor.data_to_dicts_quac(self.dataargs.test_path)
		else:
			raise Exception("Not coqa nor quac.")

		answer_list = list()

		for qa_list in qa_dicts:

			span = None
			previous_qa_dict = None

			original_context = qa_list[0]['context']

			for qa_dict in qa_list:
				qa_dict = self.data_processor.add_prompt_decode(qa_dict, span, previous_qa_dict)
				tokenized_qa_dict = self.data_processor.preprocess([qa_dict], self.tokenizer, self.model_dataargs.only_lm, self.model_dataargs.only_qa, self.model_dataargs.instruction, self.dataargs.max_length, self.dataargs.doc_stride)
				qa_logits,lm_logits = self.predictor.predict(tokenized_qa_dict).predictions
				span, score, lm_answer_start, have_span = self.data_processor.postprocess([qa_dict], tokenized_qa_dict, qa_logits, lm_logits, self.model_dataargs.only_lm, self.model_dataargs.only_qa, self.dataargs.search_size, self.dataargs.max_answer_length)
				span_original = self.data_processor.calc_original_span_positions(qa_dict['prompt_positions_original'],span)

				if not self.model_dataargs.only_qa:
					lm_answer = self.data_processor.postprocess_lm(self.predictor, tokenized_qa_dict, lm_answer_start, have_span, self.tokenizer.eos_token_id, self.dataargs.max_answer_length)
				else:
					lm_answer = [self.tokenizer.pad_token_id]

				previous_qa_dict = qa_dict
				
				answer_list.append({"id": qa_dict['id'], "turn_id": qa_dict['turn_id'], "question": qa_dict['question'], "gold": qa_dict['original_answer'], "span": (span_original[0],span_original[1]), "human": qa_dict['human_answer'], "qa_answer": original_context[span_original[0]:span_original[1]], "lm_answer": self.tokenizer.decode(lm_answer)})

#				self.write(qa_dict, span, score, span_original, original_context)

		if not self.model_dataargs.only_lm:
			self.write_coqa_qa_answer(answer_list)
			# self.write_quac_qa_answer(answer_list)
		if not self.model_dataargs.only_qa:
			self.write_coqa_lm_answer(answer_list)
		# self.write_quac_answer(answer_list)
		print("Decoding finished!")


	def evaluate(self):   ## For evaluation of prompted test dataset.

		# Handles models with and without QA support.

		if "prompt" in self.dataargs.test_path:
			test_dataset = self.data_processor.load(self.dataargs.test_path)
		else:
			test_dataset = self.data_processor.data_to_dicts_coqa(self.dataargs.test_path)
			test_dataset = [qa_dict for qa_list in test_dataset for qa_dict in qa_list]
			if not self.model_dataargs.only_lm:
				raise Exception("Use decode().")
		
		answer_list = list()

		for qa_dict in test_dataset:
			tokenized_qa_dict = self.data_processor.preprocess([qa_dict], self.tokenizer, self.model_dataargs.only_lm, self.model_dataargs.only_qa, self.model_dataargs.instruction, self.dataargs.max_length, self.dataargs.doc_stride)
			qa_logits,lm_logits = self.predictor.predict(tokenized_qa_dict).predictions
			span, score, lm_answer_start, have_span = self.data_processor.postprocess([qa_dict], tokenized_qa_dict, qa_logits, lm_logits, self.model_dataargs.only_lm, self.model_dataargs.only_qa, self.dataargs.search_size, self.dataargs.max_answer_length)

			if not self.model_dataargs.only_lm:
				span_original = self.data_processor.calc_original_span_positions(qa_dict['prompt_positions_original'],span)
			else:
				span_original = (-1,-1)

			if not self.model_dataargs.only_qa:
				lm_answer = self.data_processor.postprocess_lm(self.predictor, tokenized_qa_dict, lm_answer_start, have_span, self.tokenizer.eos_token_id, self.dataargs.max_answer_length)
			else:
				lm_answer = [self.tokenizer.pad_token_id]
			
			if "prompt" in self.dataargs.test_path:
				answer_list.append({"id": qa_dict['id'], "turn_id": qa_dict['turn_id'], "question": qa_dict['question'], "gold": qa_dict['original_answer'], "span": (span_original[0],span_original[1]), "human": qa_dict['human_answer'], "qa_answer": qa_dict['original_context'][span_original[0]:span_original[1]], "lm_answer": self.tokenizer.decode(lm_answer)})
			else:
				answer_list.append({"id": qa_dict['id'], "turn_id": qa_dict['turn_id'], "question": qa_dict['question'], "gold": qa_dict['answer'], "span": (span_original[0],span_original[1]), "human": qa_dict['human_answer'], "qa_answer": qa_dict['context'][span_original[0]:span_original[1]], "lm_answer": self.tokenizer.decode(lm_answer)})

#			self.write(qa_dict, spans[i], scores[i], spans_original[i], qa_dict['original_context'])
		
		if not self.model_dataargs.only_lm:
			self.write_coqa_qa_answer(answer_list)
			# self.write_quac_qa_answer(answer_list)
		if not self.model_dataargs.only_qa:
			self.write_coqa_lm_answer(answer_list)


		print("Evaluation finished!")
			
			
			
	def write_coqa_qa_answer(self, answer_list):
		for i in range(len(answer_list)):
			answer_list[i]["answer"] = answer_list[i]["qa_answer"]
		with open(self.output_path+'.json','w') as f:
			json.dump(answer_list,f)

	def write_coqa_lm_answer(self, answer_list):
		for i in range(len(answer_list)):
			answer_list[i]["answer"] = answer_list[i]["lm_answer"]
		with open(self.output_path+'_lm.json','w') as f:
			json.dump(answer_list,f)


	def write_quac_qa_answer(self, answer_list):

		with open(self.output_path,'w') as f:
			f.write("")

		quac_answer_list = {"best_span_str":[], "qid":[], "gold":[], "span":[], "yesno":[], "followup":[]}
		id0 = answer_list[0]['id']

		for answers in answer_list:
			id = answers["id"]
			if id != id0:
				with open(self.output_path,'a') as f:
					json.dump(quac_answer_list,f)
					f.write('\n')
				quac_answer_list = {"best_span_str":[], "qid":[], "gold":[], "span":[], "yesno":[], "followup":[]}
				id0 = id
			turn_id = answers["turn_id"]
			qid = id + "_q#" + str(turn_id-1)
			gold = answers["gold"]
			answer = answers["answer"]
			span = answers["span"]

			if span[0] == -1 and span[1] == -1:
				quac_answer_list["best_span_str"].append("CANNOTANSWER")

			else:
				quac_answer_list["best_span_str"].append(answer)
			quac_answer_list["gold"].append(gold)
			quac_answer_list["qid"].append(qid)
			quac_answer_list["span"].append(span)
			quac_answer_list["yesno"].append("x")
			quac_answer_list["followup"].append("y")


		with open(self.output_path,'a') as f:
			json.dump(quac_answer_list,f)
			f.write('\n')




		


	def write(self, qa_dict, span, score, span_original, original_context):
		with open(self.output_path, 'a') as output:
			output.write(qa_dict['id'])
			output.write("\n")
			output.write(str(qa_dict['turn_id']))
			output.write("\n")
			output.write(qa_dict['question'])
			output.write("\n")
			output.write(str(span[0]))
			output.write(",")
			output.write(str(span[1]))
			output.write("\n")
			output.write(qa_dict['context'][span[0]:span[1]])
			output.write("\n")
			output.write(str(span_original[0]))
			output.write(",")
			output.write(str(span_original[1]))
			output.write("\n")
			output.write(original_context[span_original[0]:span_original[1]])
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


class generate_QA_longformer(generate_QA):

	def __init__(self, args, dataargs):

		self.args = args
		self.dataargs = dataargs
		self.data_processor = decode_data_longformer()

		# Initialize tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(dataargs.tokenizer_path)
		# Initialize model
		model = AutoModelForQuestionAnswering.from_pretrained(dataargs.model_path)
		self.predictor = QALongformerTrainer(model, self.args)
		self.output_path = path.join(self.args.output_dir,self.dataargs.output_file)

		with open(self.output_path, 'w') as output:
			output.write("")
			
		print("Output to {}.\n".format(self.output_path))




if __name__ == "__main__":
	pass


