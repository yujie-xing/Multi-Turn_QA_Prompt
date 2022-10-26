from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2ForTokenClassification
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import GPT2Model
from transformers import Trainer


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
		if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
			classifier_dropout = config.classifier_dropout
		elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
			classifier_dropout = config.hidden_dropout
		else:
			classifier_dropout = 0.1
		self.dropout = nn.Dropout(classifier_dropout)
		
		### self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		self.classifier = nn.Linear(config.hidden_size, self.num_labels)  # Only need 2 labels: start and end

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
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		start_labels: Optional[torch.LongTensor] = None,
		end_labels: Optional[torch.LongTensor] = None,
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
		logits = self.classifier(hidden_states)

		### Split start and end logits. 0 for start and 1 for end
		start_logits = logits[:,:,0]
		end_logits = logits[:,:,1]

		loss = None
		if start_labels is not None and end_labels is not None:
			loss_fct = CrossEntropyLoss()
			### loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			start_loss = loss_fct(start_logits, start_labels)
			end_loss = loss_fct(end_logits, end_labels)
			loss = (start_loss + end_loss) / 2

		if not return_dict:
			output = (logits,) + transformer_outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return TokenClassifierOutput(
			loss=loss,
			## logits=logits
			logits=(start_logits,end_logits),
			hidden_states=transformer_outputs.hidden_states,
			attentions=transformer_outputs.attentions,
		)
		

class QATrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		# forward pass
		outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'], start_labels=inputs['start_labels'], end_labels=inputs['end_labels'])
		loss = outputs.get("loss")
		# compute custom loss (suppose one has 3 labels with different weights)
		return (loss, outputs) if return_outputs else loss





if __name__ == "__main__":
	pass


