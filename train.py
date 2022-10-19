from transformers import Trainer

class CustomTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		# forward pass
		outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'], start_labels=inputs['start_labels'], end_labels=inputs['end_labels'])
		loss = outputs.get("loss")
		# compute custom loss (suppose one has 3 labels with different weights)
		return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":

	import json
	import transformers
	from transformers import HfArgumentParser, TrainingArguments
	from transformers import AutoTokenizer, AutoModel, AutoConfig
	from transformers import Trainer
	from datasets import Dataset
	from data import load_train_dev_data, prepare_train_features
	from GPT2forQA import *


	parser = HfArgumentParser((TrainingArguments))
	parser.add_argument('--model_checkpoint', type=str, default='gpt2', help="pretrained model's name")
	parser.add_argument('--batch_size', type=int, default=0, help='a simpler way to change batch size')

	try:
		args = parser.parse_args_into_dataclasses()
	except:
		saving_folder = "gpt2-finetuned-test"
		args = TrainingArguments(saving_folder)
		args.model_checkpoint = 'gpt2'
		args.batch_size = 2

	if args.batch_size != 0:
		args.per_device_train_batch_size = args.batch_size
		args.per_device_eval_batch_size = args.batch_size

	print(args)

	# Load dataset
	train_dataset_path = "dataset/coqa-train-prompted.json"
	dev_dataset_path = "dataset/coqa-dev-prompted.json"
	train_dataset = load_train_dev_data(train_dataset_path)
	dev_dataset = load_train_dev_data(dev_dataset_path)

	# Tokenizer

	tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
	special_tokens_dict = {'pad_token': '<|paddingtokencustomized|>'}
	tokenizer.add_special_tokens(special_tokens_dict)

	max_length = 1020
	doc_stride = 128

	tokenized_train_dataset = prepare_train_features(train_dataset[:5], tokenizer, max_length, doc_stride)
	tokenized_dev_dataset = prepare_train_features(dev_dataset[:5], tokenizer, max_length, doc_stride)
	tokenized_train_dataset = Dataset.from_dict(tokenized_train_dataset)
	tokenized_dev_dataset = Dataset.from_dict(tokenized_dev_dataset)


	config = AutoConfig.from_pretrained("gpt2")
	model = GPT2forQA(config)
	model.transformer = AutoModel.from_pretrained("gpt2")
	model.resize_token_embeddings(len(tokenizer))

	for param in model.transformer.parameters():
		param.requires_grad = False

	trainer = CustomTrainer(
		model,
		args,
		train_dataset=tokenized_train_dataset,
		eval_dataset=tokenized_dev_dataset
	)

	trainer.train()

	trainer.save_model()
	tokenizer.save_pretrained(saving_folder+"/tokenizer")
	with open (saving_folder+"/log.json",'w') as f:
		json.dump(trainer.state.log_history,f)
