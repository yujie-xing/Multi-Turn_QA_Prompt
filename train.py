
from os import path, mkdir, environ
import json, pickle
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer, AutoModel, AutoConfig
from data import DataArguments, train_data
from GPT2forQA import *

#===============================
# Parser -> args
try:
	parser = HfArgumentParser((DataArguments,TrainingArguments))
	dataargs, args = parser.parse_args_into_dataclasses()
	if dataargs.batch_size != 0:
		args.per_device_train_batch_size = dataargs.batch_size
		args.per_device_eval_batch_size = dataargs.batch_size
	print(args)
	print(dataargs)
except:  ## Only for test
	print("\n======================\n")
	args = TrainingArguments(output_dir="test")
	try:
		mkdir("test")
	except:
		pass
	dataargs = DataArguments(train_path="dataset/coqa-train-prompted.json",dev_path="dataset/coqa-dev-prompted.json",test_path=None)
	print("Test Mode")
	print(args)
	print(dataargs)
	

#===============================

# Initialize data processor

data_processor = train_data()
train_dataset = data_processor.load(dataargs.train_path)
dev_dataset = data_processor.load(dataargs.dev_path)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(dataargs.tokenizer_path)
sharp_id = tokenizer.vocab["<"]
space_sharp_id = tokenizer.vocab["Ġ<"]
tokenizer.pad_token = tokenizer.eos_token
# special_tokens_dict = {'pad_token': '<|paddingtokencustomized|>'}
# tokenizer.add_special_tokens(special_tokens_dict)

# Tokenize dataset & prepared labels
tokenized_train_dataset = data_processor.preprocess(train_dataset, tokenizer, dataargs.max_length, dataargs.doc_stride, sharp_id, space_sharp_id)
tokenized_dev_dataset = data_processor.preprocess(dev_dataset, tokenizer, dataargs.max_length, dataargs.doc_stride, sharp_id, space_sharp_id)

print("data tokenized")

#===============================
# Initialize config
config = AutoConfig.from_pretrained(dataargs.model_path)
config.pad_token_id = tokenizer.pad_token_id
config.eos_token_id = tokenizer.eos_token_id

# Initialize model
model = GPT2forQA(config).from_pretrained(dataargs.model_path)
model.resize_token_embeddings(len(tokenizer))

# Fix main weights in the model
#for param in model.transformer.parameters():
#	param.requires_grad = False
#model.transformer.wte.weight[tokenizer.pad_token_id]=0

print("model initialized")

#===============================
# Initialize Trainer
trainer = QATrainer(
	model,
	args,
	train_dataset=tokenized_train_dataset,
	eval_dataset=tokenized_dev_dataset,
)

# Train
trainer.train()

#===============================
# Save model & tokenizer
trainer.save_model()
tokenizer.save_pretrained(path.join(args.output_dir,"tokenizer"))

# Save args
with open (path.join(args.output_dir,"args.pkl"),'wb') as f:
	pickle.dump(args,f)
with open (path.join(args.output_dir,"dataargs.pkl"),'wb') as f:
	pickle.dump(dataargs,f)
print("\nArgs file saved in {} and {}\n".format(path.join(args.output_dir,"args.pkl"),path.join(args.output_dir,"dataargs.pkl")))

# Save log
with open (path.join(args.output_dir,"log.json"),'w') as f:
	json.dump(trainer.state.log_history,f)
print("Log history file saved in {}".format(path.join(args.output_dir,"log.json")))
