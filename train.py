from os import path, mkdir, environ
import json, pickle
import torch
import transformers, optuna
from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer, AutoModel, AutoConfig
from data import DataArguments, train_data
from GPT2forQA import *

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 6e-5),
        }

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
	dataargs = DataArguments(train_path="dataset/coqa-train.json",dev_path="dataset/coqa-dev.json",test_path=None, instruction="Answer the question based on the given passage.", only_lm=True)
	print("Test Mode\n\n")
	print(args)
	print(dataargs)
	

#===============================

# Initialize data processor

data_processor = train_data()

if dataargs.only_lm:
	train_dataset  = data_processor.data_to_dicts_coqa(dataargs.train_path)
	dev_dataset  = data_processor.data_to_dicts_coqa(dataargs.dev_path)
	train_dataset = [qa_dict for qa_list in train_dataset for qa_dict in qa_list]
	dev_dataset = [qa_dict for qa_list in dev_dataset for qa_dict in qa_list]
else:
	train_dataset = data_processor.load(dataargs.train_path)
	dev_dataset = data_processor.load(dataargs.dev_path)
	

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(dataargs.tokenizer_path)
# sharp_id = tokenizer.vocab["<"]
# space_sharp_id = tokenizer.vocab["Ġ<"]
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {'pad_token': '<|padding|>'}
tokenizer.add_special_tokens(special_tokens_dict)

# Tokenize dataset & prepared labels
tokenized_train_dataset = data_processor.preprocess(train_dataset, tokenizer, dataargs.only_lm, dataargs.only_qa, dataargs.instruction, dataargs.max_length, dataargs.doc_stride)
tokenized_dev_dataset = data_processor.preprocess(dev_dataset, tokenizer, dataargs.only_lm, dataargs.only_qa, dataargs.instruction, dataargs.max_length, dataargs.doc_stride)

print(tokenized_train_dataset.features)
print()

#===============================
# Initialize config
config = AutoConfig.from_pretrained(dataargs.model_path)
config.pad_token_id = tokenizer.pad_token_id
config.eos_token_id = tokenizer.eos_token_id
config.vocab_size += 1

# Initialize model
if dataargs.fine_tune:
	model = GPT2forQA.from_pretrained(dataargs.model_path)
else:
	pre_trained_model = AutoModel.from_pretrained(dataargs.model_path)
	model = GPT2forQA(config)
	model.transformer = pre_trained_model
	model.resize_token_embeddings(len(tokenizer))

def model_init(trial):
	return model

# Fix main weights in the model
#for param in model.transformer.parameters():
#	param.requires_grad = False
#model.transformer.wte.weight[tokenizer.pad_token_id]=0

print("model initialized")

#===============================
# Initialize Trainer
trainer = QATrainer(
	args=args,
	train_dataset=tokenized_train_dataset,
	eval_dataset=tokenized_dev_dataset,
	model_init=model_init)

# Train
# trainer.train()
trainer.hyperparameter_search(direction="minimize", backend="optuna", hp_space=my_hp_space, n_trials=10)

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
