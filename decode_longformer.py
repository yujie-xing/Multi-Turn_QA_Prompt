from GPT2forQA import *
from data import DataArguments
from transformers import HfArgumentParser, TrainingArguments

# Parser -> args
try:
	parser = HfArgumentParser((DataArguments,TrainingArguments))
	dataargs, args = parser.parse_args_into_dataclasses()
	if dataargs.batch_size != 0:
		args.per_device_eval_batch_size = dataargs.batch_size
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
QA_model = generate_QA_longformer(args, dataargs)

if dataargs.evaluate:
	QA_model.evaluate()
elif dataargs.decode:
	QA_model.decode()
