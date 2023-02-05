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
	dataargs = DataArguments(test_path='dataset/coqa-dev.json', tokenizer_path='prompt_QA_gen/tokenizer', model_path='prompt_QA_gen', max_answer_length = 10, decode=True)
	print("Test Mode")
	print(args)
	print(dataargs)


#============================
QA_model = generate_QA(args, dataargs)

if dataargs.evaluate:
	QA_model.evaluate()
elif dataargs.decode:
	QA_model.decode()