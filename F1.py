
from os import path
import json
from collections import Counter

with open("output/answers_decode.json",'r') as f:
	answers = json.load(f)

def f1(original, answer):
	common = Counter(original) & Counter(answer)
	num_same = sum(common.values())
	if len(original) == 0 or len(answer) == 0:
		# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
		return int(original == original)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(original)
	recall = 1.0 * num_same / len(answer)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

scores = list()
for turn in answers:
	original = turn['gold'].strip().split()
	answer = turn['answer'].strip().split()
	score = f1(original, answer)
	scores.append(score)

s = sum(scores)/len(scores)
print(s)
# for i, ss in enumerate(scores):
# 	if ss < 0.5:
# 		print(answers[i])



