import sys

word_dict = {}

with open('The_man_of_Property.txt', 'r') as fd:
	for line in fd:
		ss = line.strip().split(' ')
		for w in ss:
			if w.strip() not in word_dict:
				word_dict[w.strip()] = 1
			else:
				word_dict[w.strip()] += 1

for k, v in word_dict.items():
	print(k, v)

word_list = [ (k, int(v)) for k, v in word_dict.items() ]
word_list = sorted(word_list, key = lambda x:x[1], reverse=True)

print(word_list[0])

for a in word_list:
	print(a)
	print('\t'.join([a[0], str(a[1])]))
