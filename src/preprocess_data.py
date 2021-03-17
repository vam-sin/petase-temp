# libraries
from Bio import SeqIO
import pandas as pd
import re
from get_fasta import get_seq

# data
# sequences
# print(record_dict)
# for i in record_dict:
# 	print(i.name.split('|')[1])

def get_sequence(seq_name):
	record_dict = SeqIO.parse("../Tm_files/proteins.faa", "fasta")
	seq_ret = 'No'
	for i in record_dict:
		if str(seq_name) in i.name.split('|'):
			# print(i, seq_ret)
			seq_ret = i.seq
			break

	# print(seq_ret)
	if seq_ret == 'No':
		print("NOOOOOOOOOOOO")
		try:
			seq_return = get_seq(seq_name)
			return seq_return
		except:
			pass

	return seq_ret

# temperatures
f = open('../Tm_files/Tm_all', 'r')

sequences = []
temperatures = []

count = 0
for line in f:
	if line != '\n':
		seq_name = line.replace('\n','').split(' ')[0]
		temp = line.replace('\n','').split(' ')[1]
		print(count, seq_name, temp)
		# print(seq_name, temp)
		sequences.append(get_sequence(seq_name))
		temperatures.append(temp)
		count += 1

# print(count, non_seq)

print(len(sequences), len(temperatures))

# print(sequences)

df = pd.DataFrame(list(zip(sequences, temperatures)), columns =['sequences', 'temperatures'])
df.to_csv('../Tm_files/seq_temp_all.csv') 
