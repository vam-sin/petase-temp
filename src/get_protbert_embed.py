# libraries
import numpy as np
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import pandas as pd 

embedder = ProtTransBertBFDEmbedder()

def get_pb(sequences):
	embeddings = []
	for i in range(len(sequences)):
		print(i, len(sequences))
		embeddings.append(np.mean(np.asarray(embedder.embed(sequences[i])), axis=0))

	return embeddings

if __name__ == '__main__':
	# get sequence list
	ds = pd.read_csv('../processed_data/seq_temp_all.csv')
	ds = ds[ds['sequences'].notna()] # 20 sequences not available
	seq = list(ds["sequences"])
	# seq = ["A", "M", "AM"]
	embed = np.asarray(get_pb(seq))
	print(embed.shape)
	np.savez_compressed('../processed_data/PB_all.npz', embed)