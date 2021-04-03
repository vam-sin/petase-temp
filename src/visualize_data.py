# libraries
import pandas as pd 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# Apply the default theme

# dataset import 
ds = pd.read_csv('../processed_data/seq_temp_abh.csv')
ds = ds[ds['sequences'].notna()] # 20 sequences not available
ds.to_csv('processed_temp_all.csv')

X = np.load('../processed_data/PB_abh.npz')['arr_0']
X = np.array(X, dtype = 'f')
y = list(ds["temperatures"])
y = np.array(y, dtype='f')

# threshold 55
thresh = 55.0
y_55 = []

for i in y:
    if i >= thresh:
        y_55.append(1)
    else:
        y_55.append(0)

y_55 = np.asarray(y_55)

unique, counts = np.unique(y_55, return_counts=True)
print(unique, counts)

print(X, y_55)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X = tsne.fit_transform(X)

scatter_x = X[:,0]
scatter_y = X[:,1]
group = y_55
cdict = {0: 'teal', 1: 'purple'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()