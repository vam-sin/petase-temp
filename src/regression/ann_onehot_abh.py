#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
from sklearn.preprocessing import OneHotEncoder 
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, MaxPooling1D, Bidirectional, LSTM
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import regularizers
from keras import backend as K
import keras
from keras_self_attention import SeqSelfAttention

# GPU config for Vamsi's Laptop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# dataset import 
ds = pd.read_csv('../../processed_data/seq_temp_abh.csv')

# 5,515 unique classes -> 5481 unique classes with more than one datapoint
X = list(ds["sequences"])
y = list(ds["temperatures"])
y = np.array(y, dtype='f')

max_length = 837

# maximum sequence length is 837 residues in the ds

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
	char_dict = {}
	for index, val in enumerate(codes):
		char_dict[val] = index+1

	return char_dict

char_dict = create_dict(codes)

print(char_dict)
print("Dict Length:", len(char_dict))

def integer_encoding(data):
	"""
	- Encodes code sequence to integer values.
	- 20 common amino acids are taken into consideration
	and rest 4 are categorized as 0.
	"""
	encode_list = []
	for row in data:
		row_encode = []	
		for code in row: 
			row_encode.append(char_dict.get(code, 0))
		encode_list.append(np.array(row_encode))

	return encode_list
  
# y process
print("Loaded X and y")

X, y = shuffle(X, y, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Conducted Train-Test Split")

# generator
def bm_generator(X_t, y_t, batch_size):
    val = 0

    while True:
        X_batch = []
        y_batch = []

        for j in range(batch_size):

            if val == len(X_t):
                val = 0

            X_batch.append(X_t[val])
            y_batch.append(y_t[val])
            val += 1

        X_batch = integer_encoding(X_batch)
        X_batch = pad_sequences(X_batch, maxlen=max_length, padding='post', truncating='post')
        X_batch = to_categorical(X_batch)
        X_batchT = []
        for arr in X_batch:
        	X_batchT.append(arr.T)
        X_batch = np.asarray(X_batchT)
        # print(X_batch.shape)
        y_batch = np.asarray(y_batch)

        yield X_batch, y_batch

# batch size
bs = 32

# test and train generators
train_gen = bm_generator(X_train, y_train, bs)
test_gen = bm_generator(X_test, y_test, bs)

# keras nn model
input_ = Input(shape = (21, max_length,))
x = Flatten()(input_)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x) 
out = Dense(1, activation = 'linear')(x)
model = Model(input_, out)

print(model.summary())

# adam optimizer
opt = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = "adam", loss = "mean_squared_error", metrics=['mse'])

# callbacks
mcp_save = keras.callbacks.ModelCheckpoint('../saved_models/ann_oh.h5', save_best_only=True, monitor='val_mse', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training
num_epochs = 500
with tf.device('/gpu:0'): # use gpu
    history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = len(X_train), verbose=1, validation_data = test_gen, validation_steps = len(X_test), workers = 0, shuffle = False, callbacks = callbacks_list)

'''
Results:
MSE: 443.39963 (BS 1)
MSE: 339.28094 (BS 2)
MSE: 270.67087 (BS 4)
MSE: 247.76332 (BS 8)
MSE: 243.97980 (BS 16)
'''