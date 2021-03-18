#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, MaxPooling1D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow import keras

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

# threshold 70
y_70 = []

for i in y:
    if i >= 70.0:
        y_70.append(1)
    else:
        y_70.append(0)

y_70 = np.asarray(y_70)
y_70 = to_categorical(y_70)

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

X, y_70 = shuffle(X, y_70, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X, y_70, test_size=0.2, random_state=42)
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
bs = 2

# test and train generators
train_gen = bm_generator(X_train, y_train, bs)
test_gen = bm_generator(X_test, y_test, bs)

# keras nn model
input_ = Input(shape = (21, max_length,))
x = Conv1D(512, (3), padding="same", activation = "relu")(input_)
# x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Conv1D(512, (3), padding="same", activation = "relu")(x)
# x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
# x = Dropout(0.4)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
# x = Dropout(0.4)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x) 
out = Dense(2, activation = 'softmax')(x)
model = Model(input_, out)

print(model.summary())

# adam optimizer
opt = keras.optimizers.Adam(learning_rate = 1e-4)

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# specificity metric
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy', sensitivity])

# callbacks
mcp_save = keras.callbacks.ModelCheckpoint('../saved_models/cnn_oh.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training
num_epochs = 100
with tf.device('/gpu:0'): # use gpu
    history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/bs), verbose=1, validation_data = test_gen, validation_steps = math.ceil(len(X_test)/bs), workers = 0, shuffle = False, callbacks = callbacks_list, class_weight = {0: 1, 1: 4})

    X_test = integer_encoding(X_test)
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
    X_test = to_categorical(X_test)
    X_testT = []
    for arr in X_test:
        X_testT.append(arr.T)
    X_test = np.asarray(X_testT)
    y_pred = model.predict(X_test)
    # print(y_test, y_pred)
    # Metrics
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    print(matrix)

    print("F1 Score")
    print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted'))
'''
Results:
'''